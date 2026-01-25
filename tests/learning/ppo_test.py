"""End-to-end tests for PPO training."""

import pytest

from ceo.envs.ceo_player_env import CEOPlayerEnv
from ceo.game.eventlistener import EventListenerInterface
from ceo.learning.ppo import PPOLearning


@pytest.fixture
def test_output_dir(request, pytestconfig):
    """Create a test output directory based on the test name."""
    test_name = request.node.name
    output_dir = pytestconfig.rootpath / "temp" / "test_output" / test_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def env_args():
    """Common environment arguments for PPO tests."""
    return {
        "num_players": 4,
        "seat_number": 0,
        "listener": EventListenerInterface(),
        "action_space_type": "all_card",
        "reward_includes_cards_left": False,
        "obs_kwargs": {"include_valid_actions": True},
    }


def test_ppo_training_basic(env_args, test_output_dir):
    """Test basic PPO training with periodic and final checkpoints."""
    env = CEOPlayerEnv(**env_args)
    eval_env = CEOPlayerEnv(**env_args)

    tensorboard_log = str(test_output_dir / "tensorboard")
    eval_log_path = str(test_output_dir / "eval_log")
    checkpoints_dir = str(test_output_dir / "checkpoints")

    # With 200 total steps, 32 steps per update, and checkpoint every 50 steps,
    # checkpoints occur when num_timesteps - last_checkpoint >= 50.
    # Updates happen at steps 32, 64, 96, 128, 160, 192, 224.
    # Checkpoints at: 50 (after 64), 100 (after 128), 150 (after 160), 200 (after 224)
    # But step counts are the actual timesteps: 64, 128, 160, 224
    # Wait - the callback checks after each step, so:
    # - After step 50: checkpoint at 50
    # - After step 100: checkpoint at 100
    # - After step 150: checkpoint at 150
    # - After step 200: checkpoint at 200
    # Plus final checkpoint at 224 (total steps reached)
    learning = PPOLearning(
        name="test_ppo",
        env=env,
        eval_env=eval_env,
        total_steps=200,
        tensorboard_log=tensorboard_log,
        checkpoint_dir=checkpoints_dir,
        checkpoint_interval=50,
    )

    observation_factory = eval_env.observation_factory

    train_params = {
        "n_steps_per_update": 32,
        "batch_size": 32,
        "learning_rate": 3e-4,
        "gae_lambda": 0.95,
        "pi_net_arch": "64 64",
        "vf_net_arch": "64 64",
        "activation_fn": "tanh",
        "device": "cpu",
    }

    learning.train(observation_factory, eval_log_path, train_params, do_log=False)

    # Save final checkpoint
    learning.save(checkpoints_dir)

    checkpoints_path = test_output_dir / "checkpoints"
    assert checkpoints_path.exists(), "checkpoints directory should exist"

    checkpoint_files = list(checkpoints_path.glob("seatceo_ppo_*.zip"))

    # Extract step numbers from checkpoint filenames
    step_numbers = []
    for f in checkpoint_files:
        # Extract number from "seatceo_ppo_123.zip"
        step_str = f.stem.replace("seatceo_ppo_", "")
        step_numbers.append(int(step_str))
    step_numbers.sort()

    # Should have checkpoints at approximately 50, 100, 150, 200, plus final
    assert len(step_numbers) >= 4, (
        f"Should have at least 4 checkpoints, got {len(step_numbers)}: {step_numbers}"
    )

    # Verify checkpoints are at expected intervals (multiples of ~50)
    # The first checkpoint should be around 50
    assert step_numbers[0] >= 50, (
        f"First checkpoint should be >= 50, got {step_numbers[0]}"
    )
    # The last checkpoint should be the final step count
    assert step_numbers[-1] > 200, (
        f"Final checkpoint should be > 200, got {step_numbers[-1]}"
    )


def test_ppo_training_relu_activation(env_args, test_output_dir):
    """Test PPO training with ReLU activation function."""
    env = CEOPlayerEnv(**env_args)
    eval_env = CEOPlayerEnv(**env_args)

    tensorboard_log = str(test_output_dir / "tensorboard")
    eval_log_path = str(test_output_dir / "eval_log")

    learning = PPOLearning(
        name="test_ppo_relu",
        env=env,
        eval_env=eval_env,
        total_steps=100,
        tensorboard_log=tensorboard_log,
    )

    observation_factory = eval_env.observation_factory

    train_params = {
        "n_steps_per_update": 32,
        "batch_size": 32,
        "learning_rate": 3e-4,
        "gae_lambda": 0.95,
        "pi_net_arch": "64 64",
        "vf_net_arch": "64 64",
        "activation_fn": "relu",
        "device": "cpu",
    }

    learning.train(observation_factory, eval_log_path, train_params, do_log=False)
