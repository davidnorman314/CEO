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
    """Test basic PPO training runs without errors and saves checkpoint correctly."""
    env = CEOPlayerEnv(**env_args)
    eval_env = CEOPlayerEnv(**env_args)

    tensorboard_log = str(test_output_dir / "tensorboard")
    eval_log_path = str(test_output_dir / "eval_log")

    learning = PPOLearning(
        name="test_ppo",
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
        "activation_fn": "tanh",
        "device": "cpu",
    }

    learning.train(observation_factory, eval_log_path, train_params, do_log=False)

    # Save checkpoint and verify it's saved correctly
    learning.save(str(test_output_dir))

    checkpoints_dir = test_output_dir / "checkpoints"
    assert checkpoints_dir.exists(), "checkpoints directory should exist"

    checkpoint_files = list(checkpoints_dir.glob("seatceo_ppo_*.zip"))
    assert len(checkpoint_files) == 1, "Should have exactly one checkpoint file"
    assert checkpoint_files[0].name.endswith(".zip")


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
