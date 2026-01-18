"""CLI for PPO training using Hydra configuration."""

import cProfile
import pathlib
import random
from pstats import SortKey

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from ceo.azure_rl.azure_client import AzureClient
from ceo.envs.ceo_player_env import CEOPlayerEnv
from ceo.game.eventlistener import EventListenerInterface
from ceo.learning.ppo import PPOLearning
from ceo.learning.ppo_agents import process_ppo_agents


def validate_continue_training(cfg: DictConfig, prev_cfg: DictConfig) -> None:
    """Validate that continue_training config matches previous run."""
    if cfg.env.num_players != prev_cfg.env.num_players:
        raise ValueError(
            f"num_players mismatch: current={cfg.env.num_players}, "
            f"previous={prev_cfg.env.num_players}"
        )
    if cfg.env.seat_number != prev_cfg.env.seat_number:
        raise ValueError(
            f"seat_number mismatch: current={cfg.env.seat_number}, "
            f"previous={prev_cfg.env.seat_number}"
        )
    # Validate ppo_agents match
    prev_agents = prev_cfg.get("ppo_agents", [])
    if list(cfg.ppo_agents) != list(prev_agents):
        raise ValueError(
            f"ppo_agents mismatch: current={list(cfg.ppo_agents)}, "
            f"previous={list(prev_agents)}"
        )


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print("In main")
    print(OmegaConf.to_yaml(cfg))

    # Set up learning kwargs
    learning_kwargs: dict = {}
    if cfg.ppo.total_steps:
        learning_kwargs["total_steps"] = cfg.ppo.total_steps
    if cfg.azure:
        learning_kwargs["azure_client"] = AzureClient()

    random.seed(0)
    listener = EventListenerInterface()
    obs_kwargs = {"include_valid_actions": True}

    # Set up the parameters file location
    eval_log_path = f"eval_log/{cfg.name}"
    param_file = f"{eval_log_path}/params.yaml"

    # Handle continue_training
    if cfg.continue_training:
        prev_cfg = OmegaConf.load(param_file)
        validate_continue_training(cfg, prev_cfg)
        # Use previous env settings
        num_players = prev_cfg.env.num_players
        seat_number = prev_cfg.env.seat_number
        print(
            f"Loading previous num_players {num_players} "
            f"and seat_number {seat_number}"
        )
    else:
        num_players = cfg.env.num_players
        seat_number = cfg.env.seat_number

    # Set up custom behaviors from ppo_agents
    custom_behaviors, custom_behavior_descs = process_ppo_agents(
        list(cfg.ppo_agents), device=cfg.device, num_players=num_players
    )

    # Validate custom behaviors don't include the training seat
    if custom_behaviors is not None and seat_number in custom_behaviors:
        raise ValueError(
            f"Seat {seat_number} has a custom behavior, "
            "but that is the seat being trained."
        )

    # Create the environment
    env_args = {
        "num_players": num_players,
        "seat_number": seat_number,
        "listener": listener,
        "action_space_type": "all_card",
        "reward_includes_cards_left": False,
        "custom_behaviors": custom_behaviors,
        "obs_kwargs": obs_kwargs,
    }

    if cfg.env.parallel_env_count is None:
        env = CEOPlayerEnv(**env_args)
    else:
        env = make_vec_env(
            CEOPlayerEnv, n_envs=cfg.env.parallel_env_count, env_kwargs=env_args
        )

    # Create eval environment
    eval_env = CEOPlayerEnv(
        num_players=num_players,
        seat_number=seat_number,
        listener=listener,
        action_space_type="all_card",
        custom_behaviors=custom_behaviors,
        reward_includes_cards_left=False,
        obs_kwargs=obs_kwargs,
    )

    observation_factory = eval_env.observation_factory

    # Use Hydra's output directory for tensorboard logs
    hydra_output_dir = HydraConfig.get().runtime.output_dir
    learning_kwargs["tensorboard_log"] = f"{hydra_output_dir}/tensorboard"

    learning = PPOLearning(cfg.name, env, eval_env, **learning_kwargs)

    # Build train_params from config
    train_params = {
        "n_steps_per_update": cfg.ppo.n_steps_per_update,
        "batch_size": cfg.ppo.batch_size,
        "learning_rate": cfg.ppo.learning_rate,
        "gae_lambda": cfg.ppo.gae_lambda,
        "pi_net_arch": cfg.network.pi_net_arch,
        "vf_net_arch": cfg.network.vf_net_arch,
        "activation_fn": cfg.network.activation_fn,
        "device": cfg.device,
    }

    if not cfg.continue_training:
        # Save configuration to eval_log directory
        eval_log_path_obj = pathlib.Path(eval_log_path)
        if not eval_log_path_obj.is_dir():
            eval_log_path_obj.mkdir(parents=True)

        # Save full config (Hydra also saves this, but we want it in eval_log)
        OmegaConf.save(cfg, param_file)
    else:
        ppo = PPO.load(
            f"{eval_log_path}/best_model.zip",
            device=cfg.device,
            print_system_info=True,
            env=env,
        )
        train_params["continue_ppo"] = ppo

    if cfg.profile:
        print("Running with profiling")
        cProfile.run(
            "learning.train(observation_factory, eval_log_path, train_params, cfg.log)",
            sort=SortKey.CUMULATIVE,
        )
    else:
        learning.train(observation_factory, eval_log_path, train_params, cfg.log)

    # Save the agent to checkpoints directory inside hydra output dir
    learning.save(hydra_output_dir)


if __name__ == "__main__":
    main()
