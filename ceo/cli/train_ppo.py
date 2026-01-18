"""CLI for PPO training."""

import argparse
import copy
import cProfile
import json
import pathlib
import random
from pstats import SortKey

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from ceo.azure_rl.azure_client import AzureClient
from ceo.envs.ceo_player_env import CEOPlayerEnv
from ceo.game.eventlistener import EventListenerInterface, PrintAllEventListener
from ceo.learning.ppo import PPOLearning
from ceo.learning.ppo_agents import process_ppo_agents


def main():
    print("In main")

    parser = argparse.ArgumentParser(description="Do learning")
    parser.add_argument(
        "--profile",
        dest="profile",
        action="store_const",
        const=True,
        default=False,
        help="Do profiling.",
    )
    parser.add_argument(
        "--log",
        dest="log",
        action="store_const",
        const=True,
        default=False,
        help="Do logging.",
    )
    parser.add_argument(
        "--name",
        dest="name",
        type=str,
        required=True,
        help="The name of the run. Used for eval and tensorboard logging.",
    )
    parser.add_argument(
        "--parallel-env-count",
        dest="parallel_env_count",
        type=int,
        default=None,
        help="The number of parallel environments to run in parallel.",
    )
    parser.add_argument(
        "--total-steps",
        dest="total_steps",
        type=int,
        default=None,
        help="The steps to use in training",
    )
    parser.add_argument(
        "--n-steps-per-update",
        dest="n_steps_per_update",
        type=int,
        default=None,
        help="The number of steps per neural network update",
    )
    parser.add_argument(
        "--learning-rate",
        dest="learning_rate",
        type=float,
        default=None,
        help="The learning rate",
    )
    parser.add_argument(
        "--gae-lambda",
        dest="gae_lambda",
        type=float,
        default=None,
        help="The gae lambda value",
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=None,
        help="The batch size",
    )
    parser.add_argument(
        "--pi-net-arch",
        dest="pi_net_arch",
        type=str,
        default=None,
        help="The policy network architecture",
    )
    parser.add_argument(
        "--vf-net-arch",
        dest="vf_net_arch",
        type=str,
        default=None,
        help="The value function network architecture",
    )
    parser.add_argument(
        "--activation-fn",
        dest="activation_fn",
        type=str,
        default=None,
        help=(
            "The neural network activation function network architecture, "
            "e.g., relu or tanh. Optional."
        ),
    )
    parser.add_argument(
        "--device",
        dest="device",
        type=str,
        default=None,
        help="The CUDA device to use, e.g., cuda or cuda:0",
    )
    parser.add_argument(
        "--azure",
        dest="azure",
        action="store_const",
        const=True,
        default=False,
        help="Save agent and log information to azure blob storage.",
    )
    parser.add_argument(
        "--seat-number",
        dest="seat_number",
        type=int,
        default=None,
        help="The seat number for the agent.",
    )
    parser.add_argument(
        "--num-players",
        dest="num_players",
        type=int,
        default=None,
        help="The number of players in the game.",
    )
    parser.add_argument(
        "--continue-training",
        dest="continue_training",
        action="store_const",
        const=True,
        default=False,
        help="Continue training the previously saved agent.",
    )
    parser.add_argument(
        "--ppo-agents",
        dest="ppo_agents",
        type=str,
        nargs="*",
        default=[],
        help=(
            "Specifies directories containing trained PPO agents "
            "to play other seats in the game."
        ),
    )

    args = parser.parse_args()

    learning_kwargs = dict()

    if args.total_steps:
        learning_kwargs["total_steps"] = args.total_steps
    if args.azure:
        learning_kwargs["azure_client"] = AzureClient()

    do_log = False
    if args.log:
        do_log = args.log

    random.seed(0)
    listener = PrintAllEventListener()
    listener = EventListenerInterface()

    obs_kwargs = {"include_valid_actions": True}

    # Set up the parameters file location
    eval_log_path = "eval_log/" + args.name
    param_file = eval_log_path + "/params.json"

    # If we are continuing a previous training, then load the environment args from
    # the saved parameters.
    if args.continue_training:
        if args.num_players:
            raise Exception(
                "Can't specify --num-players when using --continue-training"
            )

        if args.seat_number:
            raise Exception(
                "Can't specify --seat-number when using --continue-training"
            )

        with open(param_file) as data_file:
            prev_params = json.load(data_file)

        args.num_players = prev_params["env_args"]["num_players"]
        args.seat_number = prev_params["env_args"]["seat_number"]

        print(
            f"Loading previous num_players {args.num_players} "
            f"and seat_number {args.seat_number}"
        )

        assert args.num_players is not None
        assert args.seat_number is not None

        prev_custom_behaviors = prev_params["env_args"].get("custom_behaviors", None)
    else:
        # Handle defaults for starting a new training
        if not args.seat_number:
            args.seat_number = 0
            print(f"Using default {args.seat_number} seat for the agent.")

        if not args.num_players:
            args.num_players = 6
            print(f"Using default {args.num_players} seat for the agent.")

        prev_custom_behaviors = None

    # Set up custom behaviors
    custom_behaviors, custom_behavior_descs = process_ppo_agents(
        args.ppo_agents, device=args.device, num_players=args.num_players
    )

    # If continuing training, validate that the custom behaviors match the previous
    # custom behaviors
    if (
        args.continue_training
        and prev_custom_behaviors is None
        and custom_behavior_descs is not None
    ):
        raise Exception(
            "The saved agent did not use custom behaviors, "
            "but they were specified on the command line."
        )
    elif prev_custom_behaviors is not None and custom_behavior_descs is None:
        raise Exception(
            "The saved agent did used custom behaviors, "
            "but they were not specified on the command line."
        )
    elif (
        prev_custom_behaviors is not None
        and custom_behavior_descs is not None
        and prev_custom_behaviors != custom_behavior_descs
    ):
        raise Exception(
            f"The saved agent did used custom behaviors {prev_custom_behaviors}, "
            "but they don't match the command line custom "
            f"behaviors {custom_behavior_descs}."
        )

    # Check that the custom behaviors don't include the seat being trained.
    if custom_behaviors is not None and args.seat_number in custom_behaviors:
        raise Exception(
            f"The seat {args.seat_number} has a custom behavior, "
            "but that is the seat being trained."
        )

    # Create the environment.
    env_args = {
        "num_players": args.num_players,
        "seat_number": args.seat_number,
        "listener": listener,
        "action_space_type": "all_card",
        "reward_includes_cards_left": False,
        "custom_behaviors": custom_behaviors,
        "obs_kwargs": obs_kwargs,
    }

    if args.parallel_env_count is None:
        env = CEOPlayerEnv(**env_args)
    else:
        env = make_vec_env(
            CEOPlayerEnv, n_envs=args.parallel_env_count, env_kwargs=env_args
        )

    # Use the usual reward for eval_env
    eval_env = CEOPlayerEnv(
        num_players=args.num_players,
        seat_number=args.seat_number,
        listener=listener,
        action_space_type="all_card",
        custom_behaviors=custom_behaviors,
        reward_includes_cards_left=False,
        obs_kwargs=obs_kwargs,
    )

    observation_factory = eval_env.observation_factory

    learning = PPOLearning(args.name, env, eval_env, **learning_kwargs)

    train_params = dict()
    if args.n_steps_per_update:
        train_params["n_steps_per_update"] = args.n_steps_per_update
    if args.batch_size:
        train_params["batch_size"] = args.batch_size
    if args.learning_rate:
        train_params["learning_rate"] = args.learning_rate
    if args.gae_lambda:
        train_params["gae_lambda"] = args.gae_lambda
    if args.pi_net_arch:
        train_params["pi_net_arch"] = args.pi_net_arch
    if args.vf_net_arch:
        train_params["vf_net_arch"] = args.vf_net_arch
    if args.activation_fn:
        train_params["activation_fn"] = args.activation_fn
    if args.device:
        train_params["device"] = args.device

    if not args.continue_training:
        # Save all parameters to the eval_log directory
        save_params = dict()
        save_params["learning_kwargs"] = learning_kwargs
        save_params["train_params"] = train_params

        save_params["env_args"] = copy.copy(env_args)
        del save_params["env_args"]["listener"]
        save_params["env_args"]["custom_behaviors"] = custom_behavior_descs

        eval_log_path_obj = pathlib.Path(eval_log_path)
        if not eval_log_path_obj.is_dir():
            eval_log_path_obj.mkdir(parents=True)

        with open(param_file, "w") as data_file:
            json.dump(save_params, data_file, indent=4, sort_keys=True)
    else:
        ppo = PPO.load(
            eval_log_path + "/best_model.zip",
            device=args.device,
            print_system_info=True,
            env=env,
        )
        train_params["continue_ppo"] = ppo

    if args.profile:
        print("Running with profiling")
        cProfile.run(
            "learning.train(observation_factory, train_params, do_log)",
            sort=SortKey.CUMULATIVE,
        )
    else:
        learning.train(observation_factory, eval_log_path, train_params, do_log)

    # Save the agent in a pickle file.
    learning.save("seatceo_ppo")


if __name__ == "__main__":
    main()
