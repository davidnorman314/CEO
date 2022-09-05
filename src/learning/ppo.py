"""Program that trains an agent using PPO."""

import gym
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import math

import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam

from azure_rl.azure_client import AzureClient

import cProfile
from pstats import SortKey
from enum import Enum

from gym_ceo.envs.seat_ceo_env import SeatCEOEnv, CEOActionSpace
from gym_ceo.envs.actions import ActionEnum
from CEO.cards.eventlistener import EventListenerInterface, PrintAllEventListener

from typing import Any, Dict, List, Optional, Sequence, TextIO, Tuple, Union


class PPOCallback(BaseCallback):
    _hyperparameters: dict
    _do_log: bool

    def __init__(
        self,
        hyperparameters: dict[str, Union[bool, str, float, int, None]],
        do_log: bool,
        verbose=0,
    ):
        super(PPOCallback, self).__init__(verbose)

        self._hyperparameters = hyperparameters
        self._do_log = do_log

        pass

    def _on_training_start(self) -> None:
        # define the metrics that will appear in the `HPARAMS` Tensorboard tab by referencing their tag
        # Tensorbaord will find & display metrics from the `SCALARS` tab
        metric_dict = {
            "eval/mean_reward": 1,
            "rollout/ep_rew_mean": 1,
            "rollout/ep_rew_mean": 1,
            # "train/value_loss": 1,
        }
        self.logger.record(
            "hparams",
            HParam(self._hyperparameters, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self):
        if self._do_log and self.num_timesteps % 10000 == 0:
            print("Timestep ", self.num_timesteps)
        return True


class PPOLearning:
    """
    Class that wraps PPO to do learning for CEO.
    """

    _name: str
    _total_steps: int
    _ppo: stable_baselines3.PPO
    _env: gym.Env
    _eval_env: gym.Env

    def __init__(
        self, name: str, env: gym.Env, eval_env: gym.Env, total_steps=1000000000, **kwargs
    ):
        self._name = name
        self._env = env
        self._eval_env = eval_env
        self._total_steps = total_steps

        if "azure_client" in kwargs:
            self._azure_client = kwargs["azure_client"]
            del kwargs["azure_client"]
        else:
            self._azure_client = None

    def train(self, params: dict, do_log: bool):
        # Load the parameters
        learning_rate = params["learning_rate"] if "learning_rate" in params else None
        if learning_rate is None:
            learning_rate = 3e-4
            learning_rate = 3e-2  # Very bad
            learning_rate = 3e-3  # Bad
            learning_rate = 3e-5  # Better than 3e-4
            params["learning_rate"] = learning_rate
            print("Using default learning_rate of", learning_rate)

        n_steps_per_update = (
            params["n_steps_per_update"] if "n_steps_per_update" in params else None
        )
        if n_steps_per_update is None:
            n_steps_per_update = 256
            params["n_steps_per_update"] = n_steps_per_update
            print("Using default n_steps_per_update of", n_steps_per_update)

        batch_size = params["batch_size"] if "batch_size" in params else None
        if batch_size is None:
            batch_size = 256
            params["batch_size"] = batch_size
            print("Using default batch_size of", batch_size)

        pi_net_arch = params["pi_net_arch"] if "pi_net_arch" in params else None
        if pi_net_arch is None:
            pi_net_arch = "256 256"
            params["pi_net_arch"] = pi_net_arch
            print("Using default pi_net_arch of", pi_net_arch)

        vf_net_arch = params["vf_net_arch"] if "vf_net_arch" in params else None
        if vf_net_arch is None:
            vf_net_arch = "256 256"
            params["vf_net_arch"] = vf_net_arch
            print("Using default vf_net_arch of", vf_net_arch)

        device = params["device"] if "device" in params else "cuda"

        print("Training with", self._total_steps, "total steps")

        # Log the start of training to Azure, if necessary.
        if self._azure_client:
            self._azure_client.start_training(
                "ppo",
                self._base_env.action_space_type,
                self._env.full_env.num_players,
                self._env.full_env.seat_number,
                params,
                self._env.feature_defs,
            )

        tensorboard_log = "tensorboard_log"
        verbose = 1
        verbose = 0

        policy_kwargs = dict()
        policy_kwargs["net_arch"] = [
            dict(pi=self.str_to_net_arch(pi_net_arch), vf=self.str_to_net_arch(vf_net_arch))
        ]

        print("net_arch", policy_kwargs["net_arch"])

        # Train the agent
        self._ppo = PPO(
            "MlpPolicy",
            self._env,
            n_steps=n_steps_per_update,
            batch_size=batch_size,
            learning_rate=learning_rate,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
        )

        callback = PPOCallback(params, do_log)

        self._ppo.learn(
            self._total_steps,
            eval_env=self._eval_env,
            eval_freq=50000,
            n_eval_episodes=10000,
            eval_log_path="eval_log/" + self._name,
            tb_log_name=self._name,
            callback=callback,
        )

        if self._azure_client:
            self._azure_client.end_training()

        return

    def save(self, file: str):
        self._ppo.save(file)

    def str_to_net_arch(self, net_arch_str):
        toks = net_arch_str.split(" ")
        return list([int(tok) for tok in toks])


# Main function
def main():
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

    args = parser.parse_args()

    kwargs = dict()

    if args.total_steps:
        kwargs["total_steps"] = args.total_steps
    if args.azure:
        kwargs["azure_client"] = AzureClient()

    do_log = False
    if args.log:
        do_log = args.log

    random.seed(0)
    listener = PrintAllEventListener()
    listener = EventListenerInterface()

    obs_kwargs = {"include_valid_actions": True}

    env = SeatCEOEnv(
        listener=listener,
        action_space_type="all_card",
        reward_includes_cards_left=True,
        obs_kwargs=obs_kwargs,
    )

    # Use the usual reward for eval_env
    eval_env = SeatCEOEnv(
        listener=listener,
        action_space_type="all_card",
        reward_includes_cards_left=False,
        obs_kwargs=obs_kwargs,
    )

    learning = PPOLearning(args.name, env, eval_env, **kwargs)

    params = dict()
    if args.n_steps_per_update:
        params["n_steps_per_update"] = args.n_steps_per_update
    if args.batch_size:
        params["batch_size"] = args.batch_size
    if args.learning_rate:
        params["learning_rate"] = args.learning_rate
    if args.pi_net_arch:
        params["pi_net_arch"] = args.pi_net_arch
    if args.vf_net_arch:
        params["vf_net_arch"] = args.vf_net_arch
    if args.device:
        params["device"] = args.device

    if args.profile:
        print("Running with profiling")
        cProfile.run("learning.train(params, do_log)", sort=SortKey.CUMULATIVE)
    else:
        learning.train(params, do_log)

    # Save the agent in a pickle file.
    learning.save("seatceo_ppo")


if __name__ == "__main__":
    main()
