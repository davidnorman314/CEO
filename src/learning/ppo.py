"""Program that trains an agent using PPO."""

import gym
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import math

import stable_baselines3
from azure_rl.azure_client import AzureClient

import cProfile
from pstats import SortKey
from enum import Enum

from gym_ceo.envs.seat_ceo_env import SeatCEOEnv, CEOActionSpace
from gym_ceo.envs.actions import ActionEnum
from CEO.cards.eventlistener import EventListenerInterface, PrintAllEventListener

from stable_baselines3 import PPO


class PPOLearning:
    """
    Class that wraps PPO to do learning.
    """

    _train_episodes: int
    _ppo: stable_baselines3.PPO
    _env: gym.Env

    def __init__(self, env: gym.Env, train_episodes=100000, **kwargs):
        self._env = env
        self._train_episodes = train_episodes

        if "azure_client" in kwargs:
            self._azure_client = kwargs["azure_client"]
            del kwargs["azure_client"]
        else:
            self._azure_client = None

    def train(self, params: dict, do_log: bool):
        # Validate the parameters
        if False:
            discount_factor = params["discount_factor"]
            if discount_factor is None:
                raise ValueError("The parameter discount_factor is missing")

            epsilon = params["epsilon"]
            if epsilon is None:
                raise ValueError("The parameter epsilon is missing")

            max_epsilon = params["max_epsilon"]
            if max_epsilon is None:
                raise ValueError("The parameter max_epsilon is missing")

            min_epsilon = params["min_epsilon"]
            if min_epsilon is None:
                raise ValueError("The parameter min_epsilon is missing")

            decay = params["decay"]
            if decay is None:
                raise ValueError("The parameter decay is missing")

            alpha_type_str = params["alpha_type"]
            if alpha_type_str is None:
                raise ValueError("The parameter alpha_type_str is missing")

            if alpha_type_str == "state_visit_count":
                alpha_type = self.AlphaType.STATE_VISIT_COUNT
            elif alpha_type_str == "constant":
                alpha_type = self.AlphaType.CONSTANT
            elif alpha_type_str == "episode_count":
                alpha_type = self.AlphaType.EPISODE_COUNT
            else:
                raise ValueError("Invalid alpha_type: " + alpha_type_str)

        print("Training with", self._train_episodes, "episodes")

        # Log the start of training to Azure, if necessary.
        if self._azure_client:
            if False:
                params = dict()
                params["discount_factor"] = discount_factor
                params["epsilon"] = epsilon
                params["max_epsilon"] = max_epsilon
                params["min_epsilon"] = min_epsilon
                params["decay"] = decay
                params["alpha_type"] = alpha_type_str
                if alpha_exponent is not None:
                    params["alpha_exponent"] = alpha_exponent
                if alpha_constant is not None:
                    params["alpha_constant"] = alpha_constant

                self._azure_client.start_training(
                    "qlearning",
                    self._base_env.action_space_type,
                    self._env.full_env.num_players,
                    self._env.full_env.seat_number,
                    params,
                    self._env.feature_defs,
                )

        n_steps = 20
        batch_size = 20
        tensorboard_log = "tensorboard_log"
        verbose = 1
        verbose = 0
        total_steps = 10 * self._train_episodes

        policy_kwargs = dict()
        policy_kwargs["net_arch"] = [dict(pi=[64, 64], vf=[64, 64])]

        # Train the agent
        self._ppo = PPO(
            "MlpPolicy",
            self._env,
            n_steps=n_steps,
            batch_size=batch_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
        )

        self._ppo.learn(total_steps, eval_freq=1000, n_eval_episodes=1000)

        if self._azure_client:
            self._azure_client.end_training()

        return

    def save(self, file: str):
        self._ppo.save(file)


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
        "--episodes",
        dest="train_episodes",
        type=int,
        default=100000,
        help="The number of rounds to play",
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
    print(args)

    kwargs = dict()
    if args.train_episodes:
        kwargs["train_episodes"] = args.train_episodes

    if args.azure:
        kwargs["azure_client"] = AzureClient()

    kwargs["disable_agent_testing"] = True

    do_log = False
    if args.log:
        do_log = args.log

    random.seed(0)
    listener = PrintAllEventListener()
    listener = EventListenerInterface()
    env = SeatCEOEnv(listener=listener, action_space_type="all_card")

    learning = PPOLearning(env, **kwargs)

    params = dict()

    if args.profile:
        print("Running with profiling")
        cProfile.run("learning.train(params, do_log)", sort=SortKey.CUMULATIVE)
    else:
        learning.train(params, do_log)

    # Save the agent in a pickle file.
    learning.save("seatceo_ppo")


if __name__ == "__main__":
    main()
