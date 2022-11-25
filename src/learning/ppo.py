"""Program that trains an agent using PPO."""

import numpy as np
import argparse
import random
import json
import pathlib
import copy
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import torch as th
from torch import nn

import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.distributions import Distribution, CategoricalDistribution

from azure_rl.azure_client import AzureClient

import cProfile
from pstats import SortKey
from enum import Enum

import gym

from gym_ceo.envs.observation import ObservationFactory, Observation
from gym_ceo.envs.ceo_player_env import CEOPlayerEnv
from gym_ceo.envs.actions import ActionEnum
from CEO.cards.eventlistener import EventListenerInterface, PrintAllEventListener
from learning.ppo_agents import PPOBehavior, load_ppo

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
        # Tensorboard will find & display metrics from the `SCALARS` tab
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


class CustomActorCriticPolicy(ActorCriticPolicy):
    _invalid_actions_layer: th.nn.Linear

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        observation_factory: ObservationFactory = None,
        device=None,
        *args,
        **kwargs,
    ):
        assert observation_factory is not None

        self.net_arch = net_arch
        self.observation_factory = observation_factory

        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # device=device,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False

        self._create_invalid_actions_layer(observation_factory, device)

    def _build(self, lr_schedule: Schedule) -> None:
        """Override the method from the base class."""
        super(CustomActorCriticPolicy, self)._build(lr_schedule)

        # Overwrite the default action_net to also adjust the probabilities based on
        # the valid actions.

    def forward(
        self, obs: th.Tensor, deterministic: bool = False
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi, features)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1,) + self.action_space.shape)
        return actions, values, log_prob

    def evaluate_actions(
        self, obs: th.Tensor, actions: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs:
        :param actions:
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(latent_pi, obs)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()

    def get_distribution(self, obs: th.Tensor) -> Distribution:
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        features = self.extract_features(obs)
        latent_pi = self.mlp_extractor.forward_actor(features)
        return self._get_action_dist_from_latent(latent_pi, features)

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor, obs: th.Tensor) -> Distribution:
        """
        Override the method from the base class.
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        assert isinstance(self.action_dist, CategoricalDistribution)

        raw_mean_actions = self.action_net(latent_pi)
        mean_actions = self._adjust_action_logits(obs, raw_mean_actions)

        # Here mean_actions are the logits before the softmax
        return self.action_dist.proba_distribution(action_logits=mean_actions)

    def _adjust_action_logits(self, features: th.Tensor, raw_action_logits: th.Tensor):
        """Adjust the outputs of the policy network based on which actions are invalid."""

        # print("_adjust_action_logits features", features.dtype, features)
        # print("_adjust_action_logits raw_action_logits", raw_action_logits.dtype, raw_action_logits)
        # print("invalid layer", self._invalid_actions_layer.dtype)
        invalid_actions_float = self._invalid_actions_layer(features)
        invalid_actions = invalid_actions_float > 0.0

        # self._raw_action_logits = raw_action_logits
        # self._invalid_actions = invalid_actions
        if False:
            self._valid_actions = self._get_invalid_actions_obj.get_valid_actions(obs)

        # TODO: Figure out how to set the invalid action minimum on a per-action basis.
        if False:
            print("")
            print("----------------------------")
            print("")
            print("action_logits", action_logits)
            print("action_logits len", len(action_logits))
            mint = action_logits.min(1, keepdim=True)
            print("mint true", mint)
            print("mint true len", len(mint.values))
            # mint = action_logits.min(1, keepdim=False)
            # print("mint false", mint)

            print("invalid", invalid_actions)
            test_new_action_logits = action_logits.clone()
            test_new_action_logits[invalid_actions] = mint.values
            print(test_new_action_logits)

        # Adjust the logits so that the invalid actions have small probabilities.
        invalid_logit_value = raw_action_logits.min() - 10.0
        if invalid_logit_value >= -10.0:
            invalid_logit_value = -10.0

        # print( "raw_action_logits", raw_action_logits.shape, raw_action_logits.dtype, raw_action_logits)
        # print("invalid_actions", invalid_actions.shape, invalid_actions.dtype, invalid_actions)
        new_action_logits = raw_action_logits.clone()
        new_action_logits[invalid_actions] = invalid_logit_value

        # self._invalid_logit_value = invalid_logit_value
        # self._new_action_logits = new_action_logits

        if False:
            print("")
            print("----------------------------")
            print("")
            print("action_logits", action_logits)
            print("new_action_logits", new_action_logits)
            print("invalid_actions", invalid_actions)
            print("obs", obs)
            print("invalid_logit_value", invalid_logit_value)
            # traceback.print_stack(file=sys.stdout)

        return new_action_logits

    def _create_invalid_actions_layer(self, observation_factory: ObservationFactory, device: str):
        self._observation_factory = observation_factory

        (begin, end) = self._observation_factory.get_valid_action_range()
        feature_size = self._observation_factory.observation_dimension
        action_size = end - begin

        weights = []
        bias = []
        for a in range(action_size):
            weights.append([])

            for f in range(feature_size):
                val = -1.0 if a == f - begin else 0.0
                weights[a].append(val)

        for a in range(action_size):
            bias.append(1.0)

        self._invalid_actions_layer = th.nn.Linear(
            feature_size,
            action_size,
            device=device,  # dtype=th.float32
        )
        self._invalid_actions_layer.weight = th.nn.parameter.Parameter(
            data=th.tensor(weights, dtype=th.float, device=device).float(), requires_grad=False
        )
        self._invalid_actions_layer.bias = th.nn.parameter.Parameter(
            data=th.tensor(bias, dtype=th.float, device=device).float(), requires_grad=False
        )


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

    def train(self, observation_factory, eval_log_path: str, params: dict, do_log: bool):
        # Load the parameters
        learning_rate = params["learning_rate"] if "learning_rate" in params else None
        if learning_rate is None:
            learning_rate = 3e-4
            learning_rate = 3e-2  # Very bad
            learning_rate = 3e-3  # Bad
            learning_rate = 3e-5  # Better than 3e-4
            params["learning_rate"] = learning_rate
            print("Using default learning_rate of", learning_rate)

        gae_lambda = params["gae_lambda"] if "gae_lambda" in params else None
        if gae_lambda is None:
            gae_lambda = 0.95
            params["gae_lambda"] = gae_lambda
            print("Using default gae_lambda of", gae_lambda)

        n_steps_per_update = (
            params["n_steps_per_update"] if "n_steps_per_update" in params else None
        )
        if n_steps_per_update is None:
            n_steps_per_update = 32
            params["n_steps_per_update"] = n_steps_per_update
            print("Using default n_steps_per_update of", n_steps_per_update)

        batch_size = params["batch_size"] if "batch_size" in params else None
        if batch_size is None:
            batch_size = 32
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
        policy_kwargs["observation_factory"] = observation_factory

        print("net_arch", policy_kwargs["net_arch"])

        # Train the agent
        self._ppo = PPO(
            CustomActorCriticPolicy,
            self._env,
            n_steps=n_steps_per_update,
            batch_size=batch_size,
            learning_rate=learning_rate,
            gae_lambda=gae_lambda,
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
            eval_log_path=eval_log_path,
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


def make_env(env_number, env_args: dict):
    def _init():
        random.seed(env_number)
        env = CEOPlayerEnv(**env_args)

        return env

    return _init


def process_ppo_agents(ppo_agents: list[str], device: str, num_players: int) -> tuple[dict, dict]:
    if not ppo_agents:
        return None

    agents = dict()
    agent_descs = dict()
    for ppo_dir in ppo_agents:
        ppo, params = load_ppo(ppo_dir, device)

        agent_num_players = params["env_args"]["num_players"]
        seat_num = params["env_args"]["seat_number"]

        if num_players != agent_num_players:
            raise Exception(
                (
                    f"The agent {ppo_dir} is for a game with {agent_num_players} players, "
                    f"but the game has {num_players} players."
                )
            )

        behavior = PPOBehavior(
            seat_num=seat_num, num_players=num_players, ppo=ppo, params=params, device=device
        )

        agents[seat_num] = behavior
        agent_descs[seat_num] = ppo_dir

    return agents, agent_descs


# Main function
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
        "--ppo-agents",
        dest="ppo_agents",
        type=str,
        nargs="*",
        default=[],
        help="Specifies directories containing trained PPO agents to play other seats in the game.",
    )

    args = parser.parse_args()

    if not args.seat_number:
        args.seat_number = 0
        print(f"Using default {args.seat_number} seat for the agent.")

    if not args.num_players:
        args.num_players = 6
        print(f"Using default {args.num_players} seat for the agent.")

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

    custom_behaviors, custom_behavior_descs = process_ppo_agents(
        args.ppo_agents, device=args.device, num_players=args.num_players
    )
    print("main", type(custom_behaviors))

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
        env = SubprocVecEnv([make_env(i, env_args) for i in range(args.parallel_env_count)])

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
    if args.device:
        train_params["device"] = args.device

    # Save all parameters to the eval_log directory
    eval_log_path = "eval_log/" + args.name
    param_file = eval_log_path + "/params.json"

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

    if args.profile:
        print("Running with profiling")
        cProfile.run(
            "learning.train(observation_factory, train_params, do_log)", sort=SortKey.CUMULATIVE
        )
    else:
        learning.train(observation_factory, eval_log_path, train_params, do_log)

    # Save the agent in a pickle file.
    learning.save("seatceo_ppo")


if __name__ == "__main__":
    main()
