"""PPO training module for CEO."""

import random
from collections.abc import Callable

import gymnasium
import stable_baselines3
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.distributions import CategoricalDistribution, Distribution
from stable_baselines3.common.logger import HParam
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
from torch import nn

from ceo.envs.ceo_player_env import CEOPlayerEnv
from ceo.envs.observation import ObservationFactory


class PPOCallback(BaseCallback):
    _hyperparameters: dict
    _do_log: bool

    def __init__(
        self,
        hyperparameters: dict[str, bool | str | float | int | None],
        do_log: bool,
        verbose=0,
    ):
        super().__init__(verbose)

        print(f"Hyperparameters {hyperparameters}")
        self._hyperparameters = hyperparameters
        self._do_log = do_log

        pass

    def _on_training_start(self) -> None:
        # define the metrics that will appear in the `HPARAMS` Tensorboard tab
        # by referencing their tag
        # Tensorboard will find & display metrics from the `SCALARS` tab
        metric_dict = {
            "eval/mean_reward": 1,
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
        observation_space: gymnasium.spaces.Space,
        action_space: gymnasium.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: list[int | dict[str, list[int]]] | None = None,
        activation_fn: type[nn.Module] = nn.Tanh,
        observation_factory: ObservationFactory = None,
        device=None,
        *args,
        **kwargs,
    ):
        assert observation_factory is not None

        self.net_arch = net_arch
        self.observation_factory = observation_factory

        super().__init__(
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
        super()._build(lr_schedule)

        # Overwrite the default action_net to also adjust the probabilities based on
        # the valid actions.

    def forward(
        self, obs: th.Tensor, deterministic: bool = False
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
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
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
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

    def _get_action_dist_from_latent(
        self, latent_pi: th.Tensor, obs: th.Tensor
    ) -> Distribution:
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
        """Adjust the outputs of the policy network based on which actions
        are invalid."""

        # print("_adjust_action_logits features", features.dtype, features)
        # print("_adjust_action_logits raw_action_logits", raw_action_logits.dtype,
        #       raw_action_logits)
        # print("invalid layer", self._invalid_actions_layer.dtype)
        invalid_actions_float = self._invalid_actions_layer(features)
        invalid_actions = invalid_actions_float > 0.0

        # Adjust the logits so that the invalid actions have small probabilities.
        invalid_logit_value = raw_action_logits.min() - 10.0
        if invalid_logit_value >= -10.0:
            invalid_logit_value = -10.0

        # print( "raw_action_logits", raw_action_logits.shape, raw_action_logits.dtype,
        #        raw_action_logits)
        # print("invalid_actions", invalid_actions.shape, invalid_actions.dtype,
        #       invalid_actions)
        new_action_logits = raw_action_logits.clone()
        new_action_logits[invalid_actions] = invalid_logit_value

        # self._invalid_logit_value = invalid_logit_value
        # self._new_action_logits = new_action_logits

        return new_action_logits

    def _create_invalid_actions_layer(
        self, observation_factory: ObservationFactory, device: str
    ):
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

        for _ in range(action_size):
            bias.append(1.0)

        self._invalid_actions_layer = th.nn.Linear(
            feature_size,
            action_size,
            device=device,  # dtype=th.float32
        )
        self._invalid_actions_layer.weight = th.nn.parameter.Parameter(
            data=th.tensor(weights, dtype=th.float, device=device).float(),
            requires_grad=False,
        )
        self._invalid_actions_layer.bias = th.nn.parameter.Parameter(
            data=th.tensor(bias, dtype=th.float, device=device).float(),
            requires_grad=False,
        )


class PPOLearning:
    """
    Class that wraps PPO to do learning for CEO.
    """

    _name: str
    _total_steps: int
    _ppo: stable_baselines3.PPO
    _env: gymnasium.Env
    _eval_env: gymnasium.Env

    def __init__(
        self,
        name: str,
        env: gymnasium.Env,
        eval_env: gymnasium.Env,
        total_steps=1000000000,
        **kwargs,
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

    def train(
        self, observation_factory, eval_log_path: str, params: dict, do_log: bool
    ):
        if "continue_ppo" in params:
            self._ppo = params["continue_ppo"]
            del params["continue_ppo"]
            reset_timesteps = False
        else:
            self._create_ppo(params, observation_factory)
            reset_timesteps = True

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

        ppo_callback = PPOCallback(params, do_log)
        eval_callback = EvalCallback(
            eval_env=self._eval_env,
            eval_freq=50000,
            n_eval_episodes=10000,
            log_path=eval_log_path,
        )
        callback = CallbackList([ppo_callback, eval_callback])

        self._ppo.learn(
            self._total_steps,
            reset_num_timesteps=reset_timesteps,
            tb_log_name=self._name,
            callback=callback,
        )

        if self._azure_client:
            self._azure_client.end_training()

        return

    def _create_ppo(self, params, observation_factory):
        # Load the parameters
        learning_rate = params.get("learning_rate", None)
        if learning_rate is None:
            learning_rate = 3e-4
            learning_rate = 3e-2  # Very bad
            learning_rate = 3e-3  # Bad
            learning_rate = 3e-5  # Better than 3e-4
            params["learning_rate"] = learning_rate
            print("Using default learning_rate of", learning_rate)

        gae_lambda = params.get("gae_lambda", None)
        if gae_lambda is None:
            gae_lambda = 0.95
            params["gae_lambda"] = gae_lambda
            print("Using default gae_lambda of", gae_lambda)

        n_steps_per_update = params.get("n_steps_per_update", None)
        if n_steps_per_update is None:
            n_steps_per_update = 32
            params["n_steps_per_update"] = n_steps_per_update
            print("Using default n_steps_per_update of", n_steps_per_update)

        batch_size = params.get("batch_size", None)
        if batch_size is None:
            batch_size = 32
            params["batch_size"] = batch_size
            print("Using default batch_size of", batch_size)

        pi_net_arch = params.get("pi_net_arch", None)
        if pi_net_arch is None:
            pi_net_arch = "256 256"
            params["pi_net_arch"] = pi_net_arch
            print("Using default pi_net_arch of", pi_net_arch)

        vf_net_arch = params.get("vf_net_arch", None)
        if vf_net_arch is None:
            vf_net_arch = "256 256"
            params["vf_net_arch"] = vf_net_arch
            print("Using default vf_net_arch of", vf_net_arch)

        activation_fn = params.get("activation_fn", None)
        if activation_fn is None:
            activation_fn = "tanh"
            params["activation_fn"] = activation_fn
            print("Using default activation_fn of", activation_fn)

        device = params.get("device", "cuda")

        print("Training with", self._total_steps, "total steps")

        tensorboard_log = "tensorboard_log"

        verbose = 1
        verbose = 0

        policy_kwargs = dict()
        policy_kwargs["net_arch"] = dict(
            pi=self.str_to_net_arch(pi_net_arch), vf=self.str_to_net_arch(vf_net_arch)
        )

        if activation_fn is None or activation_fn == "tanh":
            policy_kwargs["activation_fn"] = th.nn.Tanh
        elif activation_fn == "relu":
            policy_kwargs["activation_fn"] = th.nn.ReLU
        else:
            raise Exception(f"Unknown activation function {activation_fn}")

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


