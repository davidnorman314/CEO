from ceo.envs.ceo_any_seat_env import CEOAnySeatEnv
from ceo.envs.observation import ObservationFactory
from ceo.game.hand import Hand
from ceo.game.simplebehavior import BasicBehavior


class CEOPlayerEnv(CEOAnySeatEnv):
    """
    Environment for a player in a given seat. This environment's observations
    contains all the information available to a player.

    This is a specialization of CEOAnySeatEnv where the seat is fixed
    rather than randomly selected each episode.
    """

    seat_number: int
    """The fixed seat number for this environment."""

    def __init__(
        self,
        seat_number: int,
        num_players=6,
        behaviors=None,
        custom_behaviors=None,
        hands=None,
        listener=None,
        skip_passing=False,
        *,
        action_space_type="ceo",
        reward_includes_cards_left=False,
        obs_kwargs=None,
    ):
        self.seat_number = seat_number

        # Handle behaviors - CEOPlayerEnv accepts behaviors with None at the
        # agent's seat or an empty list, while CEOAnySeatEnv expects a full list
        if behaviors is None or len(behaviors) == 0:
            behaviors = [BasicBehavior() for _ in range(num_players)]
        else:
            # Replace None at agent's seat with a placeholder (won't be used)
            behaviors = list(behaviors)
            if behaviors[seat_number] is None:
                behaviors[seat_number] = BasicBehavior()

        # Handle custom_behaviors by merging into behaviors list
        if custom_behaviors is not None:
            for i, behavior in custom_behaviors.items():
                behaviors[i] = behavior

        if obs_kwargs is None:
            obs_kwargs = {}

        # Call parent constructor
        super().__init__(
            num_players=num_players,
            behaviors=behaviors,
            hands=hands,
            listener=listener,
            skip_passing=skip_passing,
            action_space_type=action_space_type,
            reward_includes_cards_left=reward_includes_cards_left,
            obs_kwargs=obs_kwargs,
        )

        # Override observation factory to NOT include seat number (original behavior)
        self.observation_factory = ObservationFactory(
            num_players, seat_number=seat_number, **obs_kwargs
        )
        self._observation_dimension = self.observation_factory.observation_dimension
        self.observation_space = self.observation_factory.create_observation_space()

        # Set current seat to fixed value
        self._current_seat = seat_number

    def _select_seat(self) -> int:
        """Override to always return the fixed seat number."""
        return self.seat_number

    def reset(self, *, seed=None, options=None, hands: list[Hand] = None):
        # Ensure observation factory uses fixed seat
        self.observation_factory.set_seat_number(self.seat_number)
        return super().reset(seed=seed, options=options, hands=hands)

    def step_full_action(self, full_action):
        """Step using a full action value."""
        self.step(self.action_space.find_full_action(full_action))
