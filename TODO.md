# TODO Items: PPO
- Convert StableBaselines changes to a custom policy, see [here](https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html).

- Learning rate.
  - Tune learning rate.
  - Variable learning rate

- Tune gaelambda parameter.

- Support loading a saved agent and playing games with it.

- Load a trained model and train it further

- Test different activation functions

# TODO Items: Q learning

- Features need to contain information that if the agent played a triple,
  then other players are less likely to play, especially if it is a high
  triple.

- When the bucket feature is card counts, if the only remaining cards are in buckets, 
  then the maximum bucket count can give quadruples, triples, and pairs the same observation.

- Current actions do not allow breaking up four kings into two pair if there is an ace
  in the hand.

- Keep track of variance of estimates and use it to terminate search if the estimates are tight enough.

- Add an action to break up a quadruple and triple

- Action to play second highest

- Add feature to HandSummary giving the number of singles in the buckets.

# Completed Items

- Some states aren't getting visited enough, even with 30 million training episodes.
  - Add max_initial_visit_count


