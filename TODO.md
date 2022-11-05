# TODO Items: PPO
- Save environment or environment args in eval_log
  - Currently play_qagent.py creates the enviornment from scratch, so it might not match.

- Learning rate.
  - Tune learning rate.
  - Variable learning rate

- Tune gaelambda parameter.

- Load a trained model and train it further

- Test different activation functions

- Add feature for number of aces remaining and see how much better the agent does.

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


