# TODO Items

- Features need to contain information that if the agent played a triple,
  then other players are less likely to play, especially if it is a high
  triple.

- When the bucket feature is card counts, if the only remaining cards are in buckets, 
  then the maximum bucket count can give quadruples, triples, and pairs the same observation.

- With buckets, playing lowest and second lowest can result in the same afterstate. Currently
  the ordering leads to second lowest being played in this case.
