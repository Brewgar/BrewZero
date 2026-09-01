"""Engine package: Stockfish UCI wrapper, WDL math, reward computation.

Perspective convention (identical to the rest of the project):

    * All engine scores are first normalized into **White-perspective**
      floating values using the WDL-derived centered score
      ``S_white = 2 * (p_w + p_d / 2) - 1`` in [-1, +1].
    * Side-to-move relative scores are derived via a single sign flip:
      ``S_stm = S_white`` if White to move else ``-S_white``.
    * After a move, the opponent is to move, so the previous mover's relative
      post-move score is ``-S_stm(next_position)``.
"""