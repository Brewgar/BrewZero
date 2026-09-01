"""Chess environment: state handling, canonicalization, encoding, action space.

All modules in this package operate on ``chess.Board`` instances and numpy
tensors. Perspective convention:

    * The network always acts from the *side-to-move* perspective.
    * A canonical board is a board in which the side to move is represented as
      *White* moving "up" the board (rank 1 -> rank 8).
"""