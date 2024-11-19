from random import *

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move


class NaiveAgent(AgentBase):
    def __init__(self, colour: Colour):
        super().__init__(colour)
        self._board_size = 11
        # Pre-calculate all possible positions
        self._all_positions = [(i, j) for i in range(self._board_size) 
                             for j in range(self._board_size)]

    def get_valid_moves(self, board: Board) -> list[tuple[int, int]]:
        """Optimized valid moves calculation"""
        # Use list comprehension with early exit
        return [pos for pos in self._all_positions 
                if board.tiles[pos[0]][pos[1]].colour is None]

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        """Make a random valid move"""
        # Special case for second turn - consider swap
        if turn == 2 and random() < 0.5:  # 50% chance to swap
            return Move(-1, -1)
        
        valid_moves = self.get_valid_moves(board)
        if not valid_moves:
            return Move(-1, -1)
            
        # Choose random move from valid moves
        x, y = choice(valid_moves)
        return Move(x, y)
