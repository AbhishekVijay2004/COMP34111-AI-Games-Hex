from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move
from math import inf
from random import choice
from collections import deque

from src.Tile import Tile


class MinimaxAgent(AgentBase):
    """NaiveAgent with a minimax implementation."""

    def __init__(self, colour: Colour):
        super().__init__(colour)
        self._opponent_colour = Colour.RED if colour == Colour.BLUE else Colour.BLUE
        self._max_depth = 3  # Set a reasonable depth to limit computation.

    def minimax(self, board: Board, depth: int, alpha: float, beta: float, maximizing: bool) -> tuple[float, Move | None]:
        """Minimax algorithm with alpha-beta pruning."""
        # Base case: check if game has ended or depth limit reached
        if depth == 0 or board.has_ended(self.colour) or board.has_ended(self._opponent_colour):
            return self.evaluate_board(board), None

        best_move = None

        if maximizing:
            max_eval = -inf
            for move in self.generate_valid_moves(board):
                # Simulate the move
                board.set_tile_colour(move.x, move.y, self.colour)
                eval, _ = self.minimax(board, depth - 1, alpha, beta, False)
                # Undo the move
                board.set_tile_colour(move.x, move.y, None)

                if eval > max_eval:
                    max_eval = eval
                    best_move = move

                alpha = max(alpha, eval)
                if beta <= alpha:
                    break

            return max_eval, best_move
        else:
            min_eval = inf
            for move in self.generate_valid_moves(board):
                # Simulate the move
                board.set_tile_colour(move.x, move.y, self._opponent_colour)
                eval, _ = self.minimax(board, depth - 1, alpha, beta, True)
                # Undo the move
                board.set_tile_colour(move.x, move.y, None)

                if eval < min_eval:
                    min_eval = eval
                    best_move = move

                beta = min(beta, eval)
                if beta <= alpha:
                    break

            return min_eval, best_move

    def generate_valid_moves(self, board: Board) -> list[Move]:
        """Generate all valid moves for the current board."""
        valid_moves = []
        for x in range(board.size):
            for y in range(board.size):
                if board.tiles[x][y].colour is None:  # Only unoccupied tiles
                    valid_moves.append(Move(x, y))
        return valid_moves

    def evaluate_board(self, board: Board) -> float:
        """Heuristic evaluation function emphasizing connected tiles."""
        # Use a heuristic to evaluate the board's favorability.
        if board.has_ended(self.colour):
            return inf  # Winning state
        if board.has_ended(self._opponent_colour):
            return -inf  # Losing state

        # Calculate largest connected component sizes for both players.
        def largest_connected_component(colour):
            visited = set()

            def bfs(x, y):
                """Breadth-First Search to compute the size of a connected component."""
                queue = deque([(x, y)])
                component_size = 0
                while queue:
                    cx, cy = queue.popleft()
                    if (cx, cy) in visited:
                        continue
                    visited.add((cx, cy))
                    component_size += 1

                    # Explore neighbors
                    for dx, dy in zip(Tile.I_DISPLACEMENTS, Tile.J_DISPLACEMENTS):
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < board.size and 0 <= ny < board.size:
                            if (nx, ny) not in visited and board.tiles[nx][ny].colour == colour:
                                queue.append((nx, ny))

                return component_size

            # Find the largest connected component for the given colour.
            max_size = 0
            for x in range(board.size):
                for y in range(board.size):
                    if board.tiles[x][y].colour == colour and (x, y) not in visited:
                        max_size = max(max_size, bfs(x, y))
            return max_size

        # Calculate heuristic based on largest connected components.
        my_largest_component = largest_connected_component(self.colour)
        opp_largest_component = largest_connected_component(self._opponent_colour)

        return my_largest_component - opp_largest_component
    





    

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        """Select a move using the minimax algorithm."""
        if turn == 2:
            return Move(-1, -1)  # Handle swap logic if required

        _, best_move = self.minimax(board, self._max_depth, -inf, inf, True)
        if best_move is None:
            raise Exception("No valid moves found.")
        return best_move