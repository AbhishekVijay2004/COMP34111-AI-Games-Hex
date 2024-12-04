from math import log, sqrt
import random

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move


class MCTSNode:
    def __init__(self, board: Board, parent=None, move=None):
        self.board = board
        self.parent = parent
        self.move = move
        self.children = []
        self.wins = 0
        self.visits = 0
        self.unexplored_children = self.get_possible_moves(board)
        self.board_state = self.get_board_state(board)

    def get_possible_moves(self, board: Board) -> list[Move]:
        valid_moves = []
        for x in range(board.size):
            for y in range(board.size):
                if board.tiles[x][y].colour is None:
                    valid_moves.append(Move(x, y))
        return valid_moves

    def get_board_state(self, board: Board) -> tuple:
        """Create a tuple of board state for faster comparison"""
        return tuple(board.tiles[i][j].colour for i in range(board.size) for j in range(board.size))

    def calculate_uct_score(self, c_param=1.4) -> float:
        if self.visits == 0:
            return float('inf')
        return ((self.wins / self.visits) + c_param * sqrt((2 * log(self.parent.visits)) / self.visits))

    def apply_move(self, move: Move, colour: Colour) -> Board:
        """Apply move and return board for chaining"""
        self.board.set_tile_colour(move.x, move.y, colour)
        return self.board

    def undo_move(self, move: Move):
        """Undo a move on the board"""
        self.board.undo_move(move.x, move.y)


class RaveAgent(AgentBase):
    def __init__(self, colour: Colour):
        super().__init__(colour)
        self._colour = colour

    def verify_board_state(self, node: MCTSNode, state: Board, stage: str) -> bool:
        current_state = node.get_board_state(state)
        if current_state != node.board_state:
            print(f"WARNING: Board state mismatch at {stage}!")
            print(f"Expected:\n{node.board_state}")
            print(f"Got:\n{current_state}")
            print(f"Board:\n{state}")
            return False
        return True

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        if opp_move and opp_move.x == -1 and opp_move.y == -1:
            return Move(board.size // 2, board.size // 2)
        
        if turn == 2 and board.tiles[board.size // 2][board.size // 2].colour is not None:
            return Move(-1, -1)
        
        root = MCTSNode(board)
        
        for _ in range(1000):
            # Verify board state before each iteration
            if not self.verify_board_state(root, board, "iteration start"):
                print("WARNING: Board state changed between iterations!")
            
            current_node = self.select_node(root)
            current_node = self.expand_node(current_node)
            result = self.simulate(current_node.board)
            self.backpropagate(current_node, result)
            
            # Verify board returned to original state
            if not self.verify_board_state(root, board, "iteration end"):
                print("WARNING: Board not restored after iteration!")

        best_child = max(root.children, key=lambda c: c.visits)
        return Move(best_child.move.x, best_child.move.y)

    def select_node(self, current_node: MCTSNode) -> MCTSNode:
        while current_node.unexplored_children == [] and current_node.children:
            current_node = max(current_node.children, key=lambda child: child.calculate_uct_score())
        return current_node

    def expand_node(self, node: MCTSNode) -> MCTSNode:
        if node.unexplored_children:
            move = random.choice(node.unexplored_children)
            node.unexplored_children.remove(move)
            node.apply_move(move, self.colour)
            child = MCTSNode(node.board, node, move)
            node.children.append(child)
            # Undo the move after creating child
            node.undo_move(move)
            return child
        return node

    def simulate(self, state: Board) -> bool:
        moves_played = []
        simulation_colour = self._colour
        
        # Store valid moves indices instead of Move objects
        empty_positions = [(x, y) 
                         for x in range(state.size) 
                         for y in range(state.size) 
                         if state.tiles[x][y].colour is None]
        
        while not state.has_ended(Colour.RED) and not state.has_ended(Colour.BLUE):
            if not empty_positions:
                break
                
            # Pick and apply random move
            pos_idx = random.randrange(len(empty_positions))
            x, y = empty_positions.pop(pos_idx)
            state.set_tile_colour(x, y, simulation_colour)
            moves_played.append((x, y))
            
            simulation_colour = Colour.opposite(simulation_colour)

        result = state._winner == self.colour

        # Undo moves
        for x, y in reversed(moves_played):
            state.undo_move(x, y)

        return result

    def backpropagate(self, node: MCTSNode, won: bool):
        while node:
            node.visits += 1
            if won:
                node.wins += 1
            node = node.parent
