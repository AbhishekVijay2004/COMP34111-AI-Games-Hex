from math import sqrt, log
from random import choice
from copy import deepcopy
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
        self.untried_moves = self._get_valid_moves(board)
    
    def _get_valid_moves(self, board: Board) -> list[tuple[int, int]]:
        return [(i, j) for i in range(board.size) 
                for j in range(board.size) 
                if board.tiles[i][j].colour is None]

    def ucb1(self, exploration=1.41):
        if self.visits == 0:
            return float('inf')
        return (self.wins / self.visits) + exploration * sqrt(log(self.parent.visits) / self.visits)

    def apply_move(self, move: tuple[int, int], colour: Colour) -> Board:
        """Creates a new board with the move applied"""
        new_board = deepcopy(self.board)
        new_board.tiles[move[0]][move[1]].colour = colour
        return new_board

class MCTSAgent(AgentBase):
    def __init__(self, colour: Colour):
        super().__init__(colour)
        self.simulations = 1000  # Number of MCTS iterations

    def select_node(self, node: MCTSNode) -> MCTSNode:
        while node.untried_moves == [] and node.children:
            node = max(node.children, key=lambda n: n.ucb1())
        return node

    def expand(self, node: MCTSNode) -> MCTSNode:
        if node.untried_moves:
            move = choice(node.untried_moves)
            node.untried_moves.remove(move)
            new_board = node.apply_move(move, self.colour)
            child = MCTSNode(new_board, node, move)
            node.children.append(child)
            return child
        return node

    def simulate(self, board: Board) -> bool:
        temp_board = deepcopy(board)
        current_colour = self.colour
        
        while True:
            # Check if either player has won
            temp_board.has_ended(Colour.RED)
            temp_board.has_ended(Colour.BLUE)
            if temp_board._winner is not None:
                break
                
            moves = [(i, j) for i in range(temp_board.size) 
                    for j in range(temp_board.size) 
                    if temp_board.tiles[i][j].colour is None]
            if not moves:
                break
                
            x, y = choice(moves)
            temp_board.set_tile_colour(x, y, current_colour)
            current_colour = Colour.RED if current_colour == Colour.BLUE else Colour.BLUE
            
        return temp_board._winner == self.colour

    def backpropagate(self, node: MCTSNode, result: bool):
        while node:
            node.visits += 1
            node.wins += result
            node = node.parent

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        if turn == 2 and board.tiles[board.size//2][board.size//2].colour is not None:
            return Move(-1, -1)  # Swap if center is taken
            
        root = MCTSNode(board)
        
        for _ in range(self.simulations):
            node = self.select_node(root)
            node = self.expand(node)
            result = self.simulate(node.board)
            self.backpropagate(node, result)
        
        best_child = max(root.children, key=lambda c: c.visits)
        return Move(best_child.move[0], best_child.move[1])