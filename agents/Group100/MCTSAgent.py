from collections import defaultdict
from math import log, sqrt
import copy
import random
import time

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
    

    def get_possible_moves(self, board: Board) -> list[Move]:
        """ Get all valid moves at the current game state. """
        valid_moves = []

        for x in range(board.size):
            for y in range(board.size):
                t = board.tiles[x][y]
                if t.colour is None:
                    valid_moves.append(Move(x, y))

        return valid_moves


    def calculate_uct_score(self, c_param=1.4) -> float:
        """ Choose the best move using Upper Confidence bounds for Trees (UCT) selection method.

            At parent node v, choose child v' which maximises:
                (Q(v') / N(v')) + (C * sqrt((2 * ln(N(v))) / N(v')))
            where 
                Q(v) = the sum of all payoffs received,
                N(v) = the number of times the node has been visited
            and
                C = a custom parameter to determine exploration-exploitation trade-off
        """
        if self.visits == 0:
            return float('inf')  # Prioritize unvisited nodes
        
        # Return the UCT score of the current node
        return ((self.wins / self.visits) + c_param * sqrt((2 * log(self.parent.visits)) / self.visits))
    

    def apply_move(self, move: Move, colour: Colour) -> Board:
        """ Creates a new board with the move applied. """
        new_board = copy.deepcopy(self.board)
        new_board.tiles[move.x][move.y].colour = colour
        return new_board
    

class MCTSAgent(AgentBase):
    """ This class describes the Monte Carlo Tree Search (MCTS) Hex agent.
        In Monte Carlo learning, you learn while you play (between moves).

        1. Build a game tree, incrementally but asymmetrically.
        2. Use a tree policy to decide which node to expand. This policy must balance exploration and exploitation appropriately.
        3. For the expanded node, run a simulation using a default policy (often a random choice).
        4. Update the search tree. This will update the expanded node and all its ancestors.
        5. Return the best child of the current root node. 
        6. Repeat steps 2-5.

        The class inherits from AgentBase, which is an abstract class.
        The AgentBase contains the colour property which you can use to get the agent's colour.
        You must implement the make_move method to make the agent functional.
        You CANNOT modify the AgentBase class, otherwise your agent might not function.
    """

    _board_size: int = 11
    _colour: Colour

    def __init__(self, colour: Colour):
        super().__init__(colour)
        self._colour = colour


    def get_possible_moves(self, board: Board) -> list[Move]:
        """ Get all valid moves at the current game state. """
        valid_moves = []

        for x in range(board.size):
            for y in range(board.size):
                t = board.tiles[x][y]
                if t.colour is None:
                    valid_moves.append(Move(x, y))

        return valid_moves
    

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        """ The game engine will call this method to request a move from the agent.
            If the agent is to make the first move, opp_move will be None.
            If the opponent has made a move, opp_move will contain the opponent's move.
            If the opponent has made a swap move, opp_move will contain a Move object with x=-1 and y=-1,
            the game engine will also change your colour to the opponent colour.

            Args:
                turn (int): The current turn
                board (Board): The current board state
                opp_move (Move | None): The opponent's last move

            Returns:
                Move: The agent move
        """

        start_time = time.time()
        max_time = 5

        # If opponent swaps
        if opp_move and opp_move.x == -1 and opp_move.y == -1:  # Swap move
            return Move(board.size // 2, board.size // 2)  # Center is often a good move after swap
        
        # If opponent plays center on their first move
        if turn == 2 and board.tiles[board.size // 2][board.size // 2].colour is not None:
            return Move(-1, -1)  # Swap if center is taken
        
        root = MCTSNode(board)
            
        iterations = 0
        while time.time() - start_time < max_time:
            current_node = self.select_node(root)

            current_node = self.expand_node(current_node)

            result = self.simulate(current_node.board)

            self.backpropagate(current_node, result)

            iterations += 1

        print(f"MCTS Iterations: {iterations}, Time Spent: {time.time() - start_time:.2f}s")

        # Return the move with the most visits (the 'best' one) because it has 
        # been selected the most after all the MCTS iterations.
        best_child = max(root.children, key=lambda c: c.visits)

        return Move(best_child.move.x, best_child.move.y)


    def select_node(self, current_node: MCTSNode) -> MCTSNode:
        """ Select a child of current node to become the next current node. """

        # When all child nodes of current node have been explored, this calculates the UCT 
        # of each child and chooses the best one to become the new current node.
        while current_node.unexplored_children == [] and current_node.children:
            current_node = max(current_node.children, key=lambda child: child.calculate_uct_score())  # Select the best child

        # If the current node has unexplored children, the current node will be returned
        # This is the node that will be expanded to explore the children

        return current_node
        

    def expand_node(self, node: MCTSNode) -> MCTSNode:
        """ Performs exploration to add children to the current node.
            current_node.unexplored_children is reduced by 1, and current_node.children is increased by 1.
        """
        if node.unexplored_children:
            # Select a random move
            move = random.choice(node.unexplored_children)
            node.unexplored_children.remove(move)

            # Make the move
            new_board = node.apply_move(move, self.colour)

            # Make this a child of the current node
            child = MCTSNode(new_board, node, move)
            node.children.append(child)

            return child
        
        return node

    

    def simulate(self, state: Board) -> bool:
        """ Play a random simulation to completion. 
        
            Args:
                board (Board): The current board state

            Returns:
                result (bool): True if this agent has won, False otherwise.
        """

        simulation_board = copy.deepcopy(state)
        simulation_colour = self._colour
        
        # Loop forever until we reach a terminal node or the game ends
        while not simulation_board.has_ended(Colour.RED) and not simulation_board.has_ended(Colour.BLUE):
            moves = self.get_possible_moves(simulation_board)

            # If no moves available (we are at a terminal node)
            if not moves:
                break

            # Otherwise, simulate a random move
            random_move = random.choice(moves)
            simulation_board.set_tile_colour(random_move.x, random_move.y, simulation_colour)  # Make the move
            simulation_colour = Colour.opposite(simulation_colour)  # Swap colour to simulate opposite player's move
        
        # Returns true if this agent has won, false otherwise
        return simulation_board._winner == self.colour
    

    def backpropagate(self, node: MCTSNode, won: bool):
        """ Backpropagate result to all visited nodes. """

        while node:
            node.visits += 1

            if won:
                node.wins += 1

            # Move to parent
            node = node.parent
