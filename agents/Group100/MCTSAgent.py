from collections import defaultdict
import copy
from math import log, sqrt
import random

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move


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

    _choices: list[Move]
    _board_size: int = 11
    _colour: Colour
    _currentMoveColour: Colour


    def __init__(self, colour: Colour):
        super().__init__(colour)
        self._colour = colour
        self._choices = [
            (i, j) for i in range(self._board_size) for j in range(self._board_size)
        ]

        # Create a dictionary to represent the game tree. If we try to access a 
        # node which doesn't exist, it will be created within the dictionary/tree.
        self.game_tree = defaultdict(lambda: {"wins": 0, "visits": 0, "children": []})


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

        # If make_move is called, it is our turn to play. Set to this agent's colour.
        self._currentMoveColour = self._colour

        best_move = None
        best_score = -float('inf')
        possible_moves = self.get_possible_moves(board)
        NUM_SIMULATIONS = 10

        for move in possible_moves:
            score = self.run_mcts(move, board, NUM_SIMULATIONS)

            if score > best_score:
                best_score = score
                best_move = move

        print("Best move found: ", best_move)
        
        return best_move


    def run_mcts(self, move: Move, board: Board, iterations: int) -> float:
        """ Run Monte Carlo Tree Search (MCTS) for the number of iterations specified. """
        print("Running MCTS for move:", move)

        # Set the root node to be the current move
        root = self.game_tree[move]

        for _ in range(iterations):
            current_node = root
            board_copy = copy.deepcopy(board)

            # A record of the sequence of moves so that we can 
            # backpropagate results after random simulation.
            path = [(move, current_node)]

            # At the very start, selection does nothing because the root node has 
            # no children, so we need to expand the root node first.

            # Expand current node if children are unknown
            #  and current_node["visits"] == 0 
            # WHAT IF THE NODE IS TERMINAL AND HAS NO CHILDREN?
            print("current node has children:", len(current_node["children"]))
            if not current_node["children"]:
                print("expanding current node...")
                self.expand_node(board_copy, current_node)
            else:
                """ If children of the current node exist, perform the algorithm: select the best child from 
                    the current node, simulate this move, and then simulating random play until a result is 
                    obtained. Then, backpropagate this result to update the path. """
                if current_node["children"]:
                    self.selection(current_node, board_copy, path)  # Select the best child, simulate the move

                    result = self.rollout(board_copy)  # Simulate random play till the end of the game

                    self.backpropagate(path, result)  # Update the path with the result

        print("\n\n\nRoot:", root["wins"], max(1, root["visits"]), "- Score:", root["wins"] / max(1, root["visits"]))
        return root["wins"] / max(1, root["visits"])
    

    def selection(self, current_node: dict, board_copy: Board, path: list):
        """ If the current node is not terminal (i.e. it has children), select the best child and simulate the move. """

        # If the current node has children
        if current_node["children"]:
            move, current_node = self.select_best_child(current_node)  # Select the best child move
            board_copy.set_tile_colour(move.x, move.y, self._colour)  # Simulate the move
            path.append((move, current_node))  # Add this move to the path record


    def rollout(self, board: Board) -> bool:
        """ Play a random simulation to completion. 
        
            Args:
                board (Board): The current board state

            Returns:
                result (int):   +1 if this agent has won, 
                                -1 if this agent has lost, 
                                0 if the game is still on-going
        """
        
        # Loop forever until we reach a terminal node or the game ends
        while not board.has_ended(self._currentMoveColour):
            moves = self.get_possible_moves(board)

            # If no moves available (we are at a terminal node)
            if not moves:
                break

            # Otherwise, simulate a random move
            random_move = random.choice(moves)
            board.set_tile_colour(random_move.x, random_move.y, self._currentMoveColour)  # Make the move
            self._currentMoveColour = Colour.opposite(self._currentMoveColour)  # Swap colour to simulate opposite player's move
        
        # Returns true if this agent has won, false otherwise
        return board.has_ended(self._colour)
    

    def expand_node(self, board_copy: Board, current_node: dict):
        """ Expands one level deeper from the current node to explore child nodes. Adds these to the game tree. """
        if not board_copy.has_ended(self._colour):
            possible_moves = self.get_possible_moves(board_copy)

            # Add every possible move as a child node of the current node
            for child_move in possible_moves:
                child = {"wins": 0, "visits": 0, "children": []}
                self.game_tree[child_move] = child
                current_node["children"].append((child_move, child))


    def select_best_child(self, node: dict) -> tuple:
        """ Select the child with the highest UCT value from the current node. """

        # Add up all visits that involve (pass through) the current node
        total_vists = sum(child["visits"] for _, child in node["children"])

        # For each child of the current node, calculate the score and choose the largest value
        best_move, best_node = max(
            node["children"],
            key=lambda x: self.calculate_uct_score(x[1], total_vists)
        )

        return best_move, best_node
    

    def calculate_uct_score(self, child_node: dict, total_vists: int, c_param=1.4) -> float:
        """ Choose the best move using Upper Confidence bounds for Trees (UCT) selection method.

            At parent node v, choose child v' which maximises:
                (Q(v') / N(v')) + (C * sqrt((2 * ln(N(v))) / N(v')))
            where 
                Q(v) = the sum of all payoffs received,
                N(v) = the number of times the node has been visited
            and
                C = a custom parameter to determine exploration-exploitation trade-off

            Args:
                node (dict): A node in the game tree in the form {"wins": 0, "visits": 0, "children": []}
        """
        if child_node["visits"] == 0:
            return float('inf')  # Prioritize unvisited nodes
        
        # Return the UCT score of the current node
        return ((child_node["wins"] / child_node["visits"]) + c_param * sqrt((2 * log(total_vists)) / child_node["visits"]))


    def backpropagate(self, path: list, result: bool):
        """ Backpropagate the simulation result. """

        for move, node in path:
            node["visits"] += 1
            if result:
                node["wins"] += 1
