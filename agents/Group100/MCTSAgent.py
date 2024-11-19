from collections import defaultdict
from math import log, sqrt
import copy
import random
import time

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
        # game_tree stores all the nodes we have explored in our MCTS.
        self.game_tree = defaultdict(lambda: {
            "wins": 0,
            "visits": 0,
            "children": [],
            "unexplored_moves": []
        })


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
        max_time = 2.0

        if opp_move:
            if opp_move.x == -1 and opp_move.y == -1:  # Swap move
                return Move(board.size // 2, board.size // 2)  # Center is often a good move after swap
            
        root_state = copy.deepcopy(board)
        root = self.game_tree[str(root_state)]
        root["unexplored_moves"] = self.get_possible_moves(board)

        iterations = 0
        while time.time() - start_time < max_time:

            state = copy.deepcopy(root_state)
            current_node = root

            # If make_move is called, it is our turn to play. Set to this agent's colour.
            current_colour = self._colour

            # 1. Selection
            # If the current node has children, and all of these have been explored:
            path = []
            while current_node["unexplored_moves"] == [] and current_node["children"]:
                # If the current node is not terminal (i.e. it has children), select the best child and simulate the move.
                move, current_node = self.select_best_child(current_node)  # Select the best child
                state.set_tile_colour(move.x, move.y, current_colour)  # Make the move
                path.append((move, current_node, current_colour))  # Add move to the path, ready for backpropagation
                current_colour = Colour.opposite(current_colour)  # Swap to opponents colour

            # 2. Expansion
            # If the current node has child nodes that are unexplored:
            if current_node["unexplored_moves"]:
                move = random.choice(current_node["unexplored_moves"])  # Pick a random child node
                current_node["unexplored_moves"].remove(move)
                state.set_tile_colour(move.x, move.y, current_colour)  # Make the move for this child node
                
                # Create the child node that is associated with making the move above.
                # After expanding, add this new child node to the children of the current node.
                next_state_key = str(state)
                next_node = self.game_tree[next_state_key]
                next_node["unexplored_moves"] = self.get_possible_moves(state)
                current_node["children"].append((move, next_node))
            
                path.append((move, next_node, current_colour))
                current_colour = Colour.opposite(current_colour)

            # 3. Simulation
            result = self.simulate(state, current_colour)

            # 4. Backpropagation
            self.backpropagate(path, result)

            iterations += 1

        print(f"MCTS Iterations: {iterations}, Time Spent: {time.time() - start_time:.2f}s")

        # Return the best move
        if not root["children"]:
            # If no children, return a random valid move
            return random.choice(root["unexplored_moves"])
        else:
            # Otherwise, select the move with the highest visit count
            return max(root["children"], key=lambda x: x[1]["visits"])[0]


    def simulate(self, state: Board, current_colour: Colour) -> bool:
        """ Play a random simulation to completion. 
        
            Args:
                board (Board): The current board state

            Returns:
                result (int):   +1 if this agent has won, 
                                -1 if this agent has lost, 
                                0 if the game is still on-going
        """

        simulation_state = copy.deepcopy(state)
        simulation_colour = current_colour
        
        # Loop forever until we reach a terminal node or the game ends
        while not simulation_state.has_ended(self._colour) and not simulation_state.has_ended(Colour.opposite(self._colour)):
            moves = self.get_possible_moves(simulation_state)

            # If no moves available (we are at a terminal node)
            if not moves:
                break

            # Otherwise, simulate a random move
            random_move = random.choice(moves)
            simulation_state.set_tile_colour(random_move.x, random_move.y, simulation_colour)  # Make the move
            simulation_colour = Colour.opposite(simulation_colour)  # Swap colour to simulate opposite player's move
        
        # Returns true if this agent has won, false otherwise
        return simulation_state.has_ended(self._colour)


    def select_best_child(self, node: dict) -> tuple:
        """ Select the child with the highest UCT value from the current node. """

        # Add up all visits that involve (pass through) the current node
        total_vists = sum(child["visits"] for _, child in node["children"])

        # For each child of the current node, calculate the score and choose the largest value
        # Calculate UCT scores for every child node
        scored_moves = [(move, child, self.calculate_uct_score(child, total_vists)) for move, child in node["children"]]

        # Select the child with the greatest score
        best_move, best_node = max(scored_moves, key=lambda x: x[2])[0:2]  # [0:2] means the first two elements: (move, child)

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


    def backpropagate(self, path: list, won: bool):
        """ Backpropagate result to all visited nodes. """

        for move, node, colour in path:
            node["visits"] += 1
            # Only increment wins if the result is good for the color that made the move
            if (colour == self._colour) == won:
                node["wins"] += 1
