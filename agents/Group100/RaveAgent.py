from math import log, sqrt
from itertools import combinations
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
                if board.tiles[x][y].colour is None:
                    valid_moves.append(Move(x, y))
        return valid_moves

    def calculate_uct_score(self, c_param=1.4) -> float:
        """ Calculate UCT score for node selection. """
        if self.visits == 0:
            return float('inf')
        return ((self.wins / self.visits) +
                c_param * sqrt((2 * log(self.parent.visits)) / self.visits))
    
    def copy_board(self, board: Board) -> Board:
        """Create an efficient copy of the board state."""
        new_board = Board(board.size)
        new_board._winner = board._winner
        for i in range(board.size):
            for j in range(board.size):
                new_board.tiles[i][j].colour = board.tiles[i][j].colour
        return new_board

    def apply_move(self, move: Move, colour: Colour) -> Board:
        """ Apply a move to a copy of the board and return it. """
        new_board = self.copy_board(self.board)
        new_board.tiles[move.x][move.y].colour = colour
        return new_board


class MCTSAgent(AgentBase):
    _board_size: int = 11
    _colour: Colour

    def __init__(self, colour: Colour):
        super().__init__(colour)
        self.weights = {
            'center_weight': 1.403,  # Moves closer to the center are prioritised
            'neighbour_weight': 0.943,
            'bridge_weight': 0.943,
            'edge_weight': 0.943,
            'defensive_weight': 6.420,
            'connection_weight': 2.5,  # Moves that connect tiles
            'parallel_two_bridge_weight': 6.421,  # Moves that create two bridges in the intended direction
            'perpendicular_two_bridge_weight': 6.221,  # Moves that create two bridges in the unintended direction
            'diagonal_two_bridge_weight': 6.321,  # Moves that create two bridges diagonally
            'opponent_bridge_block': 6.118,  # Moves that block opponent two bridges
            'explore_constant': 2.005,
            'rave_constant': 340.901,
            'early_stop_threshold': 0.934,
            'min_visits_ratio': 0.140,
            'swap_strength_threshold': 0.741,
        }
        self._colour = colour
        self.current_simulation = 0  # Keep track of current simulation (out of max total_simulations)
        self.total_simulations = 1000  # Total number of simulations to run per move
        self.move_scores = {}  # Cache for move evaluations

        # Initialize Zobrist hashing
        random.seed(42)
        self.zobrist_table = {}
        for x in range(self._board_size):
            for y in range(self._board_size):
                self.zobrist_table[(x, y, Colour.RED)] = random.getrandbits(64)
                self.zobrist_table[(x, y, Colour.BLUE)] = random.getrandbits(64)
        self.transposition_table = {}

    def get_possible_moves(self, board: Board) -> list[Move]:
        """ Get all valid moves in the current board state. """
        valid_moves = []
        for x in range(board.size):
            for y in range(board.size):
                if board.tiles[x][y].colour is None:
                    valid_moves.append(Move(x, y))
        return valid_moves
    
    def get_smart_moves(self, board: Board) -> list[Move]:
        """
        Return an ordered list of valid moves such that the first entry is the best move. 
        
        The list is sorted by the evaluation score of each move.
        """
        valid_moves = self.get_possible_moves(board)

        # Sort the moves by their evaluation score
        smart_moves = sorted(valid_moves, 
                                key=lambda move: self.evaluate_move(board, move), 
                                reverse=True)
        
        return smart_moves
    
    def is_my_tile(self, board: Board, move: Move | tuple[int, int]) -> bool:
        """ Returns True if the given move is the same colour as (owned by) the agent, False otherwise. """
        x, y = (move.x, move.y) if isinstance(move, Move) else move
        return (0 <= x < board.size and 0 <= y < board.size and
                board.tiles[x][y].colour is self._colour)
    
    def is_opponent_tile(self, board: Board, move: Move | tuple[int, int]) -> bool:
        """ Returns True if the given move is an opponent tile, False otherwise. """
        x, y = (move.x, move.y) if isinstance(move, Move) else move
        return (0 <= x < board.size and 0 <= y < board.size and
                board.tiles[x][y].colour is Colour.opposite(self._colour))
    
    def get_neighbours_of_colour(self, board: Board, move: Move | tuple[int, int], colour: Colour) -> list[Move]:
        """ Returns a list of neighbours of the current move if they match the specified player colour. """
        x, y = (move.x, move.y) if isinstance(move, Move) else move
        neighbours = []
        for dx, dy in [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]:
            if 0 <= x + dx < board.size and 0 <= y + dy < board.size:
                if board.tiles[x + dx][y + dy].colour is colour:
                    neighbours.append(Move(x + dx, y + dy))
        return neighbours
    
    def is_neighbour(self, move1: Move | tuple[int, int], move2: Move | tuple[int, int]) -> bool:
        """ Returns True if the two given moves are neighbours, False otherwise. """
        x1, y1 = (move1.x, move1.y) if isinstance(move1, Move) else move1
        x2, y2 = (move2.x, move2.y) if isinstance(move2, Move) else move2

        move1_relative_neighbors_positions = [
            (x1 - 1, y1),
            (x1 + 1, y1),
            (x1 - 1, y1 + 1),
            (x1 + 1, y1 + 1),
            (x1, y1 - 1),
            (x1, y1 + 1),
        ]

        # Return True if move2 is in the list of relative neighbours of move1
        return (x2, y2) in move1_relative_neighbors_positions

    def check_immediate_win(self, board: Board, move: Move | tuple[int, int], player: Colour) -> bool:
        x, y = (move.x, move.y) if isinstance(move, Move) else move
        if not self.is_valid_move(board, (x, y)):
            return False
        board.set_tile_colour(x, y, player)
        won = board.has_ended(player)
        board.set_tile_colour(x, y, None)
        return won

    def get_all_positions_for_colour(self, board: Board, colour: Colour) -> list[Move]:
        positions = []
        for x in range(board.size):
            for y in range(board.size):
                if board.tiles[x][y].colour is colour:
                    positions.append(Move(x, y))
        return positions

    def copy_board(self, board: Board) -> Board:
        """Create an efficient copy of the board state."""
        new_board = Board(board.size)
        new_board._winner = board._winner
        for i in range(board.size):
            for j in range(board.size):
                new_board.tiles[i][j].colour = board.tiles[i][j].colour
        return new_board

    def is_valid_move(self, board: Board, move: Move | tuple[int, int]) -> bool:
        x, y = (move.x, move.y) if isinstance(move, Move) else move
        return (0 <= x < board.size and 0 <= y < board.size and
                board.tiles[x][y].colour is None)
    
    def evaluate_move(self, board: Board, move: Move | tuple[int, int]) -> float:
        """ 
        Evaluates the quality of a given move. 
        
        Returns a score where a higher score represents a better move.
        """

        score = 0  # Initialize score
        x, y = (move.x, move.y) if isinstance(move, Move) else move
        center = board.size // 2
        
        # Prioritise moves closer to the center
        dist_to_center = abs(x - center) + abs(y - center)
        score += (max(0, (board.size - dist_to_center)) / board.size) * self.weights['center_weight']

        # Prioritise moves that make or block two bridges
        two_bridge_score = self.get_two_bridges_score(board, move)
        score += two_bridge_score

        # Prioritise moves that connect tiles
        connection_score = self.get_connection_score(board, move)
        score += connection_score

        return score
    
    def should_we_swap(self, opp_move: Move) -> bool:
        """ Returns True if we should swap, False otherwise. """

        # A list of moves where it is better to swap than not to swap
        # Taken from here: https://www.hexwiki.net/index.php/Swap_rule
        swap_moves = []

        # y-axis (2 to 8)
        for x in range(2,9):
            # The whole x-axis (0 to 10)
            for y in range(11):
                # Skip the corners
                if not (x == 2 and y == 0) and not (x == 8 and y == 0) and not (x == 2 and y == 10) and not (x == 8 and y == 10):
                    swap_moves.append(Move(x, y))
        
        # Hard coding the rest of the swap moves
        swap_moves.append(Move(0,10))
        swap_moves.append(Move(1,9))
        swap_moves.append(Move(1,10))
        swap_moves.append(Move(9,0))
        swap_moves.append(Move(10,0))
        swap_moves.append(Move(9,1))

        # On turn 2 our agent should always be BLUE, no need to consider RED
        if opp_move is not None and self._colour == Colour.BLUE:
            if opp_move in swap_moves:
                return True

        return False

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        start_time = time.time()
        max_time = 4.9  # Used to ensure the agent never times out

        self.get_possible_moves(board)

        # Strategic swap evaluation - should we swap?
        if turn == 2 and self.should_we_swap(opp_move):
            return Move(-1, -1)

        valid_moves = self.get_possible_moves(board)

        # Immediate win
        for move in valid_moves:
            if self.check_immediate_win(board, (move.x, move.y), self._colour):
                return move

        # Immediate block
        for move in valid_moves:
            if self.check_immediate_win(board, (move.x, move.y), Colour.opposite(self._colour)):
                return move

        # Save two-bridge if threatened
        if opp_move:
            save_move = self.find_two_bridge_saving_moves(board)
            if save_move:
                return random.choice(save_move)
            
        root = MCTSNode(board)

        # *** MCTS main loop ***
        self.current_simulation = 0
        while time.time() - start_time < max_time and self.current_simulation < self.total_simulations:
            current_node = self.select_node(root)
            current_node = self.expand_node(current_node)
            result = self.simulate(current_node.board)
            self.backpropagate(current_node, result)
            self.current_simulation += 1

        #print(f"Turn {turn}: MCTS Iterations: {self.current_simulation}, Time Spent: {time.time() - start_time:.2f}s")

        best_child = max(root.children, key=lambda c: c.visits)

        return Move(best_child.move.x, best_child.move.y)

    def select_node(self, current_node: MCTSNode) -> MCTSNode:
        while not current_node.unexplored_children and current_node.children:
            current_node = max(current_node.children, key=lambda child: child.calculate_uct_score())
        return current_node

    def expand_node(self, node: MCTSNode) -> MCTSNode:
        if node.unexplored_children:
            # Re-evaluate all unexplored_children so that the best move is always chosen
            node.unexplored_children = sorted(node.unexplored_children, 
                                                key=lambda move: self.evaluate_move(node.board, move), 
                                                reverse=True)
            move = node.unexplored_children.pop(0)  # The first move is always the best
            new_board = node.apply_move(move, self._colour)
            child = MCTSNode(new_board, node, move)
            node.children.append(child)
            return child
        
        return node

    def simulate(self, state: Board) -> bool:
        """ Run a quick random simulation to the end from the given state. """
        # Check transposition table
        current_hash = self.hash_board(state)
        if current_hash in self.transposition_table:
            return self.transposition_table[current_hash]

        # Simulate on a copy
        simulation_board = self.copy_board(state)
        simulation_colour = self._colour
        moves = self.get_smart_moves(simulation_board)

        # Fast random simulation without intermediate hashing
        while not simulation_board.has_ended(Colour.RED) and not simulation_board.has_ended(Colour.BLUE):
            if not moves:
                break
            move = moves[0]  # The first move in get_smart_moves is the best move
            simulation_board.set_tile_colour(move.x, move.y, simulation_colour)
            moves.remove(move)
            simulation_colour = Colour.opposite(simulation_colour)

        result = (simulation_board._winner == self._colour)
        self.transposition_table[current_hash] = result
        return result

    def hash_board(self, board: Board) -> int:
        h = 0
        for x in range(board.size):
            for y in range(board.size):
                color = board.tiles[x][y].colour
                if color is not None:
                    h ^= self.zobrist_table[(x, y, color)]
        return h

    def hash_update(self, hash_val: int, x: int, y: int, color: Colour) -> int:
        return hash_val ^ self.zobrist_table[(x, y, color)]

    def backpropagate(self, node: MCTSNode, won: bool):
        while node:
            node.visits += 1
            if won:
                node.wins += 1
            node = node.parent

    def get_two_bridges(self, board: Board, colour: Colour, move: Move = None) -> list[tuple[Move, tuple[Move, Move], Move]]:
        """
        Returns a list of two bridges that can be created by the given colour.

        If a move is specified, it will return two bridges that can be created by making that move. If not, 
        it will return all two bridges that already exist for the given colour in the given board state.

        Returns the move needed to make the two bridge, the original piece, and the two empty cells that will be created in the form of:
            [(bridge_pos1, (empty_cell1, empty_cell2), bridge_pos2), ...]

        This function assumes move is a valid move.
        """

        two_bridges = []

        if move:
            two_bridges = self.get_two_bridges_helper(board, move)
        else: 
            # Find already existing two bridges
            current_nodes = self.get_all_positions_for_colour(board, colour)

            for node in current_nodes:
                found_two_bridges = self.get_two_bridges_helper(board, node)

                # Add the found two bridges to the master list
                for two_bridge in found_two_bridges:
                    two_bridges.append(two_bridge)

        return two_bridges
    
    def get_two_bridges_helper(self, board: Board, move: Move) -> list[tuple[Move, tuple[Move, Move]]]:
        """ A helper function for get_two_bridges. This just abstracts repeated code. """

        two_bridges = []

        # Pattern 1: Top-left bridge
        if (self.is_my_tile(board, (move.x - 1, move.y - 1)) and
            self.is_valid_move(board, (move.x - 1, move.y)) and
            self.is_valid_move(board, (move.x, move.y - 1))):
            two_bridges.append((Move(move.x - 1, move.y - 1),
                                (Move(move.x - 1, move.y), Move(move.x, move.y - 1)), 
                                move))

        # Pattern 2: Top-right bridge
        if (self.is_my_tile(board, (move.x + 1, move.y - 2)) and
            self.is_valid_move(board, (move.x, move.y - 1)) and
            self.is_valid_move(board, (move.x + 1, move.y - 1))):
            two_bridges.append((Move(move.x + 1, move.y - 2),
                                (Move(move.x, move.y - 1), Move(move.x + 1, move.y - 1)), 
                                move))

        # Pattern 3: Right bridge
        if (self.is_my_tile(board, (move.x + 2, move.y - 1)) and
            self.is_valid_move(board, (move.x + 1, move.y - 1)) and
            self.is_valid_move(board, (move.x + 1, move.y))):
            two_bridges.append((Move(move.x + 2, move.y - 1),
                                (Move(move.x + 1, move.y - 1), Move(move.x + 1, move.y)), 
                                move))

        # Pattern 4: Bottom-right bridge
        if (self.is_my_tile(board, (move.x + 1, move.y + 1)) and
            self.is_valid_move(board, (move.x + 1, move.y)) and
            self.is_valid_move(board, (move.x, move.y + 1))):
            two_bridges.append((Move(move.x + 1, move.y + 1),
                                (Move(move.x + 1, move.y), Move(move.x, move.y + 1)), 
                                move))

        # Pattern 5: Bottom-left bridge
        if (self.is_my_tile(board, (move.x - 1, move.y + 2)) and
            self.is_valid_move(board, (move.x, move.y + 1)) and
            self.is_valid_move(board, (move.x - 1, move.y + 1))):
            two_bridges.append((Move(move.x - 1, move.y + 2),
                                (Move(move.x, move.y + 1), Move(move.x - 1, move.y + 1)), 
                                move))

        # Pattern 6: Left bridge
        if (self.is_my_tile(board, (move.x - 2, move.y + 1)) and
            self.is_valid_move(board, (move.x - 1, move.y + 1)) and
            self.is_valid_move(board, (move.x - 1, move.y))):
            two_bridges.append((Move(move.x - 2, move.y + 1),
                                (Move(move.x - 1, move.y + 1), Move(move.x - 1, move.y)), 
                                move))
            
        return two_bridges
    
    def bridge_direction(self, bridge: tuple[Move, tuple[Move, Move], Move]) -> str:
        """ Returns the direction of a bridge. Vertical if going up/down, horizontal if left/right, diagonal if in between. """
        bridge_pos1, empty_cells, bridge_pos2 = bridge

        if abs(bridge_pos1.x - bridge_pos2.x) >= 2:
            return "vertical"
        elif abs(bridge_pos1.y - bridge_pos2.y) >= 2:
            return "horizontal"
        else:
            return "diagonal"
        
    def get_two_bridges_score(self, board: Board, move: Move) -> float:
        """ 
        Return a "two bridge" score that is calculated based on:
            +parallel_two_bridge_weight for each two bridge created by making the given move in the intended direction (e.g. top-bottom or left-right). 

            +perpendicular_two_bridge_weight for each two bridge created by making the given move in the unintended direction 
            (e.g. top-bottom or left-right). This is less than parallel_two_bridge_weight, but still better than a random move.

            +diagonal_two_bridge_weight for each two bridge created by making the given move in the middle of the intended and 
            unintended direction. This is less than parallel_two_bridge_weight, but better than perpendicular_two_bridge_weight.

            +opponent_bridge_block for every opponent two bridge that is blocked by the given move.

        The more two bridges, the better the move. The more opponent two 
        bridges that are blocked, the better the move.

        A move will have a higher score if it creates multiple two bridges.

        This function assumes move is a valid move.
        """

        two_bridges_score = 0

        # Creates a two bridge for current player
        player_bridges = self.get_two_bridges(board, self._colour, move)
        for bridge in player_bridges:
            if (self._colour == Colour.RED and self.bridge_direction(bridge) == 'vertical') or \
               (self._colour == Colour.BLUE and self.bridge_direction(bridge) == 'horizontal'):
                # Greatest weighting for parallel bridges (those that travel the furthest in the intended direction)
                two_bridges_score += self.weights['parallel_two_bridge_weight']
            elif (self._colour == Colour.RED and self.bridge_direction(bridge) == 'horizontal') or \
               (self._colour == Colour.BLUE and self.bridge_direction(bridge) == 'vertical'):
                # Least weighting for perpendicular bridges (those that travel the furthest in the unintended direction)
                two_bridges_score += self.weights['perpendicular_two_bridge_weight']
            else:
                # Slightly less weighting for diagonal bridges (those that move left/right in the intended direction)
                two_bridges_score += self.weights['diagonal_two_bridge_weight']

        # Blocks opponent two bridges
        # This is not worth as much as prioritising our own bridges
        opponent = self._colour.opposite()
        two_bridges_score += (len(self.get_two_bridges(board, opponent, move)) * self.weights['opponent_bridge_block'])

        return two_bridges_score
    
    def get_connection_score(self, board: Board, move: Move) -> float:
        """ 
        Calculates a "connection" score that is calculated based on:
            +connection_weight for each tile that is connected to another tile by making the given move.

        The more tiles that are connected, the better the move. This ignores already connected tiles.
        """

        connection_score = 0

        # Get all neighbours (that we own, not interested in opponent tiles) of the move
        neighbours = self.get_neighbours_of_colour(board, move, self._colour)

        # Find all possible ways of pairing the neighbours
        neighbour_pairs = list(combinations(neighbours, 2))

        # For all pairs of neighbours,
        for pair in neighbour_pairs:
            n1, n2 = pair

            # If n1 and n2 are not already neighbours, then adding the move will connect them
            if not self.is_neighbour(n1, n2):
                connection_score += self.weights['connection_weight']

        return connection_score

    def find_two_bridge_saving_moves(self, board: Board) -> list[Move]:
        """ 
        Returns a list of moves to save any threatened two bridges on the board. 
        
        This function searches the whole board and finds two bridges that are threatened 
        by the opponent, and then returns a list of moves that can save them.
        """

        saving_moves = []

        for x in range(board.size):
            for y in range(board.size):
                if self.is_my_tile(board, (x, y)):
                    # Pattern 1: Top-left bridge
                    if (self.is_my_tile(board, (x - 1, y - 1)) and 
                        self.is_opponent_tile(board, (x - 1, y)) and 
                        self.is_valid_move(board, (x, y - 1))):
                        saving_moves.append(Move(x, y - 1))
                    
                    # Pattern 1: Top-left bridge
                    if (self.is_my_tile(board, (x - 1, y - 1)) and 
                        self.is_valid_move(board, (x - 1, y)) and 
                        self.is_opponent_tile(board, (x, y - 1))):
                        saving_moves.append(Move(x - 1, y))

                    # Pattern 2: Top-right bridge
                    if (self.is_my_tile(board, (x + 1, y - 2)) and
                        self.is_opponent_tile(board, (x, y - 1)) and
                        self.is_valid_move(board, (x + 1, y - 1))):
                        saving_moves.append(Move(x + 1, y - 1))

                    # Pattern 2: Top-right bridge
                    if (self.is_my_tile(board, (x + 1, y - 2)) and
                        self.is_valid_move(board, (x, y - 1)) and
                        self.is_opponent_tile(board, (x + 1, y - 1))):
                        saving_moves.append(Move(x, y - 1))

                    # Pattern 3: Right bridge
                    if (self.is_my_tile(board, (x + 2, y - 1)) and
                        self.is_opponent_tile(board, (x + 1, y - 1)) and
                        self.is_valid_move(board, (x + 1, y))):
                        saving_moves.append(Move(x + 1, y))

                    # Pattern 3: Right bridge
                    if (self.is_my_tile(board, (x + 2, y - 1)) and
                        self.is_valid_move(board, (x + 1, y - 1)) and
                        self.is_opponent_tile(board, (x + 1, y))):
                        saving_moves.append(Move(x + 1, y - 1))

                    # Pattern 4: Bottom-right bridge
                    if (self.is_my_tile(board, (x + 1, y + 1)) and
                        self.is_opponent_tile(board, (x + 1, y)) and
                        self.is_valid_move(board, (x, y + 1))):
                        saving_moves.append(Move(x, y + 1))

                    # Pattern 4: Bottom-right bridge
                    if (self.is_my_tile(board, (x + 1, y + 1)) and
                        self.is_valid_move(board, (x + 1, y)) and
                        self.is_opponent_tile(board, (x, y + 1))):
                        saving_moves.append(Move(x + 1, y))

                    # Pattern 5: Bottom-left bridge
                    if (self.is_my_tile(board, (x - 1, y + 2)) and
                        self.is_opponent_tile(board, (x, y + 1)) and
                        self.is_valid_move(board, (x - 1, y + 1))):
                        saving_moves.append(Move(x - 1, y + 1))

                    # Pattern 5: Bottom-left bridge
                    if (self.is_my_tile(board, (x - 1, y + 2)) and
                        self.is_valid_move(board, (x, y + 1)) and
                        self.is_opponent_tile(board, (x - 1, y + 1))):
                        saving_moves.append(Move(x, y + 1))

                    # Pattern 6: Left bridge
                    if (self.is_my_tile(board, (x - 2, y + 1)) and
                        self.is_opponent_tile(board, (x - 1, y + 1)) and
                        self.is_valid_move(board, (x - 1, y))):
                        saving_moves.append(Move(x - 1, y))

                    # Pattern 6: Left bridge
                    if (self.is_my_tile(board, (x - 2, y + 1)) and
                        self.is_valid_move(board, (x - 1, y + 1)) and
                        self.is_opponent_tile(board, (x - 1, y))):
                        saving_moves.append(Move(x - 1, y + 1))

        return saving_moves
