from math import log, sqrt
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
        self.board_hash = None  # Store board hash for this node
        self.creates_two_bridge = False  # Flag to indicate if this move creates a two-bridge

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
            'center_weight': 1.403,
            'neighbor_weight': 0.943,
            'bridge_weight': 0.943,
            'edge_weight': 0.943,
            'defensive_weight': 6.420,
            'two_bridge_weight': 6.421,
            'opponent_bridge_block': 6.418,
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
        """Get all valid moves."""
        valid_moves = []
        for x in range(board.size):
            for y in range(board.size):
                if board.tiles[x][y].colour is None:
                    valid_moves.append(Move(x, y))
        return valid_moves

    def is_my_stone(self, board: Board, move: Move | tuple[int, int]) -> bool:
        x, y = (move.x, move.y) if isinstance(move, Move) else move
        return (0 <= x < board.size and 0 <= y < board.size and
                board.tiles[x][y].colour is self._colour)

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
    
    def evaluate_move(self, board: Board, move: tuple[int, int], colour: Colour) -> float:
        """ 
        Evaluates the quality of a given move for the specified player colour. 
        
        Returns a score where a higher score represents a better move.
        """

        if move in self.move_scores:
            return self.move_scores[move]
        
        score = 0  # Initialize score
        x, y = move
        center = board.size // 2
        
        # Prioritise moves closer to the center
        dist_to_center = abs(x - center) + abs(y - center)
        score += (max(0, (board.size - dist_to_center)) / board.size) * self.weights['center_weight']

        self.move_scores[move] = score
        print("Evaluate move score for " + str(colour) + ":", score)
        return score

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        start_time = time.time()
        max_time = 4.9  # Used to ensure the agent never times out

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
            save_move = self.save_two_bridge(board, opp_move)
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

        print(f"MCTS Iterations: {self.current_simulation}, Time Spent: {time.time() - start_time:.2f}s")

        best_child = max(root.children, key=lambda c: c.visits)


        # Strategic Swap Evaluation - should we swap?
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
        if turn == 2 and opp_move is not None and self._colour == Colour.BLUE:
            if opp_move in swap_moves:
                return Move(-1, -1)

        # If no swap,
        return Move(best_child.move.x, best_child.move.y)

    def select_node(self, current_node: MCTSNode) -> MCTSNode:
        while not current_node.unexplored_children and current_node.children:
            current_node = max(current_node.children, key=lambda child: child.calculate_uct_score())
        return current_node

    def expand_node(self, node: MCTSNode) -> MCTSNode:
        if node.unexplored_children:
            two_bridge_moves = self.get_possible_two_bridges(node.board, self.colour)

            # If two-bridge moves exist, pick one of them first for speed
            # TODO This is questionable, as it might not always be the best move, will need to evaluate.
            if two_bridge_moves:
                # Filter out all two-bridge moves
                candidate_moves = [m for m in node.unexplored_children if any(m.x == tb.x and m.y == tb.y for tb in two_bridge_moves)]
                if candidate_moves:
                    move = random.choice(candidate_moves)
                else:
                    move = random.choice(node.unexplored_children)
            else:
                move = random.choice(node.unexplored_children)

            node.unexplored_children.remove(move)
            new_board = node.apply_move(move, self.colour)
            child = MCTSNode(new_board, node, move)
            child.board_hash = self.hash_board(new_board)
            child.creates_two_bridge = any((move.x == tb.x and move.y == tb.y) for tb in two_bridge_moves)
            node.children.append(child)
            return child
        return node

    def simulate(self, state: Board) -> bool:
        """Run a quick random simulation to the end from the given state."""
        # Check transposition table
        current_hash = self.hash_board(state)
        if current_hash in self.transposition_table:
            return self.transposition_table[current_hash]

        # Simulate on a copy
        simulation_board = self.copy_board(state)
        simulation_colour = self._colour
        moves = self.get_possible_moves(simulation_board)

        # Fast random simulation without intermediate hashing
        while not simulation_board.has_ended(Colour.RED) and not simulation_board.has_ended(Colour.BLUE):
            if not moves:
                break
            move = random.choice(moves)
            simulation_board.set_tile_colour(move.x, move.y, simulation_colour)
            moves.remove(move)
            simulation_colour = Colour.opposite(simulation_colour)

        result = (simulation_board._winner == self.colour)
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

    def get_possible_two_bridges(self, board: Board, colour: Colour) -> list[Move]:
        """ Get all possible two-bridge moves. """
        two_bridges = []
        current_nodes = self.get_all_positions_for_colour(board, colour)

        # Using the same patterns as before
        if not current_nodes:
            return two_bridges

        for node in current_nodes:
            # Pattern 1: Top-left bridge
            if (self.is_valid_move(board, (node.x - 1, node.y - 1)) and
                self.is_valid_move(board, (node.x - 1, node.y)) and
                self.is_valid_move(board, (node.x, node.y - 1))):
                two_bridges.append(Move(node.x - 1, node.y - 1))

            # Pattern 2: Top-right bridge
            if (self.is_valid_move(board, (node.x + 1, node.y - 2)) and
                self.is_valid_move(board, (node.x, node.y - 1)) and
                self.is_valid_move(board, (node.x + 1, node.y - 1))):
                two_bridges.append(Move(node.x + 1, node.y - 2))

            # Pattern 3: Right bridge
            if (self.is_valid_move(board, (node.x + 2, node.y - 1)) and
                self.is_valid_move(board, (node.x + 1, node.y - 1)) and
                self.is_valid_move(board, (node.x + 1, node.y))):
                two_bridges.append(Move(node.x + 2, node.y - 1))

            # Pattern 4: Bottom-right bridge
            if (self.is_valid_move(board, (node.x + 1, node.y + 1)) and
                self.is_valid_move(board, (node.x + 1, node.y)) and
                self.is_valid_move(board, (node.x, node.y + 1))):
                two_bridges.append(Move(node.x + 1, node.y + 1))

            # Pattern 5: Bottom-left bridge
            if (self.is_valid_move(board, (node.x - 1, node.y + 2)) and
                self.is_valid_move(board, (node.x, node.y + 1)) and
                self.is_valid_move(board, (node.x - 1, node.y + 1))):
                two_bridges.append(Move(node.x - 1, node.y + 2))

            # Pattern 6: Left bridge
            if (self.is_valid_move(board, (node.x - 2, node.y + 1)) and
                self.is_valid_move(board, (node.x - 1, node.y + 1)) and
                self.is_valid_move(board, (node.x - 1, node.y))):
                two_bridges.append(Move(node.x - 2, node.y + 1))

        return two_bridges

    def get_two_bridges_with_positions(self, board: Board, colour: Colour) -> list[tuple[Move, tuple[Move, Move]]]:
        two_bridges = []
        current_nodes = self.get_all_positions_for_colour(board, colour)

        for node in current_nodes:
            # Pattern 1: Top-left bridge
            if (self.is_my_stone(board, (node.x - 1, node.y - 1)) and
                self.is_valid_move(board, (node.x - 1, node.y)) and
                self.is_valid_move(board, (node.x, node.y - 1))):
                two_bridges.append(((node.x - 1, node.y - 1),
                                    (Move(node.x - 1, node.y), Move(node.x, node.y - 1))))

            # Pattern 2: Top-right bridge
            if (self.is_my_stone(board, (node.x + 1, node.y - 2)) and
                self.is_valid_move(board, (node.x, node.y - 1)) and
                self.is_valid_move(board, (node.x + 1, node.y - 1))):
                two_bridges.append(((node.x + 1, node.y - 2),
                                    (Move(node.x, node.y - 1), Move(node.x + 1, node.y - 1))))

            # Pattern 3: Right bridge
            if (self.is_my_stone(board, (node.x + 2, node.y - 1)) and
                self.is_valid_move(board, (node.x + 1, node.y - 1)) and
                self.is_valid_move(board, (node.x + 1, node.y))):
                two_bridges.append(((node.x + 2, node.y - 1),
                                    (Move(node.x + 1, node.y - 1), Move(node.x + 1, node.y))))

            # Pattern 4: Bottom-right bridge
            if (self.is_my_stone(board, (node.x + 1, node.y + 1)) and
                self.is_valid_move(board, (node.x + 1, node.y)) and
                self.is_valid_move(board, (node.x, node.y + 1))):
                two_bridges.append(((node.x + 1, node.y + 1),
                                    (Move(node.x + 1, node.y), Move(node.x, node.y + 1))))

            # Pattern 5: Bottom-left bridge
            if (self.is_my_stone(board, (node.x - 1, node.y + 2)) and
                self.is_valid_move(board, (node.x, node.y + 1)) and
                self.is_valid_move(board, (node.x - 1, node.y + 1))):
                two_bridges.append(((node.x - 1, node.y + 2),
                                    (Move(node.x, node.y + 1), Move(node.x - 1, node.y + 1))))

            # Pattern 6: Left bridge
            if (self.is_my_stone(board, (node.x - 2, node.y + 1)) and
                self.is_valid_move(board, (node.x - 1, node.y + 1)) and
                self.is_valid_move(board, (node.x - 1, node.y))):
                two_bridges.append(((node.x - 2, node.y + 1),
                                    (Move(node.x - 1, node.y + 1), Move(node.x - 1, node.y))))

        return two_bridges

    def save_two_bridge(self, board: Board, opp_move: Move) -> Move | None:
        x, y = opp_move.x, opp_move.y
        moves_to_save = set()

        original_color = board.tiles[x][y].colour
        current_bridges = self.get_two_bridges_with_positions(board, self._colour)

        # Temporarily undo opponent's move
        board.set_tile_colour(x, y, None)

        previous_bridges = self.get_two_bridges_with_positions(board, self._colour)

        # Restore
        board.set_tile_colour(x, y, original_color)

        # Check each bridge that existed before
        current_bridge_positions = [b[0] for b in current_bridges]

        for bridge_pos, empty_cells in previous_bridges:
            if bridge_pos not in current_bridge_positions:
                cell1, cell2 = empty_cells
                if (x, y) == (cell1.x, cell1.y) and self.is_valid_move(board, (cell2.x, cell2.y)):
                    moves_to_save.add(Move(cell2.x, cell2.y))
                elif (x, y) == (cell2.x, cell2.y) and self.is_valid_move(board, (cell1.x, cell1.y)):
                    moves_to_save.add(Move(cell1.x, cell1.y))

        return list(moves_to_save) if moves_to_save else None
    