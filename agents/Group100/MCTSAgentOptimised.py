from math import log, sqrt
import random
import time

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move


class MCTSNode:
    """
    A node in the Monte Carlo Tree Search (MCTS) tree.
    Each node represents a game state and holds statistics for UCT calculations.
    """
    def __init__(self, board: Board, parent=None, move=None):
        self.board = board
        self.parent = parent
        self.move = move
        self.children = []
        self.wins = 0
        self.visits = 0
        self.unexplored_children = self.get_possible_moves(board)
        self.board_hash = None  # Store board hash for caching or transposition
        self.creates_two_bridge = False  # Flag if this move leads to a two-bridge formation

    def get_possible_moves(self, board: Board) -> list[Move]:
        """Get all valid moves at the current state."""
        valid_moves = []
        for x in range(board.size):
            for y in range(board.size):
                if board.tiles[x][y].colour is None:
                    valid_moves.append(Move(x, y))
        return valid_moves

    def calculate_uct_score(self, c_param=1.4) -> float:
        """
        Calculate the UCT (Upper Confidence Bound applied to Trees) score.
        """
        if self.visits == 0:
            return float('inf')
        exploitation = self.wins / self.visits
        exploration = sqrt((2 * log(self.parent.visits)) / self.visits)
        return exploitation + c_param * exploration

    def copy_board(self, board: Board) -> Board:
        """Create an efficient copy of the board state using list comprehension."""
        new_board = Board(board.size)
        new_board._winner = board._winner
        new_board.tiles = [[type(board.tiles[i][j])(board.tiles[i][j].colour) 
                           for j in range(board.size)] 
                          for i in range(board.size)]
        return new_board

    def apply_move(self, move: Move, colour: Colour) -> Board:
        """Apply a move to the board copy and return the updated state."""
        new_board = self.copy_board(self.board)
        new_board.tiles[move.x][move.y].colour = colour
        return new_board


class MCTSAgent(AgentBase):
    """
    A Monte Carlo Tree Search based agent that uses heuristics and caching 
    to select moves in a game of Hex.
    """
    _board_size: int = 11
    _colour: Colour

    # Two-bridge patterns defined as offsets from a reference stone:
    # Each pattern:
    # ((bridge_x_off, bridge_y_off), (cell1_x_off, cell1_y_off), (cell2_x_off, cell2_y_off))
    BRIDGE_PATTERNS = [
        ((-1, -1), (-1, 0), (0, -1)),    # Top-left
        ((1, -2), (0, -1), (1, -1)),     # Top-right
        ((2, -1), (1, -1), (1, 0)),      # Right
        ((1, 1), (1, 0), (0, 1)),        # Bottom-right
        ((-1, 2), (0, 1), (-1, 1)),      # Bottom-left
        ((-2, 1), (-1, 1), (-1, 0)),     # Left
    ]

    DIRECTION_VERTICAL = "vertical"
    DIRECTION_HORIZONTAL = "horizontal"
    DIRECTION_DIAGONAL = "diagonal"

    # Pre-computed neighbor patterns as (x_offset, y_offset) from the center tile
    # In Hex, each tile has 6 neighbors in this pattern:
    #    NW  NE
    #  W   *   E
    #    SW  SE
    NEIGHBOR_PATTERNS = [
        (-1, 0),   # West
        (1, 0),    # East
        (0, -1),   # Northwest
        (1, -1),   # Northeast
        (-1, 1),   # Southwest
        (0, 1),    # Southeast
    ]

    def __init__(self, colour: Colour):
        super().__init__(colour)
        self._colour = colour

        # Initialize Zobrist hashing for board states
        random.seed(42)
        self.zobrist_table = {}
        for x in range(self._board_size):
            for y in range(self._board_size):
                self.zobrist_table[(x, y, Colour.RED)] = random.getrandbits(64)
                self.zobrist_table[(x, y, Colour.BLUE)] = random.getrandbits(64)
        self.transposition_table = {}

        # Cache for bridge analysis and other board-related computations
        self.bridge_cache = {}

        # Weights or parameters (can be tuned)
        self.weights = {
            'center_weight': 0.4,  # Reduced center weight
            'opponent_bridge_block': 3.0,
            'parallel_two_bridge_weight': 6.0,  # Increased bridge weights
            'perpendicular_two_bridge_weight': 4.0,
            'diagonal_two_bridge_weight': 5.0,
            'friendly_neighbour': 0.5,     # Friendly neighbors are good
            'enemy_neighbour': -0.8,       # Enemy neighbors are very bad
            'empty_neighbour': 0.3,        # Empty neighbors provide opportunity
            'direction_bonus': 0.4        # Bonus for neighbors in winning direction
        }

        # Add size-based caching
        self.neighbor_cache = {}  # Cache for get_neighbors results
        self.evaluation_cache = {}  # Cache for move evaluations
        self.max_cache_size = 10000  # Prevent memory issues

    def _clear_bridge_cache(self, board: Board):
        """Clear the cached two-bridge analysis when the board changes."""
        self.bridge_cache.clear()

    def _get_cache_key(self, board: Board, colour: Colour) -> int:
        """Generate a cache key from the board state and colour."""
        return hash((self.hash_board(board), colour))

    def get_possible_moves(self, board: Board) -> list[Move]:
        """Get all valid moves on the given board."""
        valid_moves = []
        for x in range(board.size):
            for y in range(board.size):
                if board.tiles[x][y].colour is None:
                    valid_moves.append(Move(x, y))
        return valid_moves

    def is_my_stone(self, board: Board, position: Move | tuple[int, int]) -> bool:
        """Check if the given position is occupied by this agent's colour."""
        x, y = (position.x, position.y) if isinstance(position, Move) else position
        return (0 <= x < board.size and 0 <= y < board.size and
                board.tiles[x][y].colour == self._colour)

    def check_immediate_win(self, board: Board, position: Move | tuple[int, int], player: Colour) -> bool:
        """Check if placing a stone at 'position' immediately wins the game for 'player'."""
        x, y = (position.x, position.y) if isinstance(position, Move) else position
        if not self.is_valid_move(board, x, y):
            return False
        board.set_tile_colour(x, y, player)
        won = board.has_ended(player)
        board.set_tile_colour(x, y, None)
        return won

    def get_all_positions_for_colour(self, board: Board, colour: Colour) -> list[Move]:
        """Return all positions currently occupied by 'colour'."""
        positions = []
        for x in range(board.size):
            for y in range(board.size):
                if board.tiles[x][y].colour == colour:
                    positions.append(Move(x, y))
        return positions

    def copy_board(self, board: Board) -> Board:
        """Create a copy of the board state."""
        new_board = Board(board.size)
        new_board._winner = board._winner
        for i in range(board.size):
            for j in range(board.size):
                new_board.tiles[i][j].colour = board.tiles[i][j].colour
        return new_board

    def is_valid_move(self, board: Board, x: int, y: int) -> bool:
        """Check if placing a stone at (x, y) is valid (i.e., tile is empty)."""
        return 0 <= x < board.size and 0 <= y < board.size and board.tiles[x][y].colour is None

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        """
        Decide which move to make using MCTS and heuristics.
        """
        start_time = time.time()
        max_time = 4.9

        # Check swap logic early in the game
        if turn == 2 and opp_move and self.should_we_swap(opp_move):
            return Move(-1, -1)

        valid_moves = self.get_possible_moves(board)

        # Immediate win
        for move in valid_moves:
            if self.check_immediate_win(board, (move.x, move.y), self._colour):
                return move

        # Immediate block (if opponent is about to win)
        for move in valid_moves:
            if self.check_immediate_win(board, (move.x, move.y), Colour.opposite(self._colour)):
                return move

        # Attempt to save threatened two-bridges if possible
        if opp_move:
            save_move = self.save_two_bridge(board, opp_move)
            if save_move:
                return random.choice(save_move)

        # MCTS Initialization
        root = MCTSNode(board)
        iterations = 0

        # Run MCTS until time runs out
        while time.time() - start_time < max_time:
            current_node = self.select_node(root)
            current_node = self.expand_node(current_node)
            result = self.simulate(current_node.board)
            self.backpropagate(current_node, result)
            iterations += 1
            
        # print(f"MCTS iterations: {iterations}")

        # Choose the move of the child with the highest visit count
        best_child = max(root.children, key=lambda c: c.visits)
        return Move(best_child.move.x, best_child.move.y)

    def select_node(self, current_node: MCTSNode) -> MCTSNode:
        """Selection step of MCTS: navigate down the tree using UCT."""
        while not current_node.unexplored_children and current_node.children:
            current_node = max(current_node.children, key=lambda child: child.calculate_uct_score())
        return current_node

    def expand_node(self, node: MCTSNode) -> MCTSNode:
        """Expansion step of MCTS: expand a node by exploring an unexplored move."""
        if node.unexplored_children:
            # First, analyze bridges to see if any two-bridge moves are available
            
            # _, possible_bridges = self.analyze_two_bridges(node.board, self._colour)
            # # If two-bridge moves exist, bias towards them
            # candidate_moves = [m for m in node.unexplored_children if any(m.x == pb.x and m.y == pb.y for pb in possible_bridges)]
            
            # if candidate_moves:
            #     move = random.choice(candidate_moves)
            # else:
            #     move = random.choice(node.unexplored_children)
            
            node.unexplored_children = sorted(node.unexplored_children, 
                                                key=lambda move: self.evaluate_move(node.board, move), 
                                                reverse=True)
            move = node.unexplored_children.pop(0)
            new_board = node.apply_move(move, self._colour)
            child = MCTSNode(new_board, node, move)
            child.board_hash = self.hash_board(new_board)
            # child.creates_two_bridge = any((move.x == pb.x and move.y == pb.y) for pb in possible_bridges)
            node.children.append(child)
            return child
        return node

    def simulate(self, state: Board) -> bool:
        """
        Simulation step of MCTS: Run a random (or semi-random) play-out from the given state.
        Uses transposition table to avoid repeated calculations.
        """
        current_hash = self.hash_board(state)
        if current_hash in self.transposition_table:
            return self.transposition_table[current_hash]

        simulation_board = self.copy_board(state)
        simulation_colour = self._colour
        moves = self.get_possible_moves(simulation_board)

        while not simulation_board.has_ended(Colour.RED) and not simulation_board.has_ended(Colour.BLUE):
            if not moves:
                break
            move = random.choice(moves)
            simulation_board.set_tile_colour(move.x, move.y, simulation_colour)
            moves.remove(move)
            simulation_colour = Colour.opposite(simulation_colour)

        result = (simulation_board._winner == self._colour)
        self.transposition_table[current_hash] = result
        return result

    def hash_board(self, board: Board) -> int:
        """Compute Zobrist hash for the current board state."""
        h = 0
        for x in range(board.size):
            for y in range(board.size):
                color = board.tiles[x][y].colour
                if color is not None:
                    h ^= self.zobrist_table[(x, y, color)]
        return h

    def hash_update(self, hash_val: int, x: int, y: int, color: Colour) -> int:
        """Update Zobrist hash with a move."""
        return hash_val ^ self.zobrist_table[(x, y, color)]

    def backpropagate(self, node: MCTSNode, won: bool):
        """Backpropagation step of MCTS: update node statistics up the tree."""
        while node:
            node.visits += 1
            if won:
                node.wins += 1
            node = node.parent

    def analyze_two_bridges(self, board: Board, colour: Colour) -> tuple[list[tuple[Move, tuple[Move, Move]]], list[Move]]:
        """
        Analyze the board for two-bridges related to the given colour.

        Returns:
          existing_bridges: list of tuples (bridge_pos, (empty1, empty2)) where bridge_pos is a Move of an occupied tile 
                            that forms a two-bridge with two empty cells.
          possible_bridges: list of Moves representing positions where placing a tile would create a two-bridge.
        """
        cache_key = ("analysis", self._get_cache_key(board, colour))
        if cache_key in self.bridge_cache:
            return self.bridge_cache[cache_key]

        existing_bridges = []
        possible_bridges = set()
        current_positions = self.get_all_positions_for_colour(board, colour)

        for node in current_positions:
            mx, my = node.x, node.y
            for (bx_off, by_off), (c1x_off, c1y_off), (c2x_off, c2y_off) in self.BRIDGE_PATTERNS:
                bx, by = mx + bx_off, my + by_off
                c1x, c1y = mx + c1x_off, my + c1y_off
                c2x, c2y = mx + c2x_off, my + c2y_off

                # Check for existing bridge
                if (self.is_my_stone(board, (bx, by)) and
                    self.is_valid_move(board, c1x, c1y) and
                    self.is_valid_move(board, c2x, c2y)):
                    existing_bridges.append((Move(bx, by), (Move(c1x, c1y), Move(c2x, c2y))))

                # Check for possible two-bridge creation
                # If placing a stone at (bx, by) plus filling c1 and c2 would form a bridge
                if (self.is_valid_move(board, bx, by) and
                    self.is_valid_move(board, c1x, c1y) and
                    self.is_valid_move(board, c2x, c2y)):
                    possible_bridges.add(Move(bx, by))

        # Convert possible_bridges to list for consistent return type
        possible_bridges = list(possible_bridges)
        self.bridge_cache[cache_key] = (existing_bridges, possible_bridges)
        return existing_bridges, possible_bridges

    def get_possible_two_bridges(self, board: Board, colour: Colour) -> list[Move]:
        """Get all possible moves that could create a two-bridge if placed."""
        _, possible_bridges = self.analyze_two_bridges(board, colour)
        return possible_bridges

    def get_two_bridges_with_positions(self, board: Board, colour: Colour) -> list[tuple[Move, tuple[Move, Move]]]:
        """Get all existing two-bridges along with the exact empty cells they depend on."""
        existing_bridges, _ = self.analyze_two_bridges(board, colour)
        return existing_bridges

    def save_two_bridge(self, board: Board, opp_move: Move) -> list[Move] | None:
        """
        Attempt to save a threatened two-bridge after the opponent's move.
        
        Steps:
        1. Capture current existing bridges.
        2. Temporarily remove opponent's move and re-check bridges.
        3. Restore move and find which bridges were lost.
        4. Identify moves that can restore these lost bridges.
        """
        x, y = opp_move.x, opp_move.y
        original_color = board.tiles[x][y].colour

        # Current existing bridges
        current_existing, _ = self.analyze_two_bridges(board, self._colour)
        current_bridge_positions = {bridge[0] for bridge in current_existing}

        # Temporarily remove opponent's move
        board.set_tile_colour(x, y, None)
        self._clear_bridge_cache(board)

        # Bridges before opponent's move
        previous_existing, _ = self.analyze_two_bridges(board, self._colour)

        # Restore opponent's move
        board.set_tile_colour(x, y, original_color)
        self._clear_bridge_cache(board)

        moves_to_save = set()

        # Find bridges that existed before but not now -> threatened
        for bridge_pos, empty_cells in previous_existing:
            if bridge_pos not in current_bridge_positions:
                cell1, cell2 = empty_cells
                # Identify which cell was taken by the opponent and try to save with the other
                if (x, y) == (cell1.x, cell1.y) and self.is_valid_move(board, cell2.x, cell2.y):
                    moves_to_save.add(Move(cell2.x, cell2.y))
                elif (x, y) == (cell2.x, cell2.y) and self.is_valid_move(board, cell1.x, cell1.y):
                    moves_to_save.add(Move(cell1.x, cell1.y))

        return list(moves_to_save) if moves_to_save else None

    def should_we_swap(self, opp_move: Move) -> bool:
        """
        Determine if we should initiate a 'swap' based on known strategies.
        """
        # Pre-defined moves where swapping is considered advantageous
        swap_moves = []
        for x in range(2, 9):
            for y in range(11):
                # Avoid certain corners
                if not (x == 2 and y == 0) and not (x == 8 and y == 0) \
                   and not (x == 2 and y == 10) and not (x == 8 and y == 10):
                    swap_moves.append(Move(x, y))
        
        # Additional known good swap moves
        swap_moves.extend([
            Move(0, 10), Move(1, 9), Move(1, 10),
            Move(9, 0), Move(10, 0), Move(9, 1)
        ])

        # On turn 2, if agent is BLUE and the opponent's move matches a swap configuration
        if opp_move is not None and self._colour == Colour.BLUE:
            if opp_move in swap_moves:
                return True
        return False
    
    
    # NEW METHODS TO IMPLEMENT
    
    def bridge_direction(self, bridge_pos1: Move, bridge_pos2: Move) -> str:
        """
        Determine the direction between two bridge positions.
        Returns: DIRECTION_VERTICAL, DIRECTION_HORIZONTAL, or DIRECTION_DIAGONAL
        """
        dx = abs(bridge_pos1.x - bridge_pos2.x)
        dy = abs(bridge_pos1.y - bridge_pos2.y)
        
        if dx >= 2:
            return self.DIRECTION_VERTICAL
        elif dy >= 2:
            return self.DIRECTION_HORIZONTAL
        return self.DIRECTION_DIAGONAL

    def get_two_bridges_score(self, board: Board, move: Move) -> float:
        """Calculate score for two bridges created by this move."""
        score = 0.0
        
        # Get bridges before move
        existing_before, _ = self.analyze_two_bridges(board, self._colour)
        
        # Apply move and analyze new state
        board.set_tile_colour(move.x, move.y, self._colour)
        existing_after, possible_after = self.analyze_two_bridges(board, self._colour)
        
        # Look for any new bridges created
        for bridge_pos, (empty1, empty2) in existing_after:
            # Only count new bridges not present before
            if not any(before[0].x == bridge_pos.x and before[0].y == bridge_pos.y for before in existing_before):
                # Check bridge direction
                if abs(bridge_pos.x - move.x) >= 2:
                    score += self.weights['parallel_two_bridge_weight'] if self._colour == Colour.RED else self.weights['perpendicular_two_bridge_weight']
                elif abs(bridge_pos.y - move.y) >= 2:
                    score += self.weights['parallel_two_bridge_weight'] if self._colour == Colour.BLUE else self.weights['perpendicular_two_bridge_weight']
                else:
                    score += self.weights['diagonal_two_bridge_weight']
        
        # Check opponent bridges blocked
        opp_colour = Colour.opposite(self._colour)
        opp_before, _ = self.analyze_two_bridges(board, opp_colour)
        board.set_tile_colour(move.x, move.y, None)  # Reset board state
        
        # Add blocking score
        score += len(opp_before) * self.weights['opponent_bridge_block']
        
        return score

    def evaluate_move(self, board: Board, move: Move) -> float:
        """
        Evaluates quality of a move using position and bridge potential.
        Returns: float score where higher is better
        """
        cache_key = (self.hash_board(board), move.x, move.y)
        if cache_key in self.evaluation_cache:
            return self.evaluation_cache[cache_key]

        score = 0.0
        center = board.size // 2
        
        # Center proximity score
        dist_to_center = abs(move.x - center) + abs(move.y - center)
        score += (max(0, (board.size - dist_to_center)) / board.size) * self.weights['center_weight']
        
        # Bridge score
        score += self.get_two_bridges_score(board, move)
        
        # Inferior cell analysis
        score += self.get_inferior_cell_analysis_score(board, move)
        
        if len(self.evaluation_cache) < self.max_cache_size:
            self.evaluation_cache[cache_key] = score
        
        return score
    
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

    def get_neighbors(self, board: Board, move: Move) -> list[tuple[int, int]]:
        """
        Get all valid neighboring positions for a given move.
        
        Args:
            board: Current game board
            move: Move to find neighbors for
        
        Returns:
            List of (x,y) tuples representing valid neighbor positions
        """
        cache_key = (move.x, move.y)
        if cache_key in self.neighbor_cache:
            return self.neighbor_cache[cache_key]

        neighbors = []
        
        # Check each neighbor pattern
        for x_offset, y_offset in self.NEIGHBOR_PATTERNS:
            neighbor_x = move.x + x_offset
            neighbor_y = move.y + y_offset
            
            # Verify neighbor is within board boundaries
            if (0 <= neighbor_x < board.size and 
                0 <= neighbor_y < board.size):
                neighbors.append((neighbor_x, neighbor_y))
        
        if len(self.neighbor_cache) < self.max_cache_size:
            self.neighbor_cache[cache_key] = neighbors
        
        return neighbors
    
    def get_inferior_cell_analysis_score(self, board: Board, move: Move) -> float:
        """
        Evaluate the quality of a move based on neighboring cells.
        Considers:
        - Friendly/enemy/empty neighbors
        - Strategic direction importance
        - Connection potential
        
        Returns: float score where higher is better
        """
        score = 0.0
        neighbors = self.get_neighbors(board, move)
        
        for neighbor_x, neighbor_y in neighbors:
            neighbor_color = board.tiles[neighbor_x][neighbor_y].colour
            
            # Check neighbor type
            if neighbor_color is None:
                score += self.weights['empty_neighbour']
            elif neighbor_color == self._colour:
                score += self.weights['friendly_neighbour']
            else:  # Enemy color
                score += self.weights['enemy_neighbour']
            
            # Add directional bonus for neighbors in winning direction
            if self._colour == Colour.RED:
                # Red wants vertical connections
                if abs(neighbor_x - move.x) > 0:
                    score += self.weights['direction_bonus']
            else:  # BLUE
                # Blue wants horizontal connections
                if abs(neighbor_y - move.y) > 0:
                    score += self.weights['direction_bonus']
                    
        return score
