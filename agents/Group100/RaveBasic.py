from math import sqrt, log
from random import choice, random, Random
from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move
import heapq
import logging
import os

# Create logs directory if it doesn't exist
os.makedirs('agents/Group100/logs', exist_ok=True)

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create formatters and handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# File handler
file_handler = logging.FileHandler('agents/Group100/logs/rave_basic.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Prevent logger from propagating to root logger
logger.propagate = False

class SafeBoardContext:
    """Context manager for safe board state manipulation"""
    def __init__(self, board: Board, agent, operation: str):
        self.board = board
        self.agent = agent
        self.operation = operation
        self.moves_made = []
        self.initial_hash = agent.hash_board(board)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore board state on exit or exception
        for move, player in reversed(self.moves_made):
            self.board.set_tile_colour(move[0], move[1], None)
        
        final_hash = self.agent.hash_board(self.board)
        if final_hash != self.initial_hash:
            logger.error(f"Board corruption detected in {self.operation}!")
            logger.error(f"Initial hash: {self.initial_hash}, Final hash: {final_hash}")
            raise RuntimeError("Board state corruption detected")

    def apply_move(self, move: tuple[int, int], player: Colour):
        self.board.set_tile_colour(move[0], move[1], player)
        self.moves_made.append((move, player))

class RaveMCTSNode:
    """
    Enhanced MCTS node implementing RAVE (Rapid Action Value Estimation).
    """
    def __init__(self, parent=None, move=None):
        self.parent = parent
        self.move = move  # The move that led to this node
        self.children = []
        self.wins = 0  # Total wins from simulations passing through this node
        self.visits = 0  # Total simulations passing through this node
        self.untried_moves = None  # Moves that have not been tried from this node
        self.player = None  # The player who made the move to reach this node
        self.move_amaf_wins = {}  # Track AMAF wins per move
        self.move_amaf_visits = {}  # Track AMAF visits per move
        self.depth = 0 if parent is None else parent.depth + 1  # Track node depth

    def get_amaf_value(self, move: tuple[int, int]) -> float:
        """
        Calculate the AMAF value for a specific move using cached statistics.
        """
        visits = self.move_amaf_visits.get(move, 0)
        if visits == 0:
            return 0.0
        return self.move_amaf_wins.get(move, 0) / visits

    def ucb1(self, explore_constant, rave_constant, depth_decay) -> float:
        """
        Calculate node selection value using provided constants.
        """
        if self.visits == 0:
            return float('inf')

        # Apply decay to RAVE constant based on node depth
        decayed_rave = rave_constant * (depth_decay ** self.depth)
        beta = sqrt(decayed_rave / (3 * self.visits + decayed_rave))

        U = self.wins / self.visits  # Exploitation term
        E = explore_constant * sqrt(log(self.parent.visits) / self.visits)  # Exploration term
        mcts_value = U + E  # Include exploration inside the MCTS term
        amaf_value = self.get_amaf_value(self.move) if self.move else 0.0

        return (1 - beta) * mcts_value + beta * amaf_value

    def update_amaf_stats(self, move: tuple[int, int], won: bool):
        """Update AMAF statistics for a specific move"""
        if move not in self.move_amaf_visits:
            self.move_amaf_visits[move] = 0
            self.move_amaf_wins[move] = 0
        self.move_amaf_visits[move] += 1
        if won:
            self.move_amaf_wins[move] += 1

class RaveAgent(AgentBase):
    """
    RAVE-enhanced MCTS agent with specialized optimizations for Hex gameplay.
    """
    def __init__(self, colour: Colour):
        super().__init__(colour)
        self.weights = {
            'center_weight': 1.354,
            'neighbor_weight': 0.729,
            'bridge_weight': 0.897,
            'edge_weight': 0.897,
            'defensive_weight': 3.123,
            'two_bridge_weight': 6.123,
            'opponent_bridge_block': 6.122,
            'explore_constant': 1.923,
            'rave_constant': 327.61,
            'early_stop_threshold': 0.93,
            'min_visits_ratio': 0.135,
            'depth_decay': 0.909,
        }
        self.simulations = 1000
        self.win_score = 10  # Adjusted to 1 for win counts
        self.colour = colour
        self.root = RaveMCTSNode()
        self._board_size = 11
        self._all_positions = [(i, j) for i in range(self._board_size) 
                               for j in range(self._board_size)]
        self.rave_constant = self.weights['rave_constant']  # Tune this value
        self.move_history = []  # Track move history
        self.move_scores = {}  # Cache for move evaluations
        self.transposition_table = {}  # Cache for board states
        self.early_stop_threshold = self.weights['early_stop_threshold']  # Higher threshold for early stopping
        self.min_visits_ratio = self.weights['min_visits_ratio']  # Minimum visit ratio for early stopping
        self._rng = Random(42)  # Create a dedicated random number generator
        self._zobrist_table = {
            (i, j, color): self._rng.getrandbits(64)
            for i in range(self._board_size)
            for j in range(self._board_size)
            for color in [Colour.RED, Colour.BLUE]
        }
        self.opposing_colour = Colour.opposite(self.colour)
        self.critical_paths = {}  # Cache for critical path analysis
        self.opponent_bridges = {}  # Cache for opponent bridge positions
        self.defensive_scores = {}  # Cache for defensive position scores
        self.win_sequence_depth = 2  # Configurable depth for win sequence detection
        self.min_moves_for_sequence_check = 15  # Only check sequences after this many moves
        
        

    def switch_player(self, current_player: Colour) -> Colour:
        """Switch the current player"""
        if current_player == Colour.RED:
            return Colour.BLUE
        return Colour.RED

    def get_player_moves(self, current_player: Colour, move: tuple[int, int], 
                         red_moves: list, blue_moves: list) -> None:
        """Add move to appropriate player's move list"""
        if current_player == Colour.RED:
            red_moves.append(move)
        else:
            blue_moves.append(move)

    def get_valid_moves(self, board: Board) -> list[tuple[int, int]]:
        """Get all valid moves on the board"""
        return [(i, j) for i in range(board.size)
                for j in range(board.size)
                if board.tiles[i][j].colour is None]

    def get_neighbor_moves(self, board: Board, x: int, y: int) -> list[tuple[int, int]]:
        """Get valid moves adjacent to existing pieces"""
        neighbors = []
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,1), (1,-1)]:
            nx, ny = x + dx, y + dy
            if (0 <= nx < board.size and 0 <= ny < board.size and 
                board.tiles[nx][ny].colour is None):
                neighbors.append((nx, ny))
        return neighbors
    

    def evaluate_move(self, board: Board, move: tuple[int, int]) -> float:
        """Enhanced move evaluation with defensive considerations"""
        if move in self.move_scores:
            return self.move_scores[move]
        
        score = 0
        x, y = move
        center = board.size // 2
        
        # Strategic positioning
        dist_to_center = abs(x - center) + abs(y - center)
        score += max(0, (board.size - dist_to_center)) / board.size
        
        # Connection potential
        neighbors = self.get_neighbor_moves(board, x, y)
        friendly_neighbors = sum(1 for nx, ny in neighbors 
                               if board.tiles[nx][ny].colour == self.colour)
        score += friendly_neighbors * self.weights['neighbor_weight']
        
        # Bridge formation potential
        bridge_score = sum(1 for nx, ny in neighbors 
                         if abs(nx-x) == 1 and abs(ny-y) == 1 
                         and board.tiles[nx][ny].colour == self.colour)
        score += bridge_score * self.weights['bridge_weight']
        
        # Edge control
        if self.colour == Colour.RED and (x == 0 or x == board.size-1):
            score += self.weights['edge_weight']
        elif self.colour == Colour.BLUE and (y == 0 or y == board.size-1):
            score += self.weights['edge_weight']
            
        # Add defensive score
        defensive_score = self.evaluate_defensive_position(board, move)
        score += defensive_score
        
        # Add two bridge score
        two_bridge_score = self.get_two_bridges_score(board, move)
        score += two_bridge_score

        self.move_scores[move] = score
        return score

    def get_smart_moves(self, board: Board) -> list[tuple[int, int]]:
        """Enhanced move generation with defensive awareness"""
        occupied = [(i, j) for i, j in self._all_positions 
                   if board.tiles[i][j].colour is not None]
        
        if len(occupied) < 3:
            center = board.size // 2
            return [(center, center)] + self.get_neighbor_moves(board, center, center)
            
        neighbor_moves = set()
        for x, y in occupied:
            neighbor_moves.update(self.get_neighbor_moves(board, x, y))
            
        moves = list(neighbor_moves) if neighbor_moves else self.get_valid_moves(board)
        
        # Identify critical defensive moves
        opp_paths = self.find_critical_paths(board, self.opposing_colour)
        critical_moves = set()
        for path in opp_paths:
            critical_moves.update(m for m in path if board.tiles[m[0]][m[1]].colour is None)
        
        # Prioritize critical defensive moves
        if critical_moves:
            # Sort critical moves by evaluation
            critical_list = list(critical_moves)
            critical_list.sort(key=lambda m: self.evaluate_move(board, m), reverse=True)
            # Add remaining moves
            remaining = [m for m in moves if m not in critical_moves]
            remaining.sort(key=lambda m: self.evaluate_move(board, m), reverse=True)
            return critical_list + remaining
        
        # Sort regular moves by evaluation
        moves.sort(key=lambda m: self.evaluate_move(board, m), reverse=True)
        return moves

    def find_critical_paths(self, board: Board, player: Colour) -> list[list[tuple[int, int]]]:
        """Find potentially winning paths for a player"""
        board_hash = self.hash_board(board)
        cache_key = (board_hash, player)
        if cache_key in self.critical_paths:
            return self.critical_paths[cache_key]

        paths = []
        if player == Colour.RED:
            # Look for top-bottom connections
            for j in range(board.size):
                if board.tiles[0][j].colour == player:
                    path = self._trace_path(board, (0, j), player, 'vertical')
                    if path:
                        paths.append(path)
        else:
            # Look for left-right connections
            for i in range(board.size):
                if board.tiles[i][0].colour == player:
                    path = self._trace_path(board, (i, 0), player, 'horizontal')
                    if path:
                        paths.append(path)

        self.critical_paths[cache_key] = paths
        return paths
    
    def get_two_bridges_score(self, board: Board, move: tuple[int, int]) -> float:
        """ 
        Return a "two bridge" score that is calculated based on:
            +1 for each two bridge created by making the given move. 
            +0.5 for every opponent two bridge that is blocked by the given move.

        The more two bridges, the better the move. The more opponent two 
        bridges that are blocked, the better the move.

        A move will have a higher score if it creates multiple two bridges.

        This function assumes move is a valid move.
        """

        two_bridges_score = 0

        # Creates a two bridge for current player
        two_bridges_score += self.weights['two_bridge_weight'] * self.get_two_bridges(board, move, self.colour)

        # Blocks opponent two bridges
        # This is not worth as much as prioritising our own bridges
        opponent = self.colour.opposite()
        two_bridges_score += (self.get_two_bridges(board, move, opponent) * self.weights['opponent_bridge_block'])

        return two_bridges_score


    def get_two_bridge_positions(self, board: Board, move: tuple[int, int], colour: Colour) -> list[tuple[tuple[int, int], list[tuple[int, int]]]]:
        """
        Returns a list of two bridge positions, each containing:
        - The anchor stone position
        - List of the two empty cells that form the bridge path
        """
        x, y = move
        
        # Update patterns to include all valid Hex two bridge formations including diagonal bridges
        patterns = [
            # Original patterns
            [(-2, 0), (-1, 0), (-1, 1)],     # Top
            [(-1, 2), (-1, 1), (0, 1)],      # Top-right
            [(1, 1), (0, 1), (1, 0)],        # Bottom-right
            [(2, -1), (1, -1), (1, 0)],      # Bottom
            [(1, -2), (0, -1), (1, -1)],     # Bottom-left
            [(-1, -1), (-1, 0), (0, -1)],    # Top-left
            
            # Additional diagonal patterns
            [(2, 0), (1, 0), (1, 1)],        # Vertical down
            [(0, 2), (0, 1), (1, 1)],        # Horizontal right
            [(0, -2), (0, -1), (1, -1)],     # Horizontal left
            [(-2, 2), (-1, 1), (-1, 2)],     # Diagonal top-right
            [(2, -2), (1, -1), (2, -1)],     # Diagonal bottom-left
        ]
        
        bridges = []
        size = board.size
        
        def in_bounds(px, py):
            return 0 <= px < size and 0 <= py < size
        
        def is_valid_two_bridge(px, py, e1x, e1y, e2x, e2y):
            # Check that empty cells are adjacent
            if abs(e1x - e2x) + abs(e1y - e2y) != 1:
                return False
            # Check that empty cells are adjacent to respective stones
            if (abs(px - e1x) + abs(py - e1y) != 1) or (abs(x - e2x) + abs(y - e2y) != 1):
                return False
            return True
                
        for pattern in patterns:
            piece, empty1, empty2 = pattern
            px, py = x + piece[0], y + piece[1]
            e1x, e1y = x + empty1[0], y + empty1[1]
            e2x, e2y = x + empty2[0], y + empty2[1]
            
            if not (in_bounds(px, py) and in_bounds(e1x, e1y) and in_bounds(e2x, e2y)):
                continue
                
            if (board.tiles[px][py].colour == colour and
                board.tiles[e1x][e1y].colour is None and
                board.tiles[e2x][e2y].colour is None and
                is_valid_two_bridge(px, py, e1x, e1y, e2x, e2y)):
                
                bridges.append(
                    ((px, py), [(e1x, e1y), (e2x, e2y)])
                )
                    
        return bridges

    def get_two_bridges(self, board: Board, move: tuple[int, int], colour: Colour):
        """Returns the number of two bridges made by a given move for a given player."""
        return len(self.get_two_bridge_positions(board, move, colour))
    
    def copy_board(self, board: Board) -> Board:
        """Create an efficient copy of the board state."""
        new_board = Board(board.size)
        for i in range(board.size):
            for j in range(board.size):
                new_board.tiles[i][j].colour = board.tiles[i][j].colour
        return new_board

    def evaluate_defensive_position(self, board: Board, move: tuple[int, int]) -> float:
        """Evaluate move's defensive value"""
        board_hash = self.hash_board(board)
        cache_key = (board_hash, move)
        if cache_key in self.defensive_scores:
            return self.defensive_scores[cache_key]

        score = 0
        x, y = move

        # Check if move blocks opponent's critical paths
        opp_paths = self.find_critical_paths(board, self.opposing_colour)
        for path in opp_paths:
            if move in path:
                score += self.weights['defensive_weight']  # High priority for blocking critical paths
                break

        self.defensive_scores[cache_key] = score
        return score

    def select_node(self, node: RaveMCTSNode, board: Board) -> tuple:
        """
        Select most promising node to explore using UCB1 with RAVE.
        """
        with SafeBoardContext(board, self, "selection") as context:
            played_moves = []
            while node.untried_moves == [] and node.children:
                node = max(node.children, key=lambda n: n.ucb1(
                    explore_constant=self.weights['explore_constant'],
                    rave_constant=self.weights['rave_constant'],
                    depth_decay=self.weights['depth_decay']   
                ))
                move = node.move
                player = self.get_next_player(node.parent)
                context.apply_move(move, player)
                played_moves.append((move, player))
            return node, played_moves

    def is_valid_move(self, board: Board, move: tuple[int, int]) -> bool:
        """Enhanced move validation"""
        x, y = move
        if not (0 <= x < board.size and 0 <= y < board.size):
            return False
        return board.tiles[x][y].colour is None

    def expand(self, node: RaveMCTSNode, board: Board) -> RaveMCTSNode:
        """Improved expansion with move validation"""
        if node.untried_moves is None:
            node.untried_moves = [move for move in self.get_smart_moves(board)
                                if self.is_valid_move(board, move)]
        if node.untried_moves:
            move = node.untried_moves.pop(0)  # Take best move from sorted list
            next_player = self.get_next_player(node)
            board.set_tile_colour(move[0], move[1], next_player)
            child = RaveMCTSNode(parent=node, move=move)
            child.player = next_player
            node.children.append(child)
            return child
        return node

    def hash_board(self, board: Board) -> int:
        """Fast board hashing for transposition table"""
        hash_value = 0
        for i in range(board.size):
            for j in range(board.size):
                color = board.tiles[i][j].colour
                if color is not None:
                    hash_value ^= self._zobrist_table[(i, j, color)]
        return hash_value

    def simulate(self, board: Board) -> tuple:
        """
        Run a game simulation from current position to estimate position value.
        """
        board_hash = self.hash_board(board)
        if board_hash in self.transposition_table:
            return self.transposition_table[board_hash]

        with SafeBoardContext(board, self, "simulation") as context:
            moves_made = []
            current_player = self.colour  # Start with agent's color
            red_moves = []
            blue_moves = []

            while True:
                if not self.verify_board_state(board, "simulation"):
                    logger.error("Board corruption detected during simulation")
                    raise RuntimeError("Invalid board state")

                if board.has_ended(current_player):
                    result = (board._winner == self.colour)
                    break

                moves = [m for m in self.get_smart_moves(board)
                         if self.is_valid_move(board, m)]
                if not moves:
                    result = (board._winner == self.colour)
                    break

                move = moves[0] if random() < 0.8 else choice(moves)
                context.apply_move(move, current_player)
                self.get_player_moves(current_player, move, red_moves, blue_moves)
                moves_made.append((move, current_player))

                # Switch player
                current_player = self.switch_player(current_player)

            self.transposition_table[board_hash] = (result, red_moves, blue_moves)
            return result, red_moves, blue_moves

    def backpropagate(self, node: RaveMCTSNode, result: bool, moves_played: list):
        """
        Update statistics in all nodes from leaf to root.
        """
        while node is not None:
            node.visits += 1
            node.wins += self.win_score if result else 0

            # Update AMAF statistics for each move in the playout
            for move in moves_played:
                node.update_amaf_stats(move, result)

            node = node.parent

    def get_next_player(self, node: RaveMCTSNode) -> Colour:
        if node.player is None:
            return self.colour
        else:
            return Colour.RED if node.player == Colour.BLUE else Colour.BLUE
    
    def detect_bridges_under_attack(self, board: Board, opp_move: tuple[int, int]) -> list[tuple[int, int]]:
        """
        Detects bridges under attack by comparing two bridge positions before and after opponent's move.
        Returns a list of moves to save the threatened bridges.
        """
        x, y = opp_move[0], opp_move[1]
        moves_to_save = set()
        
        # Store original state and get current two bridges
        original_color = board.tiles[x][y].colour
        current_two_bridges = set()
        
        # Get all current two bridge positions
        for pos in [(i, j) for i in range(board.size) for j in range(board.size)]:
            if board.tiles[pos[0]][pos[1]].colour == self.colour:
                bridges = self.get_two_bridge_positions(board, pos, self.colour)
                for anchor, bridge_cells in bridges:
                    # Store as frozen set to make it hashable
                    current_two_bridges.add(frozenset(bridge_cells))
        
        # Temporarily undo opponent's move
        board.set_tile_colour(x, y, None)
        
        # Get all two bridge positions before opponent's move
        previous_two_bridges = set()
        for pos in [(i, j) for i in range(board.size) for j in range(board.size)]:
            if board.tiles[pos[0]][pos[1]].colour == self.colour:
                bridges = self.get_two_bridge_positions(board, pos, self.colour)
                for anchor, bridge_cells in bridges:
                    previous_two_bridges.add(frozenset(bridge_cells))
        
        # Restore board state
        board.set_tile_colour(x, y, original_color)
        
        # Find threatened bridges (bridges that existed before but not after opponent's move)
        threatened_bridges = previous_two_bridges - current_two_bridges
        
        # For each threatened bridge, add the empty cell that could save it
        for bridge_cells in threatened_bridges:
            for cell in bridge_cells:
                # If the cell isn't where opponent just played and it's empty
                if cell != opp_move and board.tiles[cell[0]][cell[1]].colour is None :
                    moves_to_save.add(cell)
        
        return list(set(moves_to_save))

    def check_immediate_win(self, board: Board, move: tuple[int, int], player: Colour) -> bool:
        """Check if a move leads to immediate win for specified player"""
        if not self.is_valid_move(board, move):
            return False
        board.set_tile_colour(move[0], move[1], player)
        won = board.has_ended(player)
        board.set_tile_colour(move[0], move[1], None)  # Undo the move
        return won

    def find_winning_sequence(self, board: Board, player: Colour, depth: int, path=None) -> list[tuple[int, int]] | None:
        """
        Look for a sequence of moves that leads to a guaranteed win.
        Returns the winning sequence if found, None otherwise.
        """
        if path is None:
            path = []
        
        if len(path) == depth:
            return None
            
        if board.has_ended(player):
            return path
            
        valid_moves = self.get_smart_moves(board)
        
        for move in valid_moves:
            if not self.is_valid_move(board, move):
                continue
                
            # Try move
            board.set_tile_colour(move[0], move[1], player)
            new_path = path + [move]
            
            # Check if this leads to a win
            if board.has_ended(player):
                board.set_tile_colour(move[0], move[1], None)
                return new_path
                
            # Recursively check opponent's responses
            opponent = Colour.opposite(player)
            can_opponent_prevent_win = False
            
            for opp_move in self.get_smart_moves(board):
                if not self.is_valid_move(board, opp_move):
                    continue
                    
                board.set_tile_colour(opp_move[0], opp_move[1], opponent)
                
                # Recursively search for our winning continuation
                sequence = self.find_winning_sequence(board, player, depth, new_path)
                
                board.set_tile_colour(opp_move[0], opp_move[1], None)
                
                if sequence is None:
                    can_opponent_prevent_win = True
                    break
            
            board.set_tile_colour(move[0], move[1], None)
            
            if not can_opponent_prevent_win:
                return new_path
        
        return None

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        """
        Main decision function combining all MCTS+RAVE elements.
        """
        # logger.info(f"Starting make_move for turn {turn}")
        initial_board_hash = self.hash_board(board)
        # logger.debug(f"Initial board hash: {initial_board_hash}")

        if not self.verify_board_state(board, "make_move start"):
            logger.error("Invalid initial board state")
            raise RuntimeError("Invalid initial board state")

        # Strategic Swap Evaluation
        if turn == 2 and opp_move is not None:
            opp_x, opp_y = opp_move.x, opp_move.y
            center = board.size // 2
            distance_to_center = abs(opp_x - center) + abs(opp_y - center)
            is_center = (opp_x, opp_y) == (center, center)

            # Evaluate the strength of the opponent's first move
            opp_move_strength = self.evaluate_move(board, (opp_x, opp_y))

            # Thresholds for swapping
            swap_distance_threshold = board.size // 4
            swap_strength_threshold = 0.7  # Adjust based on testing

            if is_center or distance_to_center <= swap_distance_threshold or opp_move_strength >= swap_strength_threshold:
                # Decide to swap
                return Move(-1, -1)

        # First check for immediate winning moves or blocking moves
        valid_moves = [m for m in self.get_valid_moves(board) if self.is_valid_move(board, m)]
        
        # Check our winning moves first
        for move in valid_moves:
            if self.check_immediate_win(board, move, self.colour):
                return Move(move[0], move[1])
        
        # Check opponent winning moves to block
        opponent = Colour.opposite(self.colour)
        for move in valid_moves:
            if self.check_immediate_win(board, move, opponent):
                return Move(move[0], move[1])

        # Count total moves played
        moves_played = sum(1 for i in range(board.size) 
                         for j in range(board.size) 
                         if board.tiles[i][j].colour is not None)
        
        if opp_move:
            bridges_to_save = self.detect_bridges_under_attack(board, (opp_move.x, opp_move.y))
            if bridges_to_save:
                move_to_save = choice(bridges_to_save)
                return Move(move_to_save[0], move_to_save[1])

        # Check for winning sequences in mid-late game
        if moves_played >= self.min_moves_for_sequence_check:
            # First check if we have a winning sequence
            our_sequence = self.find_winning_sequence(board, self.colour, self.win_sequence_depth)
            if our_sequence:
                return Move(our_sequence[0][0], our_sequence[0][1])

            # Then check if opponent has a winning sequence and block it
            opp_sequence = self.find_winning_sequence(board, Colour.opposite(self.colour), self.win_sequence_depth)
            if opp_sequence:
                return Move(opp_sequence[0][0], opp_sequence[0][1])

        # Continue with MCTS if no immediate wins/blocks found
        root_node = RaveMCTSNode()
        root_node.player = Colour.opposite(self.colour)

        try:
            for i in range(self.simulations):
                current_board_hash = self.hash_board(board)
                if current_board_hash != initial_board_hash:
                    logger.error(f"Board corruption detected in simulation {i}")
                    logger.error(f"Expected hash: {initial_board_hash}, Got: {current_board_hash}")
                    raise RuntimeError("Board state corrupted between simulations")

                node = root_node
                temp_board = self.copy_board(board)
                temp_board_hash = self.hash_board(temp_board)
                
                if temp_board_hash != initial_board_hash:
                    logger.error("Board copy corruption detected")
                    logger.error(f"Original hash: {initial_board_hash}, Copy hash: {temp_board_hash}")
                    raise RuntimeError("Board copy corruption")

                if not self.verify_board_state(temp_board, f"simulation {i}"):
                    logger.error(f"Board corruption detected in simulation {i}")
                    continue

                # Selection and expansion within SafeBoardContext
                with SafeBoardContext(temp_board, self, f"simulation {i}") as context:
                    # Selection
                    node, selection_moves = self.select_node(node, temp_board)

                    # Expansion
                    node = self.expand(node, temp_board)
                    if node.move:
                        context.apply_move(node.move, node.player)

                    # Simulation and backpropagation
                    outcome, red_moves, blue_moves = self.simulate(temp_board)
                    moves_for_backprop = red_moves if self.colour == Colour.RED else blue_moves
                    self.backpropagate(node, outcome, moves_for_backprop)

                    # No need to manually undo moves - SafeBoardContext handles it

                # Verify board state after simulation
                final_board_hash = self.hash_board(board)
                if final_board_hash != initial_board_hash:
                    logger.error(f"Board corruption after simulation {i}")
                    logger.error(f"Initial hash: {initial_board_hash}, Final hash: {final_board_hash}")
                    raise RuntimeError("Board state corrupted after simulation")

        except Exception as e:
            logger.error(f"Error during MCTS: {str(e)}")
            logger.error(f"Final board hash: {self.hash_board(board)}")
            # Fall back to safe move selection
            valid_moves = self.get_valid_moves(board)
            if valid_moves:
                return Move(valid_moves[0][0], valid_moves[0][1])

        # Add final verification
        final_state_hash = self.hash_board(board)
        if final_state_hash != initial_board_hash:
            logger.error("Board corruption detected at end of make_move")
            logger.error(f"Initial hash: {initial_board_hash}, Final hash: {final_state_hash}")
            raise RuntimeError("Final board state corrupted")

        # logger.info("Completed make_move successfully")

        # Select the best move
        if root_node.children:
            best_child = max(root_node.children, key=lambda c: c.visits)
            best_move = best_child.move
            if self.is_valid_move(board, best_move):
                self.root = best_child
                self.root.parent = None
                return Move(best_move[0], best_move[1])
        
        print("No valid moves found, returning random")

        # Safe fallback
        valid_moves = [m for m in self.get_valid_moves(board) if self.is_valid_move(board, m)]
        if valid_moves:
            move = choice(valid_moves)
            self.root = RaveMCTSNode()
            return Move(move[0], move[1])

        return Move(-1, -1)  # Safe fallback if no valid moves found

    def _trace_path(self, board: Board, start: tuple[int, int], player: Colour, direction: str) -> list[tuple[int, int]]:
        """
        Trace a potential winning path from start position using A*.
        Returns shortest path if one exists, otherwise empty list.
        
        Args:
            board: Current game board
            start: Starting position (x,y)
            player: Player color to trace for
            direction: 'vertical' for RED (top-bottom), 'horizontal' for BLUE (left-right)
        """
        open_list = [] # Discovered, but not visited, nodes
        # heapq is a min-heap priority queue (smallest value first, aka closest distance to target)
        # This is used for the open_list
        heapq.heappush(open_list, (0, start, [start]))

        visited = set() # Visited nodes
        visited.add(start)

        target = board.size - 1
            
        while open_list:
            f, (x, y), path = heapq.heappop(open_list)

            # Check winning condition
            if (direction == 'vertical' and x == target) or \
               (direction == 'horizontal' and y == target):
                return path
            
            # Check all possible directions
            for dx, dy in [(0,1), (1,0), (1,-1), (0,-1), (-1,0), (-1,1)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < board.size) and (0 <= ny < board.size) and (nx, ny) not in visited: 
                    if board.tiles[nx][ny].colour == player or board.tiles[nx][ny].colour is None:
                        visited.add((nx, ny))
                        g = len(path) # g(n) is the cost to reach this node, aka the path length
                        h = self.distance_to_target(nx, ny, target, direction) # h(n) is the estimated cost to target
                        f = g + h # f(n) = g(n) + h(n)
                        heapq.heappush(open_list, (f, (nx, ny), path + [(nx, ny)]))

        # No path found
        return []


    def distance_to_target(self, x: int, y: int, target: int, direction: str) -> int:
        """ This is used in _trace_path() to calculate the estimated cost to the target heuristic. """
        if direction == 'vertical':
            return target - x
        else:
            return target - y

    def verify_board_state(self, board: Board, context: str) -> bool:
        """Verify board state integrity"""
        valid_moves = self.get_valid_moves(board)
        occupied_spaces = [(i, j) for i in range(board.size) 
                          for j in range(board.size) 
                          if board.tiles[i][j].colour is not None]
        
        # Check for invalid color assignments
        for x, y in occupied_spaces:
            if board.tiles[x][y].colour not in [Colour.RED, Colour.BLUE]:
                logger.error(f"{context}: Invalid color at ({x}, {y})")
                return False
        
        # Check for overlapping moves
        if len(set(occupied_spaces)) != len(occupied_spaces):
            logger.error(f"{context}: Duplicate moves detected")
            return False
            
        return True