from math import sqrt, log
from random import choice, Random
from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move
import heapq

class RaveNode:
    def __init__(self, parent=None, move=None):
        self.parent = parent
        self.move = move  # The move that led to this node (x,y)
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = None
        self.player = None
        # RAVE/AMAF statistics
        self.move_amaf_wins = {}   # Track AMAF wins per move
        self.move_amaf_visits = {} # Track AMAF visits per move
        self.hash = None  # Add hash field for transposition table
        self.pruned_moves = set()  # Store inferior moves
        self.virtual_connections = set()  # Store virtual connections
        self.heavy_node_processed = False  # Track if node has been analyzed
        self.visit_threshold = 500  # Threshold for heavy node processing

    def get_amaf_value(self, move):
        """Get the AMAF value for a move"""
        visits = self.move_amaf_visits.get(move, 0)
        if visits == 0:
            return 0.0
        return self.move_amaf_wins.get(move, 0) / visits

    def ucb1_rave(self, explore_constant=1.4, rave_constant=300):
        """UCB1 formula with RAVE heuristic"""
        if self.visits == 0:
            return float('inf')

        # Calculate beta - determines how much we trust RAVE values
        beta = sqrt(rave_constant / (3 * self.visits + rave_constant))
        
        # Standard UCB1
        exploitation = self.wins / self.visits
        exploration = explore_constant * sqrt(log(self.parent.visits) / self.visits)
        ucb = exploitation + exploration

        # RAVE value
        amaf_value = self.get_amaf_value(self.move) if self.move else 0.0

        # Combine standard UCB1 with RAVE value
        return (1 - beta) * ucb + beta * amaf_value

    def update_amaf_stats(self, move, won):
        """Update AMAF statistics for a move"""
        if move not in self.move_amaf_visits:
            self.move_amaf_visits[move] = 0
            self.move_amaf_wins[move] = 0
        self.move_amaf_visits[move] += 1
        if won:
            self.move_amaf_wins[move] += 1

    def is_heavy_node(self):
        """Check if node has enough visits to be considered heavy"""
        return self.visits >= self.visit_threshold and not self.heavy_node_processed

    def find_virtual_connections(self, board: Board):
        """Find virtual connections at current position"""
        connections = set()
        # Check for each empty cell if it forms a virtual connection
        for i in range(board.size):
            for j in range(board.size):
                if board.tiles[i][j].colour is None:
                    if self._check_virtual_connection(board, (i,j)):
                        connections.add((i,j))
        return connections

    def _check_virtual_connection(self, board: Board, move: tuple[int, int]) -> bool:
        """Check if a move forms a virtual connection"""
        # Make temporary move
        board.set_tile_colour(move[0], move[1], self.player)
        
        # Check if this creates a strong connection
        has_connection = self._has_strong_connection(board, move)
        
        # Undo move
        board.set_tile_colour(move[0], move[1], None)
        
        return has_connection

    def _has_strong_connection(self, board: Board, move: tuple[int, int]) -> bool:
        """Check if position has a strong connection to winning condition"""
        # Implement connection detection logic here
        # This is a simplified check - you should implement more sophisticated detection
        x, y = move
        
        if self.player == Colour.RED:
            # Check for strong vertical connection
            return self._has_vertical_connection(board, x, y)
        else:
            # Check for strong horizontal connection
            return self._has_horizontal_connection(board, x, y)

    def _has_vertical_connection(self, board: Board, x: int, y: int) -> bool:
        """Check for strong vertical connections (RED player)"""
        # Check potential connections to both top and bottom
        top_connection = False
        bottom_connection = False
        
        # Look for connections to top edge
        for j in range(board.size):
            if board.tiles[0][j].colour == Colour.RED:
                if self._exists_connection_path(board, (0, j), (x, y), Colour.RED):
                    top_connection = True
                    break
                    
        # Look for connections to bottom edge
        for j in range(board.size):
            if board.tiles[board.size-1][j].colour == Colour.RED:
                if self._exists_connection_path(board, (board.size-1, j), (x, y), Colour.RED):
                    bottom_connection = True
                    break
                    
        return top_connection and bottom_connection

    def _has_horizontal_connection(self, board: Board, x: int, y: int) -> bool:
        """Check for strong horizontal connections (BLUE player)"""
        # Check potential connections to both left and right
        left_connection = False
        right_connection = False
        
        # Look for connections to left edge
        for i in range(board.size):
            if board.tiles[i][0].colour == Colour.BLUE:
                if self._exists_connection_path(board, (i, 0), (x, y), Colour.BLUE):
                    left_connection = True
                    break
                    
        # Look for connections to right edge
        for i in range(board.size):
            if board.tiles[i][board.size-1].colour == Colour.BLUE:
                if self._exists_connection_path(board, (i, board.size-1), (x, y), Colour.BLUE):
                    right_connection = True
                    break
                    
        return left_connection and right_connection

    def _exists_connection_path(self, board: Board, start: tuple[int, int], 
                              end: tuple[int, int], colour: Colour,
                              visited: set = None) -> bool:
        """
        Check if there exists a path between two points using A* pathfinding.
        Considers both direct connections and virtual connections through bridges.
        """
        if visited is None:
            visited = set()
            
        # Initialize priority queue with start position
        # Format: (f_score, current_pos, path)
        open_list = [(self._manhattan_distance(start, end), start, [start])]
        visited.add(start)
        
        while open_list:
            _, current, path = heapq.heappop(open_list)
            
            if current == end:
                return True
                
            # Check all possible moves including bridge connections
            for next_pos in self._get_connection_moves(board, current, colour):
                if next_pos not in visited:
                    visited.add(next_pos)
                    new_f_score = len(path) + self._manhattan_distance(next_pos, end)
                    heapq.heappush(open_list, (new_f_score, next_pos, path + [next_pos]))
        
        return False

    def _get_connection_moves(self, board: Board, pos: tuple[int, int], 
                            colour: Colour) -> list[tuple[int, int]]:
        """Get all possible moves including direct and bridge connections"""
        x, y = pos
        moves = []
        
        # Direct adjacent moves
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,1), (1,-1)]:
            nx, ny = x + dx, y + dy
            if self._is_valid_position(board, nx, ny):
                if board.tiles[nx][ny].colour == colour:
                    moves.append((nx, ny))
                    
        # Bridge connections
        bridge_patterns = [
            [(-1,-1), (-1,0), (0,-1)],  # top-left bridge
            [(1,-2), (0,-1), (1,-1)],   # top-right bridge
            [(2,-1), (1,-1), (1,0)],    # right bridge
            [(1,1), (1,0), (0,1)],      # bottom-right bridge
            [(-1,2), (0,1), (-1,1)],    # bottom-left bridge
            [(-2,1), (-1,1), (-1,0)]    # left bridge
        ]
        
        for pattern in bridge_patterns:
            endpoint, space1, space2 = pattern
            ex, ey = x + endpoint[0], y + endpoint[1]
            s1x, s1y = x + space1[0], y + space1[1]
            s2x, s2y = x + space2[0], y + space2[1]
            
            if (self._is_valid_position(board, ex, ey) and 
                self._is_valid_position(board, s1x, s1y) and 
                self._is_valid_position(board, s2x, s2y)):
                if (board.tiles[ex][ey].colour == colour and
                    board.tiles[s1x][s1y].colour is None and
                    board.tiles[s2x][s2y].colour is None):
                    moves.append((ex, ey))
        
        return moves

    def _is_valid_position(self, board: Board, x: int, y: int) -> bool:
        """Check if position is within board boundaries"""
        return 0 <= x < board.size and 0 <= y < board.size

    def _manhattan_distance(self, pos1: tuple[int, int], pos2: tuple[int, int]) -> int:
        """Calculate Manhattan distance between two positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def find_inferior_cells(self, board: Board) -> set:
        """Find inferior moves that can be pruned"""
        inferior = set()
        
        # Check each empty cell
        for i in range(board.size):
            for j in range(board.size):
                if board.tiles[i][j].colour is None:
                    if self._is_inferior_move(board, (i,j)):
                        inferior.add((i,j))
        
        return inferior

    def _is_inferior_move(self, board: Board, move: tuple[int, int]) -> bool:
        """Check if a move is inferior and can be pruned"""
        # Make the move
        board.set_tile_colour(move[0], move[1], self.player)
        
        # Check if the position is dominated by another move
        is_inferior = self._is_dominated_position(board, move)
        
        # Undo the move
        board.set_tile_colour(move[0], move[1], None)
        
        return is_inferior

    def _is_dominated_position(self, board: Board, move: tuple[int, int]) -> bool:
        """Check if position is dominated by another move"""
        # Implement domination check logic here
        # For example, check if another move achieves strictly better connectivity
        x, y = move
        
        # Check neighboring cells for better moves
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,1), (1,-1)]:
            nx, ny = x + dx, y + dy
            if (0 <= nx < board.size and 0 <= ny < board.size and 
                board.tiles[nx][ny].colour is None):
                if self._dominates(board, (nx,ny), move):
                    return True
        return False

    def _dominates(self, board: Board, move1: tuple[int, int], move2: tuple[int, int]) -> bool:
        """Check if move1 dominates move2 strategically"""
        x1, y1 = move1
        x2, y2 = move2
        
        # Make move1 and check its strategic value
        board.set_tile_colour(x1, y1, self.player)
        value1 = self._evaluate_position(board, move1)
        board.set_tile_colour(x1, y1, None)
        
        # Make move2 and check its strategic value
        board.set_tile_colour(x2, y2, self.player)
        value2 = self._evaluate_position(board, move2)
        board.set_tile_colour(x2, y2, None)
        
        return value1 > value2

    def _evaluate_position(self, board: Board, move: tuple[int, int]) -> float:
        """Evaluate the strategic value of a position"""
        x, y = move
        value = 0.0
        
        # Check connectivity to edges
        if self.player == Colour.RED:
            # For red player, check vertical connections
            if x == 0:  # Connected to top
                value += 1.0
            if x == board.size - 1:  # Connected to bottom
                value += 1.0
            # Bonus for center positions
            value += 1.0 - abs(y - board.size/2) / board.size
        else:
            # For blue player, check horizontal connections
            if y == 0:  # Connected to left
                value += 1.0
            if y == board.size - 1:  # Connected to right
                value += 1.0
            # Bonus for center positions
            value += 1.0 - abs(x - board.size/2) / board.size
        
        # Check neighbor connections
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,1), (1,-1)]:
            nx, ny = x + dx, y + dy
            if self._is_valid_position(board, nx, ny):
                if board.tiles[nx][ny].colour == self.player:
                    value += 0.5
                
        # Check for bridge potential
        bridge_score = self._count_potential_bridges(board, move)
        value += bridge_score * 0.3
        
        return value

    def _count_potential_bridges(self, board: Board, move: tuple[int, int]) -> int:
        """Count number of potential bridges from this position"""
        x, y = move
        bridge_count = 0
        
        # Bridge patterns to check
        bridge_patterns = [
            [(-1,-1), (-1,0), (0,-1)],  # top-left bridge
            [(1,-2), (0,-1), (1,-1)],   # top-right bridge
            [(2,-1), (1,-1), (1,0)],    # right bridge
            [(1,1), (1,0), (0,1)],      # bottom-right bridge
            [(-1,2), (0,1), (-1,1)],    # bottom-left bridge
            [(-2,1), (-1,1), (-1,0)]    # left bridge
        ]
        
        for pattern in bridge_patterns:
            endpoint, space1, space2 = pattern
            ex, ey = x + endpoint[0], y + endpoint[1]
            s1x, s1y = x + space1[0], y + space1[1]
            s2x, s2y = x + space2[0], y + space2[1]
            
            if (self._is_valid_position(board, ex, ey) and 
                self._is_valid_position(board, s1x, s1y) and 
                self._is_valid_position(board, s2x, s2y)):
                # Check if bridge formation is possible
                if (board.tiles[s1x][s1y].colour is None and
                    board.tiles[s2x][s2y].colour is None):
                    bridge_count += 1
        
        return bridge_count

class RaveBasicAgent(AgentBase):
    def __init__(self, colour: Colour):
        super().__init__(colour)
        self.simulations = 1000
        self.explore_constant = 1.4
        self.rave_constant = 300
        self._rng = Random(42)  # Create a dedicated random number generator
        self.transposition_table = {}  # Cache for board states
        self._zobrist_table = {
            (i, j, color): self._rng.getrandbits(64)
            for i in range(11)  # Using standard board size 11
            for j in range(11)
            for color in [Colour.RED, Colour.BLUE]
        }
        self.pruning_cache = {}  # Cache for pruning results

    def get_valid_moves(self, board: Board):
        """Get all valid moves on the board"""
        return [(i, j) for i in range(board.size) 
                for j in range(board.size)
                if board.tiles[i][j].colour is None]
    

    def select_node(self, node, board):
        """Enhanced selection considering pruned moves"""
        played_moves = []
        if node.untried_moves is None:
            node.untried_moves = [m for m in self.get_valid_moves(board) 
                                if m not in node.pruned_moves]
        
        
        while node.untried_moves or node.children:
            # Check if node is heavy and needs processing
            if node.is_heavy_node():
                self._process_heavy_node(node, board)
            
            # Filter out pruned moves
            valid_children = [c for c in node.children if c.move not in node.pruned_moves]
            
            if valid_children:
                node = max(valid_children, 
                          key=lambda n: n.ucb1_rave(self.explore_constant, self.rave_constant))
                if node.move:
                    board.set_tile_colour(node.move[0], node.move[1], 
                                        Colour.opposite(node.parent.player or self.colour))
                    played_moves.append((node.move, node.player))
            else:
                break
                
        return node, played_moves

    def _process_heavy_node(self, node: RaveNode, board: Board):
        """Process a heavy node to find virtual connections and inferior moves"""
        # Check cache first
        board_hash = self.hash_board(board)
        if board_hash in self.pruning_cache:
            node.virtual_connections = self.pruning_cache[board_hash]['vc']
            node.pruned_moves = self.pruning_cache[board_hash]['pruned']
            node.heavy_node_processed = True
            return

        # Find virtual connections
        node.virtual_connections = node.find_virtual_connections(board)
        
        # Find inferior moves
        node.pruned_moves = node.find_inferior_cells(board)
        
        # Cache results
        self.pruning_cache[board_hash] = {
            'vc': node.virtual_connections,
            'pruned': node.pruned_moves
        }
        
        node.heavy_node_processed = True

    def expand(self, node, board):
        """Modified expansion avoiding pruned moves"""
        if node.untried_moves is None:
            # Initialize with non-pruned moves
            node.untried_moves = [m for m in self.get_valid_moves(board) 
                                if m not in node.pruned_moves]
        
        if node.untried_moves:
            move = choice(node.untried_moves)
            node.untried_moves.remove(move)
            
            next_player = Colour.opposite(node.player) if node.player else self.colour
            board.set_tile_colour(move[0], move[1], next_player)
            
            child = RaveNode(parent=node, move=move)
            child.player = next_player
            node.children.append(child)
            return child
        return node

    def _detect_bridge_threats(self, board: Board, move: tuple[int, int], player: Colour) -> list[tuple]:
        """Detect bridges that are threatened by a move"""
        x, y = move
        threatened_bridges = []
        
        # Bridge patterns relative to the threatening move
        bridge_patterns = [
            [(-1,-1), (-1,0), (0,-1)],  # top-left bridge
            [(1,-2), (0,-1), (1,-1)],   # top-right bridge
            [(2,-1), (1,-1), (1,0)],    # right bridge
            [(1,1), (1,0), (0,1)],      # bottom-right bridge
            [(-1,2), (0,1), (-1,1)],    # bottom-left bridge
            [(-2,1), (-1,1), (-1,0)]    # left bridge
        ]
        
        for pattern in bridge_patterns:
            endpoint, space1, space2 = pattern
            ex, ey = x + endpoint[0], y + endpoint[1]
            s1x, s1y = x + space1[0], y + space1[1]
            s2x, s2y = x + space2[0], y + space2[1]
            
            # Check if the pattern forms a bridge
            if (0 <= ex < board.size and 0 <= ey < board.size and
                0 <= s1x < board.size and 0 <= s1y < board.size and
                0 <= s2x < board.size and 0 <= s2y < board.size):
                
                if (board.tiles[ex][ey].colour == player and
                    board.tiles[s1x][s1y].colour is None and
                    board.tiles[s2x][s2y].colour is None):
                    # Found a bridge that could be threatened
                    if (x, y) in [(s1x,s1y), (s2x,s2y)]:
                        threatened_bridges.append((ex,ey,s1x,s1y,s2x,s2y))
                        
        return threatened_bridges

    def _get_bridge_save_move(self, board: Board, bridge: tuple, threat: tuple) -> tuple[int, int]:
        """Get the move that saves a threatened bridge"""
        ex,ey,s1x,s1y,s2x,s2y = bridge
        threat_x, threat_y = threat
        
        # Return the other empty space that wasn't threatened
        if (threat_x,threat_y) == (s1x,s1y):
            return (s2x,s2y)
        return (s1x,s1y)

    def simulate(self, board, start_player):
        """Random playout simulation with bridge saving"""
        current_player = start_player
        moves_played = []

        while not board.has_ended(current_player):
            valid_moves = self.get_valid_moves(board)
            if not valid_moves:
                break
            
            # Make random move    
            move = choice(valid_moves)
            board.set_tile_colour(move[0], move[1], current_player)
            moves_played.append((move, current_player))
            
            # Check if this move threatens any opponent bridges
            next_player = Colour.opposite(current_player)
            threatened_bridges = self._detect_bridge_threats(board, move, next_player)
            
            if threatened_bridges:
                # Randomly choose one bridge to save if multiple are threatened
                bridge_to_save = choice(threatened_bridges)
                save_move = self._get_bridge_save_move(board, bridge_to_save, move)
                
                # Make the saving move if it's still available
                if (save_move[0], save_move[1]) in self.get_valid_moves(board):
                    board.set_tile_colour(save_move[0], save_move[1], next_player)
                    moves_played.append((save_move, next_player))
                    current_player = Colour.opposite(next_player)
                    continue
            
            current_player = next_player

        won = board.has_ended(self.colour) and board._winner == self.colour

        # Undo moves
        for move, _ in reversed(moves_played):
            board.set_tile_colour(move[0], move[1], None)

        return won, moves_played

    def backpropagate(self, node, won, moves_played):
        """Update node statistics including AMAF values"""
        while node:
            node.visits += 1
            node.wins += 1 if won else 0
            
            # Update AMAF statistics for each move
            for move, player in moves_played:
                # Only update AMAF stats for moves made by the same player as this node
                if player == node.player:
                    node.update_amaf_stats(move, won)
            
            node = node.parent

    def hash_board(self, board: Board) -> int:
        """Calculate Zobrist hash for current board state"""
        hash_value = 0
        for i in range(board.size):
            for j in range(board.size):
                color = board.tiles[i][j].colour
                if color is not None:
                    hash_value ^= self._zobrist_table[(i, j, color)]
        return hash_value

    def get_cached_node(self, board: Board) -> RaveNode | None:
        """Retrieve a cached node from transposition table if it exists"""
        board_hash = self.hash_board(board)
        return self.transposition_table.get(board_hash)

    def cache_node(self, board: Board, node: RaveNode):
        """Cache a node in the transposition table"""
        board_hash = self.hash_board(board)
        node.hash = board_hash
        self.transposition_table[board_hash] = node

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        """Main MCTS+RAVE implementation with transposition table"""
        # Handle swap rule cases
        if turn == 2 and opp_move and opp_move.x == -1 and opp_move.y == -1:
            # Opponent swapped - take center
            return Move(board.size // 2, board.size // 2)
            
        # Check transposition table for existing analysis
        cached_node = self.get_cached_node(board)
        if cached_node:
            root = cached_node
        else:
            root = RaveNode()
            root.player = Colour.opposite(self.colour)
            self.cache_node(board, root)

        for _ in range(self.simulations):
            node = root
            board_copy = Board(board.size)
            for i in range(board.size):
                for j in range(board.size):
                    board_copy.tiles[i][j].colour = board.tiles[i][j].colour

            # Selection
            node, moves_made = self.select_node(node, board_copy)

            # Cache the new state before expansion
            board_hash = self.hash_board(board_copy)
            cached_state = self.transposition_table.get(board_hash)
            if cached_state:
                node = cached_state
            else:
                # Expansion
                if not board_copy.has_ended(node.player or self.colour):
                    node = self.expand(node, board_copy)
                    self.cache_node(board_copy, node)

            # Simulation
            won, sim_moves = self.simulate(board_copy, 
                                         Colour.opposite(node.player) if node.player else self.colour)

            # Backpropagation
            self.backpropagate(node, won, moves_made + sim_moves)

            # Undo moves made during selection/expansion
            for move, _ in reversed(moves_made):
                board_copy.set_tile_colour(move[0], move[1], None)

        # Select best move
        if root.children:
            best_child = max(root.children, key=lambda c: c.visits)
            return Move(best_child.move[0], best_child.move[1])

        # Fallback to random move if no children (shouldn't happen)
        valid_moves = self.get_valid_moves(board)
        if valid_moves:
            move = choice(valid_moves)
            return Move(move[0], move[1])
        return Move(-1, -1)