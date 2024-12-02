from math import sqrt, log
from random import choice, random, Random
from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move
import heapq
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

    def get_amaf_value(self, move: tuple[int, int]) -> float:
        """
        Calculate the AMAF value for a specific move using cached statistics.
        """
        visits = self.move_amaf_visits.get(move, 0)
        if visits == 0:
            return 0.0
        return self.move_amaf_wins.get(move, 0) / visits

    def ucb1(self, explore_constant, rave_constant) -> float:
        """
        Calculate node selection value using provided constants.
        """
        if self.visits == 0:
            return float('inf')

        beta = sqrt(rave_constant / (3 * self.visits + rave_constant))
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
            'center_weight': 1.419,
            'neighbor_weight': 0.729,
            'bridge_weight': 0.568,
            'edge_weight': 0.568,
            'defensive_weight': 2.225,
            'two_bridge_weight': 4.452,
            'opponent_bridge_block': 3.226,
            'explore_constant': 1.568,
            'rave_constant': 322.542,
            'early_stop_threshold': 0.95,
            'min_visits_ratio': 0.142,
            'progressive_widening_constant': 1.5,
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


    def get_two_bridges(self, board: Board, move: tuple[int, int], colour: Colour):
        """Returns the number of two bridges made by a given move for a given player."""
        x, y = move
        
        # Define all possible two bridge patterns relative to move position
        patterns = [
            # Format: [(piece_x, piece_y), (empty1_x, empty1_y), (empty2_x, empty2_y)]
            [(-1, -1), (-1, 0), (0, -1)],
            [(1, -2), (0, -1), (1, -1)],
            [(2, -1), (1, -1), (1, 0)],
            [(1, 1), (1, 0), (0, 1)],
            [(-1, 2), (0, 1), (-1, 1)],
            [(-2, 1), (-1, 1), (-1, 0)]
        ]
        
        count = 0
        # Cache board size check
        size = board.size
        
        # Pre-compute function results for the move position
        def in_bounds(px, py):
            return 0 <= px < size and 0 <= py < size
        
        for pattern in patterns:
            piece, empty1, empty2 = pattern
            px, py = x + piece[0], y + piece[1]
            e1x, e1y = x + empty1[0], y + empty1[1]
            e2x, e2y = x + empty2[0], y + empty2[1]
            
            # Single bounds check for all points
            if not (in_bounds(px, py) and in_bounds(e1x, e1y) and in_bounds(e2x, e2y)):
                continue
                
            # Check pattern match
            if (board.tiles[px][py].colour is colour and
                board.tiles[e1x][e1y].colour is None and
                board.tiles[e2x][e2y].colour is None):
                count += 1
                
        return count

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
        played_moves = []
        while node.untried_moves == [] and node.children:
            node = max(node.children, key=lambda n: n.ucb1(
                explore_constant=self.weights['explore_constant'],
                rave_constant=self.weights['rave_constant']
            ))
            move = node.move
            player = self.get_next_player(node.parent)
            board.set_tile_colour(move[0], move[1], player)
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
        moves_made = []
        current_player = self.colour  # Start with agent's color
        red_moves = []
        blue_moves = []

        while True:
            if board.has_ended(current_player):
                result = (board._winner == self.colour)
                break

            moves = [m for m in self.get_smart_moves(board)
                     if self.is_valid_move(board, m)]
            if not moves:
                result = (board._winner == self.colour)
                break

            move = moves[0] if random() < 0.8 else choice(moves)
            board.set_tile_colour(move[0], move[1], current_player)
            self.get_player_moves(current_player, move, red_moves, blue_moves)
            moves_made.append((move, current_player))

            # Switch player
            current_player = self.switch_player(current_player)

        # Undo moves
        for move, player in reversed(moves_made):
            board.set_tile_colour(move[0], move[1], None)

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

    def check_immediate_win(self, board: Board, move: tuple[int, int], player: Colour) -> bool:
        """Check if a move leads to immediate win for specified player"""
        if not self.is_valid_move(board, move):
            return False
        board.set_tile_colour(move[0], move[1], player)
        won = board.has_ended(player)
        board.set_tile_colour(move[0], move[1], None)  # Undo the move
        if won:
            print(f"Immediate win found for {player} at {move}")
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

        for i in range(self.simulations):
            node = root_node
            temp_board = board

            # Selection
            played_moves = []
            node, selection_moves = self.select_node(node, temp_board)
            played_moves.extend(selection_moves)

            # Expansion
            node = self.expand(node, temp_board)
            if node.move:
                temp_board.set_tile_colour(node.move[0], node.move[1], node.player)
                played_moves.append((node.move, node.player))

            # Simulation and backpropagation with correct move lists
            outcome, red_moves, blue_moves = self.simulate(temp_board)
            moves_for_backprop = red_moves if self.colour == Colour.RED else blue_moves
            self.backpropagate(node, outcome, moves_for_backprop)

            # Undo moves
            for move, player in reversed(played_moves):
                temp_board.set_tile_colour(move[0], move[1], None)

            # Early stopping criteria
            if i > 100:
                best_child = max(root_node.children, key=lambda c: c.visits)
                visit_ratio = best_child.visits / (i + 1)
                if visit_ratio > self.min_visits_ratio and best_child.wins / best_child.visits > self.early_stop_threshold:
                    break

        # Select the best move
        if root_node.children:
            best_child = max(root_node.children, key=lambda c: c.visits)
            best_move = best_child.move
            if self.is_valid_move(board, best_move):
                self.root = best_child
                self.root.parent = None
                return Move(best_move[0], best_move[1])

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