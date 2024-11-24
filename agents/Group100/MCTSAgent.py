import heapq
from math import sqrt, log
from random import choice, random, Random
from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move
from copy import deepcopy


class RaveMCTSNode:
    """
    Enhanced MCTS node implementing RAVE (Rapid Action Value Estimation).
    
    MCTS Background:
    - Monte Carlo Tree Search builds a game tree gradually
    - Each node represents a game state
    - We track visits and wins to estimate move quality
    
    RAVE Enhancement:
    - Traditional MCTS only learns from moves in their exact position
    - RAVE assumes a good move in one position might be good in similar positions
    - Uses AMAF (All Moves As First) heuristic to share knowledge between nodes
    - Provides better estimates early in search when data is limited
    
    Storage Optimisations:
    - move_amaf_wins/visits: Track move statistics independently
    - hash_value: Quick board position lookups
    - board: Cached board state to avoid recomputation
    """
    def __init__(self, board: Board = None, parent=None, move=None):
        self.parent = parent
        self.move = move  # The move that led to this node
        self.children = []
        self.wins = 0  # Total wins from simulations passing through this node
        self.visits = 0  # Total simulations passing through this node
        self.untried_moves = None  # Moves that have not been tried from this node
        self.player = None  # The player who made the move to reach this node
        self.Q_RAVE = 0  # Total RAVE wins
        self.N_RAVE = 0  # Total RAVE visits
        self.board = board  # Need to store board state
        self.hash_value = None  # Add board hash for transposition table
        self.move_amaf_wins = {}  # Track AMAF wins per move
        self.move_amaf_visits = {}  # Track AMAF visits per move


    def get_amaf_value(self, move: tuple[int, int]) -> float:
        """
        Calculate the AMAF value for a specific move using cached statistics.
        
        Optimisation: Uses move-specific tracking instead of global RAVE values,
        which provides more accurate action value estimates and reduces noise
        from unrelated moves.
        """
        visits = self.move_amaf_visits.get(move, 0)
        if visits == 0:
            return 0.0
        return self.move_amaf_wins.get(move, 0) / visits


    def ucb1(self, explore: float = 1.41, rave_const: float = 300) -> float:
        """
        Calculate node selection value combining MCTS and RAVE scores.
        
        UCB1 Formula Components:
        1. Exploitation Term (wins/visits):
           - Higher ratio means move has been more successful
           - Encourages selecting proven good moves
        
        2. Exploration Term (sqrt(log(parent_visits)/visits)):
           - Ensures less-visited moves get tried
           - Prevents getting stuck in local optima
        
        3. RAVE Modification:
        - Beta: Balance between MCTS and RAVE scores
        - Decreases as visits increase (transitions from RAVE to MCTS)
        - RAVE values help early when MCTS data is sparse
        
        Parameters:
        - explore: Controls exploration vs exploitation (default 1.41)
        - rave_const: Controls RAVE influence decay rate (default 300)
        """
        if self.visits == 0:
            return float('inf')
        
        beta = sqrt(rave_const / (3 * self.visits + rave_const))
        mcts_value = self.wins / self.visits + explore * sqrt(log(self.parent.visits) / self.visits)
        
        # Use move-specific AMAF value
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


class MCTSAgent(AgentBase):
    """
    RAVE-enhanced MCTS agent with specialised optimisations for Hex gameplay.
    
    Key Optimisations:
    - Move reversal instead of board copying in simulation
    - Strategic move ordering using evaluation cache
    - Early stopping based on visit statistics
    - Efficient AMAF tracking with move-specific statistics
    - Zobrist hashing for board state caching
    """
    def __init__(self, colour: Colour):
        super().__init__(colour)
        self.simulations = 1000
        self.win_score = 10
        self.colour = colour
        self.root = RaveMCTSNode()
        self._board_size = 11
        self._all_positions = [(i, j) for i in range(self._board_size) 
                               for j in range(self._board_size)]
        self.rave_constant = 300  # Tune this value
        self.move_history = []  # Track move history
        self.move_scores = {}  # Cache for move evaluations
        self.transposition_table = {}  # Cache for board states
        self.early_stop_threshold = 0.95  # Higher threshold for early stopping
        self.min_visits_ratio = 0.1  # Minimum visit ratio for early stopping
        self._rng = Random(42)  # Create a dedicated random number generator
        self._zobrist_table = {
            (i, j, color): self._rng.getrandbits(64)
            for i in range(self._board_size)
            for j in range(self._board_size)
            for color in [Colour.RED, Colour.BLUE]
        }
        self.last_board = None  # Track last board state
        self.current_board_state = None  # Track current board state
        self.opposing_colour = Colour.opposite(self.colour)
        self.current_player = self.colour  # Add current player tracking
        self.critical_paths = {}  # Cache for critical path analysis
        self.opponent_bridges = {} # Cache for opponent bridge positions
        self.defensive_scores = {} # Cache for defensive position scores


    def switch_player(self, current_player: Colour) -> Colour:
        """
        Optimise player switching using direct comparison instead of multiple checks.
        Reduces branching and improves cache hit rate in the simulation phase.
        """
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
        return [(i, j) for i in range(board.size)
                for j in range(board.size)
                if board.tiles[i][j].colour is None]


    def get_neighbor_moves(self, board: Board, x: int, y: int) -> list[tuple[int, int]]:
        """Get valid moves adjacent to existing pieces"""
        neighbors = []
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                nx, ny = x + i, y + j
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
        score += friendly_neighbors * 0.5
        
        # Bridge formation potential
        bridge_score = sum(1 for nx, ny in neighbors 
                         if abs(nx-x) == 1 and abs(ny-y) == 1 
                         and board.tiles[nx][ny].colour == self.colour)
        score += bridge_score * 0.3

        # Two bridge forming potential
        # If it forms a two bridge, this is a good move
        two_bridge_score = self.get_two_bridges_score(board, move)
        print("Two bridges score:", two_bridge_score)
        score += two_bridge_score * 0.4
        
        # Edge control
        if self.colour == Colour.RED and (x == 0 or x == board.size-1):
            score += 0.4
        elif self.colour == Colour.BLUE and (y == 0 or y == board.size-1):
            score += 0.4
            
        # Add defensive score
        defensive_score = self.evaluate_defensive_position(board, move)
        score += defensive_score

        # Check for fork prevention
        test_board = deepcopy(board)
        test_board.set_tile_colour(x, y, self.opposing_colour)
        if len(self.find_critical_paths(test_board, self.opposing_colour)) > \
           len(self.find_critical_paths(board, self.opposing_colour)):
            score += 1.0  # Priority for preventing opponent forks

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
        
        # Prioritise critical defensive moves
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


    def find_bridge_threats(self, board: Board, player: Colour) -> list[tuple[int, int]]:
        """Identify potential bridge formations by opponent"""
        board_hash = self.hash_board(board)
        cache_key = (board_hash, player)
        if cache_key in self.opponent_bridges:
            return self.opponent_bridges[cache_key]

        threats = []
        for i in range(board.size):
            for j in range(board.size):
                if board.tiles[i][j].colour == player:
                    # Check diagonal bridge patterns
                    for di, dj in [(1,1), (1,-1), (-1,1), (-1,-1)]:
                        ni, nj = i + di*2, j + dj*2
                        if (0 <= ni < board.size and 0 <= nj < board.size and
                            board.tiles[ni][nj].colour == player and
                            board.tiles[i+di][j+dj].colour is None):
                            threats.append((i+di, j+dj))

        self.opponent_bridges[cache_key] = threats
        return threats


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
        two_bridges_score += self.get_two_bridges(board, move, self.colour)

        # Blocks opponent two bridges
        # This is not worth as much as prioritising our own bridges
        opponent = self.colour.opposite()
        two_bridges_score += (self.get_two_bridges(board, move, opponent) * 0.5)

        return two_bridges_score


    def get_two_bridges(self, board: Board, move: tuple[int, int], colour: Colour):
        """ Returns the number of two bridges made by a given move for a given player. """
        x, y = move
        two_bridges_score = 0

        if self.is_owned_by_player(board, (x - 1, y - 1), colour) \
            and self.is_valid_move(board, (x - 1, y)) \
            and self.is_valid_move(board, (x, y - 1)):
             two_bridges_score += 1
            
        elif self.is_owned_by_player(board, (x + 1, y - 2), colour) \
            and self.is_valid_move(board, (x, y - 1)) \
            and self.is_valid_move(board, (x + 1, y - 1)):
             two_bridges_score += 1

        elif self.is_owned_by_player(board, (x + 2, y - 1), colour) \
            and self.is_valid_move(board, (x + 1, y - 1)) \
            and self.is_valid_move(board, (x + 1, y)):
             two_bridges_score += 1

        elif self.is_owned_by_player(board, (x + 1, y + 1), colour) \
            and self.is_valid_move(board, (x + 1, y)) \
            and self.is_valid_move(board, (x, y + 1)):
             two_bridges_score += 1

        elif self.is_owned_by_player(board, (x - 1, y + 2), colour) \
            and self.is_valid_move(board, (x, y + 1)) \
            and self.is_valid_move(board, (x - 1, y + 1)):
             two_bridges_score += 1

        elif self.is_owned_by_player(board, (x - 2, y + 1), colour) \
            and self.is_valid_move(board, (x - 1, y + 1)) \
            and self.is_valid_move(board, (x - 1, y)):
             two_bridges_score += 1

        return two_bridges_score
    

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
                score += 2.0  # High priority for blocking critical paths
                break

        # Check if move blocks bridge threats
        bridge_threats = self.find_bridge_threats(board, self.opposing_colour)
        if move in bridge_threats:
            score += 1.5  # Significant priority for blocking bridges

        # Check if move prevents opponent from reaching their target side
        if (self.opposing_colour == Colour.RED and x == board.size-1) or \
           (self.opposing_colour == Colour.BLUE and y == board.size-1):
            score += 1.0  # Priority for blocking opponent's goal line

        self.defensive_scores[cache_key] = score
        return score


    def select_node(self, node: RaveMCTSNode, board: Board) -> tuple:
        """
        Select most promising node to explore using UCB1 with RAVE.
        
        Selection Process:
        1. Start at root node
        2. While node is fully expanded:
           - Calculate UCB1+RAVE value for each child
           - Choose child with highest combined score
           - Apply move to board state
           - Track moves for later reversal
        
        Early Stopping:
        - Returns immediately if unvisited node found
        - Prevents unnecessary tree traversal
        
        Returns:
        - Selected node for expansion/simulation
        - List of moves made during selection (for reversal)
        """
        played_moves = []
        while node.untried_moves == [] and node.children:
            max_value = max(node.children, key=lambda n: n.ucb1()).ucb1()
            max_nodes = [n for n in node.children if n.ucb1() == max_value]
            node = choice(max_nodes)
            move = node.move
            player = self.get_next_player(node.parent)
            board.set_tile_colour(move[0], move[1], player)
            played_moves.append((move, player))

            if node.visits == 0:
                return node, played_moves

        return node, played_moves


    def validate_board_state(self, board: Board) -> None:
        """Ensure board state is consistent"""
        self.current_board_state = deepcopy(board)


    def is_valid_move(self, board: Board, move: tuple[int, int]) -> bool:
        """ 
        Checks if a move is within the bounds of the board and that tile 
        is not already taken by either player. 
        
        Returns:
            - True; if move is valid and colour is None.
            - False; otherwise.
        """
        if not (0 <= move[0] < board.size and 0 <= move[1] < board.size):
            return False
        # Check both current state and provided board
        if self.current_board_state:
            if self.current_board_state.tiles[move[0]][move[1]].colour is not None:
                return False
        return board.tiles[move[0]][move[1]].colour is None
    

    def is_owned_by_player(self, board: Board, move: tuple[int, int], player: Colour) -> bool:
        """ 
        Checks if a tile/move is within the bounds of the board and it is owned by the specified player.
        
        Returns:
            - True; if tile is within bounds and owned by the current player.
            - False; otherwise.
        """
        # If not in bounds
        if not (0 <= move[0] < board.size and 0 <= move[1] < board.size):
            return False

        # Returns True if owned by the specified player, false otherwise.
        return board.tiles[move[0]][move[1]].colour is player


    def get_all_positions_for_colour(self, board: Board, colour: Colour) -> list[Move]:
        """ Get all nodes that are placed down, of that colour, in the current game state. """
        all_positions = []

        for x in range(board.size):
            for y in range(board.size):
                t = board.tiles[x][y]
                if t.colour is colour:
                    all_positions.append(Move(x, y))

        return all_positions


    def expand(self, node: RaveMCTSNode, board: Board) -> RaveMCTSNode:
        """Improved expansion with move validation"""
        if node.untried_moves is None:
            node.untried_moves = [move for move in self.get_smart_moves(board)
                                if self.is_valid_move(board, move)]
        if node.untried_moves:
            move = node.untried_moves[0]  # Take best move from sorted list
            if not self.is_valid_move(board, move):
                node.untried_moves.remove(move)
                return node
            
            node.untried_moves = node.untried_moves[1:]
            next_player = self.get_next_player(node)
            new_board = deepcopy(board)
            new_board.set_tile_colour(move[0], move[1], next_player)
            child = RaveMCTSNode(board=new_board, parent=node, move=move)
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
        
        Key Concepts:
        1. Transposition Table:
           - Dictionary caching previous simulation results
           - Key: Zobrist hash of board position
           - Value: Wins, visits, and move sequences
           - Avoids repeating similar simulations
        
        2. Smart Playouts:
           - Uses 80/20 mix of strategic/random moves
           - Strategic moves from evaluate_move()
           - Random moves prevent deterministic play
           - Better than pure random simulation
        
        3. Move Collection:
           - Tracks moves separately for each color
           - Enables proper RAVE statistics updates
           - Helps identify move patterns
        
        4. Early Game Termination:
           - Checks for wins after 5 moves
           - Reduces simulation length
           - More efficient than playing to end
        
        Memory Management:
        - Minimal board copying
        - Efficient move tracking
        - Cached results for common positions
        """
        board_hash = self.hash_board(board)
        if board_hash in self.transposition_table:
            cached_result = self.transposition_table[board_hash]
            if cached_result['visits'] > 10:
                return cached_result['result'], cached_result['red_moves'], cached_result['blue_moves']

        temp_board = deepcopy(board)
        current_player = self.colour  # Start with agent's color
        red_moves = []
        blue_moves = []
        moves_made = 0
        
        while True:
            if moves_made > 5:
                if temp_board.has_ended(current_player):
                    result = (temp_board._winner == self.colour)
                    self.transposition_table[board_hash] = {
                        'result': result,
                        'red_moves': red_moves,
                        'blue_moves': blue_moves,
                        'visits': 1
                    }
                    return result, red_moves, blue_moves

            moves = [m for m in self.get_smart_moves(temp_board) 
                    if self.is_valid_move(temp_board, m)]
            if not moves:
                break

            move = moves[0] if moves and random() < 0.8 else choice(moves)
            if not self.is_valid_move(temp_board, move):
                continue

            temp_board.set_tile_colour(move[0], move[1], current_player)
            self.get_player_moves(current_player, move, red_moves, blue_moves)
            
            # Use dedicated method for switching players
            current_player = self.switch_player(current_player)
            moves_made += 1
            
        result = (temp_board._winner == self.colour)
        self.transposition_table[board_hash] = {
            'result': result,
            'red_moves': red_moves,
            'blue_moves': blue_moves,
            'visits': 1
        }
        return result, red_moves, blue_moves


    def backpropagate(self, node: RaveMCTSNode, result: bool, moves_played: list):
        """
        Update statistics in all nodes from leaf to root.
        
        Update Process:
        1. Standard MCTS Updates:
           - Increment visit count
           - Add win score if simulation was won
        
        2. RAVE Updates:
           - Update AMAF stats for all moves
           - Only count moves made by winning player
           - Helps identify generally good moves
        
        3. Color Awareness:
           - Properly handles alternating players
           - Ensures moves get credited to right player
        
        Optimisation:
        - Updates both MCTS and RAVE stats in one pass
        - Efficient move list processing
        - Minimal redundant calculations
        """
        while node is not None:
            node.visits += 1
            node.wins += self.win_score if result else 0

            # Update AMAF statistics for each move in the playout
            for move in moves_played:
                # Only update AMAF stats for moves made by the same color
                if node.player == self.colour:
                    node.update_amaf_stats(move, result)
                else:
                    node.update_amaf_stats(move, not result)

            node = node.parent


    def get_next_player(self, node: RaveMCTSNode) -> Colour:
        if node.player is None:
            return self.colour
        else:
            return Colour.RED if node.player == Colour.BLUE else Colour.BLUE


    def check_immediate_win(self, board: Board, move: tuple[int, int]) -> bool:
        """Check if a move leads to immediate win"""
        test_board = deepcopy(board)
        if not self.is_valid_move(test_board, move):
            return False
        test_board.set_tile_colour(move[0], move[1], self.colour)
        test_board.has_ended(self.colour)
        return test_board._winner == self.colour


    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        """
        Main decision function combining all MCTS+RAVE elements.
        
        Decision Process:
        1. Opening Strategy:
           - Special handling for first few moves
           - Smart swap move decisions
           - Center control priority
        
        2. Win Detection:
           - Checks for immediate winning moves
           - Validates moves before committing
        
        3. MCTS Search:
           - Runs simulations up to limit
           - Uses RAVE for move evaluation
           - Early stopping when clear best move found
        
        4. Move Selection:
           - Chooses most visited child
           - Validates move legality
           - Has fallback options if primary fails
        
        Safety Features:
        - Extensive move validation
        - Multiple fallback strategies
        - Memory management
        - Time control awareness
        """
        # Update current board state
        self.validate_board_state(board)

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

            # Swap opening move if:
            # - The opponent's move is exactly in the center; or,
            # - The opponent's move is near the center (within the swap_distance_threshold); or,
            # - The opponent's move strength is greater than swap_strength_threshold.
            if is_center or distance_to_center <= swap_distance_threshold or opp_move_strength >= swap_strength_threshold:
                # Decide to swap
                return Move(-1, -1)

        # First check for immediate winning moves
        # It is impossible to win before turn 21.
        if (turn >= 21):
            valid_moves = [m for m in self.get_valid_moves(board) if self.is_valid_move(board, m)]
            for move in valid_moves:
                if self.check_immediate_win(board, move):
                    return Move(move[0], move[1])

        root_node = RaveMCTSNode(board=deepcopy(board))
        root_node.player = Colour.opposite(self.colour)

        best_visits = 0

        for i in range(self.simulations):
            node = root_node
            temp_board = board

            # Selection
            played_moves = []
            node, selection_moves = self.select_node(node, temp_board)
            played_moves.extend(selection_moves)

            # Expansion
            node = self.expand(node, temp_board)

            # Simulation and backpropagation with correct move lists
            outcome, red_moves, blue_moves = self.simulate(temp_board)
            moves_for_backprop = red_moves if self.colour == Colour.RED else blue_moves
            self.backpropagate(node, outcome, moves_for_backprop)

            # Undo moves
            for move, player in reversed(played_moves):
                temp_board.set_tile_colour(move[0], move[1], None)

            # Early stopping based on visit ratio and threshold
            if i > 100:  # Minimum simulations before checking
                best_child = max(root_node.children, key=lambda c: c.visits)
                best_visits = max(best_visits, best_child.visits)
                visit_ratio = best_child.visits / (i + 1)
                
                if visit_ratio > self.min_visits_ratio and best_child.wins / best_child.visits > self.early_stop_threshold:
                    break

        # Cleanup transposition table periodically
        if len(self.transposition_table) > 10000:
            self.transposition_table.clear()

        # Validate best move before returning
        if root_node.children:
            best_child = max(root_node.children, key=lambda c: c.visits)
            best_move = best_child.move
            if self.is_valid_move(board, best_move):
                test_board = deepcopy(board)
                test_board.set_tile_colour(best_move[0], best_move[1], self.colour)
                if test_board.tiles[best_move[0]][best_move[1]].colour == self.colour:
                    self.root = best_child
                    self.root.parent = None
                    return Move(best_move[0], best_move[1])

        # Safe fallback with explicit validation
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
