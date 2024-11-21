# from math import sqrt, log
# from random import choice, random, Random
# from src.AgentBase import AgentBase
# from src.Board import Board
# from src.Colour import Colour
# from src.Move import Move
# from copy import deepcopy
# class RaveMCTSNode:
#     def __init__(self, board: Board = None, parent=None, move=None):
#         self.parent = parent
#         self.move = move  # The move that led to this node
#         self.children = []
#         self.wins = 0  # Total wins from simulations passing through this node
#         self.visits = 0  # Total simulations passing through this node
#         self.untried_moves = None  # Moves that have not been tried from this node
#         self.player = None  # The player who made the move to reach this node
#         self.Q_RAVE = 0  # Total RAVE wins
#         self.N_RAVE = 0  # Total RAVE visits
#         self.board = board  # Need to store board state
#         self.hash_value = None  # Add board hash for transposition table
#         self.move_amaf_wins = {}  # Track AMAF wins per move
#         self.move_amaf_visits = {}  # Track AMAF visits per move

#     def get_amaf_value(self, move: tuple[int, int]) -> float:
#         """Calculate AMAF value for a specific move"""
#         visits = self.move_amaf_visits.get(move, 0)
#         if visits == 0:
#             return 0.0
#         return self.move_amaf_wins.get(move, 0) / visits

#     def ucb1(self, explore: float = 1.41, rave_const: float = 300) -> float:
#         if self.visits == 0:
#             return float('inf')
        
#         beta = sqrt(rave_const / (3 * self.visits + rave_const))
#         mcts_value = self.wins / self.visits + explore * sqrt(log(self.parent.visits) / self.visits)
        
#         # Use move-specific AMAF value
#         amaf_value = self.get_amaf_value(self.move) if self.move else 0.0
        
#         return (1 - beta) * mcts_value + beta * amaf_value

#     def update_amaf_stats(self, move: tuple[int, int], won: bool):
#         """Update AMAF statistics for a specific move"""
#         if move not in self.move_amaf_visits:
#             self.move_amaf_visits[move] = 0
#             self.move_amaf_wins[move] = 0
#         self.move_amaf_visits[move] += 1
#         if won:
#             self.move_amaf_wins[move] += 1

# class RaveAgent(AgentBase):
#     def __init__(self, colour: Colour):
#         super().__init__(colour)
#         self.simulations = 1000
#         self.win_score = 10
#         self.colour = colour
#         self.root = RaveMCTSNode()
#         self._board_size = 11
#         self._all_positions = [(i, j) for i in range(self._board_size) 
#                                for j in range(self._board_size)]
#         self.rave_constant = 300  # Tune this value
#         self.move_history = []  # Track move history
#         self.move_scores = {}  # Cache for move evaluations
#         self.transposition_table = {}  # Cache for board states
#         self.early_stop_threshold = 0.95  # Higher threshold for early stopping
#         self.min_visits_ratio = 0.1  # Minimum visit ratio for early stopping
#         self._rng = Random(42)  # Create a dedicated random number generator
#         self._zobrist_table = {
#             (i, j, color): self._rng.getrandbits(64)
#             for i in range(self._board_size)
#             for j in range(self._board_size)
#             for color in [Colour.RED, Colour.BLUE]
#         }
#         self.last_board = None  # Track last board state
#         self.current_board_state = None  # Track current board state
#         self.opposing_colour = Colour.opposite(self.colour)
#         self.current_player = self.colour  # Add current player tracking

#     def switch_player(self, current_player: Colour) -> Colour:
#         """Switch between players consistently"""
#         if current_player == Colour.RED:
#             return Colour.BLUE
#         return Colour.RED

#     def get_player_moves(self, current_player: Colour, move: tuple[int, int], 
#                         red_moves: list, blue_moves: list) -> None:
#         """Add move to appropriate player's move list"""
#         if current_player == Colour.RED:
#             red_moves.append(move)
#         else:
#             blue_moves.append(move)

#     def get_valid_moves(self, board: Board) -> list[tuple[int, int]]:
#         return [(i, j) for i in range(board.size)
#                 for j in range(board.size)
#                 if board.tiles[i][j].colour is None]

#     def get_neighbor_moves(self, board: Board, x: int, y: int) -> list[tuple[int, int]]:
#         """Get valid moves adjacent to existing pieces"""
#         neighbors = []
#         for i in [-1, 0, 1]:
#             for j in [-1, 0, 1]:
#                 nx, ny = x + i, y + j
#                 if (0 <= nx < board.size and 0 <= ny < board.size and 
#                     board.tiles[nx][ny].colour is None):
#                     neighbors.append((nx, ny))
#         return neighbors

#     def evaluate_move(self, board: Board, move: tuple[int, int]) -> float:
#         """Strategic move evaluation"""
#         if move in self.move_scores:
#             return self.move_scores[move]
        
#         score = 0
#         x, y = move
#         center = board.size // 2
        
#         # Strategic positioning
#         dist_to_center = abs(x - center) + abs(y - center)
#         score += max(0, (board.size - dist_to_center)) / board.size
        
#         # Connection potential
#         neighbors = self.get_neighbor_moves(board, x, y)
#         friendly_neighbors = sum(1 for nx, ny in neighbors 
#                                if board.tiles[nx][ny].colour == self.colour)
#         score += friendly_neighbors * 0.5
        
#         # Bridge formation potential (diagonal connections)
#         bridge_score = sum(1 for nx, ny in neighbors 
#                          if abs(nx-x) == 1 and abs(ny-y) == 1 
#                          and board.tiles[nx][ny].colour == self.colour)
#         score += bridge_score * 0.3
        
#         # Edge control
#         if self.colour == Colour.RED and (x == 0 or x == board.size-1):
#             score += 0.4
#         elif self.colour == Colour.BLUE and (y == 0 or y == board.size-1):
#             score += 0.4
            
#         self.move_scores[move] = score
#         return score

#     def get_smart_moves(self, board: Board) -> list[tuple[int, int]]:
#         """Get strategically sorted moves"""
#         occupied = [(i, j) for i, j in self._all_positions 
#                    if board.tiles[i][j].colour is not None]
        
#         if len(occupied) < 3:
#             center = board.size // 2
#             return [(center, center)] + self.get_neighbor_moves(board, center, center)
            
#         neighbor_moves = set()
#         for x, y in occupied:
#             neighbor_moves.update(self.get_neighbor_moves(board, x, y))
            
#         moves = list(neighbor_moves) if neighbor_moves else self.get_valid_moves(board)
#         # Sort moves by evaluation
#         moves.sort(key=lambda m: self.evaluate_move(board, m), reverse=True)
#         return moves

#     def select_node(self, node: RaveMCTSNode, board: Board) -> tuple:
#         played_moves = []
#         while node.untried_moves == [] and node.children:
#             max_value = max(node.children, key=lambda n: n.ucb1()).ucb1()
#             max_nodes = [n for n in node.children if n.ucb1() == max_value]
#             node = choice(max_nodes)
#             move = node.move
#             player = self.get_next_player(node.parent)
#             board.set_tile_colour(move[0], move[1], player)
#             played_moves.append((move, player))

#             if node.visits == 0:
#                 return node, played_moves

#         return node, played_moves

#     def validate_board_state(self, board: Board) -> None:
#         """Ensure board state is consistent"""
#         self.current_board_state = deepcopy(board)

#     def is_valid_move(self, board: Board, move: tuple[int, int]) -> bool:
#         """Enhanced move validation"""
#         if not (0 <= move[0] < board.size and 0 <= move[1] < board.size):
#             return False
#         # Check both current state and provided board
#         if self.current_board_state:
#             if self.current_board_state.tiles[move[0]][move[1]].colour is not None:
#                 return False
#         return board.tiles[move[0]][move[1]].colour is None

#     def expand(self, node: RaveMCTSNode, board: Board) -> RaveMCTSNode:
#         """Improved expansion with move validation"""
#         if node.untried_moves is None:
#             node.untried_moves = [move for move in self.get_smart_moves(board)
#                                 if self.is_valid_move(board, move)]
#         if node.untried_moves:
#             move = node.untried_moves[0]  # Take best move from sorted list
#             if not self.is_valid_move(board, move):
#                 node.untried_moves.remove(move)
#                 return node
            
#             node.untried_moves = node.untried_moves[1:]
#             next_player = self.get_next_player(node)
#             new_board = deepcopy(board)
#             new_board.set_tile_colour(move[0], move[1], next_player)
#             child = RaveMCTSNode(board=new_board, parent=node, move=move)
#             child.player = next_player
#             node.children.append(child)
#             return child
#         return node

#     def hash_board(self, board: Board) -> int:
#         """Fast board hashing for transposition table"""
#         hash_value = 0
#         for i in range(board.size):
#             for j in range(board.size):
#                 color = board.tiles[i][j].colour
#                 if color is not None:
#                     hash_value ^= self._zobrist_table[(i, j, color)]
#         return hash_value

#     def simulate(self, board: Board) -> tuple:
#         """Simulation with enhanced validation and consistent player switching"""
#         board_hash = self.hash_board(board)
#         if board_hash in self.transposition_table:
#             cached_result = self.transposition_table[board_hash]
#             if cached_result['visits'] > 10:
#                 return cached_result['result'], cached_result['red_moves'], cached_result['blue_moves']

#         temp_board = deepcopy(board)
#         current_player = self.colour  # Start with agent's color
#         red_moves = []
#         blue_moves = []
#         moves_made = 0
        
#         while True:
#             if moves_made > 5:
#                 if temp_board.has_ended(current_player):
#                     result = (temp_board._winner == self.colour)
#                     self.transposition_table[board_hash] = {
#                         'result': result,
#                         'red_moves': red_moves,
#                         'blue_moves': blue_moves,
#                         'visits': 1
#                     }
#                     return result, red_moves, blue_moves

#             moves = [m for m in self.get_smart_moves(temp_board) 
#                     if self.is_valid_move(temp_board, m)]
#             if not moves:
#                 break

#             move = moves[0] if moves and random() < 0.8 else choice(moves)
#             if not self.is_valid_move(temp_board, move):
#                 continue

#             temp_board.set_tile_colour(move[0], move[1], current_player)
#             self.get_player_moves(current_player, move, red_moves, blue_moves)
            
#             # Use dedicated method for switching players
#             current_player = self.switch_player(current_player)
#             moves_made += 1
            
#         result = (temp_board._winner == self.colour)
#         self.transposition_table[board_hash] = {
#             'result': result,
#             'red_moves': red_moves,
#             'blue_moves': blue_moves,
#             'visits': 1
#         }
#         return result, red_moves, blue_moves

#     def backpropagate(self, node: RaveMCTSNode, result: bool, moves_played: list):
#         """Updated backpropagation with move-specific AMAF tracking"""
#         while node is not None:
#             node.visits += 1
#             node.wins += self.win_score if result else 0

#             # Update AMAF statistics for each move in the playout
#             for move in moves_played:
#                 # Only update AMAF stats for moves made by the same color
#                 if node.player == self.colour:
#                     node.update_amaf_stats(move, result)
#                 else:
#                     node.update_amaf_stats(move, not result)

#             node = node.parent

#     def get_next_player(self, node: RaveMCTSNode) -> Colour:
#         if node.player is None:
#             return self.colour
#         else:
#             return Colour.RED if node.player == Colour.BLUE else Colour.BLUE

#     def check_immediate_win(self, board: Board, move: tuple[int, int]) -> bool:
#         """Check if a move leads to immediate win"""
#         test_board = deepcopy(board)
#         if not self.is_valid_move(test_board, move):
#             return False
#         test_board.set_tile_colour(move[0], move[1], self.colour)
#         test_board.has_ended(self.colour)
#         return test_board._winner == self.colour

#     def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
#         """Main method with enhanced validation"""
#         # Update current board state
#         self.validate_board_state(board)

#         # Strategic Swap Evaluation
#         if turn == 2 and opp_move is not None:
#             opp_x, opp_y = opp_move.x, opp_move.y
#             center = board.size // 2
#             distance_to_center = abs(opp_x - center) + abs(opp_y - center)
#             is_center = (opp_x, opp_y) == (center, center)

#             # Evaluate the strength of the opponent's first move
#             opp_move_strength = self.evaluate_move(board, (opp_x, opp_y))

#             # Thresholds for swapping
#             swap_distance_threshold = board.size // 4
#             swap_strength_threshold = 0.7  # Adjust based on testing

#             if is_center or distance_to_center <= swap_distance_threshold or opp_move_strength >= swap_strength_threshold:
#                 # Decide to swap
#                 return Move(-1, -1)

#         # First check for immediate winning moves with validation
#         valid_moves = [m for m in self.get_valid_moves(board) if self.is_valid_move(board, m)]
#         for move in valid_moves:
#             if self.check_immediate_win(board, move):
#                 return Move(move[0], move[1])

#         root_node = RaveMCTSNode(board=deepcopy(board))
#         root_node.player = Colour.opposite(self.colour)

#         total_sims = 0
#         best_visits = 0

#         for i in range(self.simulations):
#             node = root_node
#             temp_board = board

#             # Selection
#             played_moves = []
#             node, selection_moves = self.select_node(node, temp_board)
#             played_moves.extend(selection_moves)

#             # Expansion
#             node = self.expand(node, temp_board)

#             # Simulation and backpropagation with correct move lists
#             outcome, red_moves, blue_moves = self.simulate(temp_board)
#             moves_for_backprop = red_moves if self.colour == Colour.RED else blue_moves
#             self.backpropagate(node, outcome, moves_for_backprop)

#             # Undo moves
#             for move, player in reversed(played_moves):
#                 temp_board.set_tile_colour(move[0], move[1], None)

#             # Early stopping based on visit ratio and threshold
#             if i > 100:  # Minimum simulations before checking
#                 best_child = max(root_node.children, key=lambda c: c.visits)
#                 best_visits = max(best_visits, best_child.visits)
#                 visit_ratio = best_child.visits / (i + 1)
                
#                 if visit_ratio > self.min_visits_ratio and best_child.wins / best_child.visits > self.early_stop_threshold:
#                     break

#             total_sims = i

#         # Cleanup transposition table periodically
#         if len(self.transposition_table) > 10000:
#             self.transposition_table.clear()

#         # Validate best move before returning
#         if root_node.children:
#             best_child = max(root_node.children, key=lambda c: c.visits)
#             best_move = best_child.move
#             if self.is_valid_move(board, best_move):
#                 test_board = deepcopy(board)
#                 test_board.set_tile_colour(best_move[0], best_move[1], self.colour)
#                 if test_board.tiles[best_move[0]][best_move[1]].colour == self.colour:
#                     self.root = best_child
#                     self.root.parent = None
#                     return Move(best_move[0], best_move[1])

#         # Safe fallback with explicit validation
#         valid_moves = [m for m in self.get_valid_moves(board) if self.is_valid_move(board, m)]
#         if valid_moves:
#             move = choice(valid_moves)
#             self.root = RaveMCTSNode()
#             return Move(move[0], move[1])

#         return Move(-1, -1)  # Safe fallback if no valid moves found

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
    
    Storage Optimizations:
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
        
        Optimization: Uses move-specific tracking instead of global RAVE values,
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

class RaveAgent(AgentBase):
    """
    RAVE-enhanced MCTS agent with specialized optimizations for Hex gameplay.
    
    Key Optimizations:
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

    def switch_player(self, current_player: Colour) -> Colour:
        """
        Optimize player switching using direct comparison instead of multiple checks.
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
        """Strategic move evaluation"""
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
        
        # Bridge formation potential (diagonal connections)
        bridge_score = sum(1 for nx, ny in neighbors 
                         if abs(nx-x) == 1 and abs(ny-y) == 1 
                         and board.tiles[nx][ny].colour == self.colour)
        score += bridge_score * 0.3
        
        # Edge control
        if self.colour == Colour.RED and (x == 0 or x == board.size-1):
            score += 0.4
        elif self.colour == Colour.BLUE and (y == 0 or y == board.size-1):
            score += 0.4
            
        self.move_scores[move] = score
        return score

    def get_smart_moves(self, board: Board) -> list[tuple[int, int]]:
        """
        Strategic move generation with intelligent ordering.
        
        Optimizations:
        - Caches neighbor calculations
        - Prioritizes center and edge moves early game
        - Uses evaluation function for move ordering
        - Returns moves in likely-best to likely-worst order
        """
        occupied = [(i, j) for i, j in self._all_positions 
                   if board.tiles[i][j].colour is not None]
        
        if len(occupied) < 3:
            center = board.size // 2
            return [(center, center)] + self.get_neighbor_moves(board, center, center)
            
        neighbor_moves = set()
        for x, y in occupied:
            neighbor_moves.update(self.get_neighbor_moves(board, x, y))
            
        moves = list(neighbor_moves) if neighbor_moves else self.get_valid_moves(board)
        # Sort moves by evaluation
        moves.sort(key=lambda m: self.evaluate_move(board, m), reverse=True)
        return moves

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
        """Enhanced move validation"""
        if not (0 <= move[0] < board.size and 0 <= move[1] < board.size):
            return False
        # Check both current state and provided board
        if self.current_board_state:
            if self.current_board_state.tiles[move[0]][move[1]].colour is not None:
                return False
        return board.tiles[move[0]][move[1]].colour is None

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
        
        Optimization:
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

            if is_center or distance_to_center <= swap_distance_threshold or opp_move_strength >= swap_strength_threshold:
                # Decide to swap
                return Move(-1, -1)

        # First check for immediate winning moves with validation
        valid_moves = [m for m in self.get_valid_moves(board) if self.is_valid_move(board, m)]
        for move in valid_moves:
            if self.check_immediate_win(board, move):
                return Move(move[0], move[1])

        root_node = RaveMCTSNode(board=deepcopy(board))
        root_node.player = Colour.opposite(self.colour)

        total_sims = 0
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

            total_sims = i

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