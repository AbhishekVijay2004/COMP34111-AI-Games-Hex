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
    Each node represents a game state and holds statistics for UCT and RAVE calculations.
    """
    def __init__(self, board: Board, parent=None, move=None):
        self.board = board
        self.parent = parent
        self.move = move
        self.children = []
        self.wins = 0
        self.visits = 0
        self.unexplored_children = self.get_possible_moves(board)
        self.board_hash = None
        self.creates_two_bridge = False

        # RAVE (AMAF) statistics: store for moves (x,y)
        self.amaf_wins = {}
        self.amaf_visits = {}

    def get_possible_moves(self, board: Board) -> list[Move]:
        """Get all valid moves at the current state."""
        valid_moves = []
        for x in range(board.size):
            for y in range(board.size):
                if board.tiles[x][y].colour is None:
                    valid_moves.append(Move(x, y))
        return valid_moves

    def calculate_uct_rave_score(self, child, c_param=1.4, rave_equivalence=300) -> float:
        """
        Calculate combined UCT-RAVE score.
        UCT: standard upper confidence bound measure
        RAVE: using AMAF statistics to speed convergence
        """
        if child.visits == 0:
            return float('inf')

        # UCT part
        exploitation = child.wins / child.visits
        exploration = sqrt((2 * log(self.visits)) / child.visits)
        uct_value = exploitation + c_param * exploration

        # RAVE part
        move_key = (child.move.x, child.move.y)
        amaf_w = self.amaf_wins.get(move_key, 0)
        amaf_v = self.amaf_visits.get(move_key, 0)
        amaf_value = (amaf_w / amaf_v) if amaf_v > 0 else 0.5

        # Combine UCT and RAVE
        beta = sqrt(rave_equivalence / (3 * self.visits + rave_equivalence))
        combined = beta * amaf_value + (1 - beta) * uct_value
        return combined

    def copy_board(self, board: Board) -> Board:
        """Create an independent copy of the board state."""
        new_board = Board(board.size)
        new_board._winner = board._winner
        for i in range(board.size):
            for j in range(board.size):
                new_board.tiles[i][j].colour = board.tiles[i][j].colour
        return new_board

    def apply_move(self, move: Move, colour: Colour) -> Board:
        """Apply a move to the board copy and return the updated state."""
        new_board = self.copy_board(self.board)
        new_board.tiles[move.x][move.y].colour = colour
        return new_board


class MCTSAgent(AgentBase):
    """
    A Monte Carlo Tree Search based agent that uses heuristics, caching, 
    two-bridge knowledge, and AMAF/RAVE optimization.
    """
    _board_size: int = 11
    _colour: Colour

    BRIDGE_PATTERNS = [
        ((-1, -1), (-1, 0), (0, -1)),    # Top-left
        ((1, -2), (0, -1), (1, -1)),     # Top-right
        ((2, -1), (1, -1), (1, 0)),      # Right
        ((1, 1), (1, 0), (0, 1)),        # Bottom-right
        ((-1, 2), (0, 1), (-1, 1)),      # Bottom-left
        ((-2, 1), (-1, 1), (-1, 0)),     # Left
    ]

    def __init__(self, colour: Colour):
        super().__init__(colour)
        self._colour = colour

        # Initialize Zobrist hashing
        random.seed(42)
        self.zobrist_table = {}
        for x in range(self._board_size):
            for y in range(self._board_size):
                self.zobrist_table[(x, y, Colour.RED)] = random.getrandbits(64)
                self.zobrist_table[(x, y, Colour.BLUE)] = random.getrandbits(64)
        self.transposition_table = {}

        self.bridge_cache = {}

        self.weights = {
            'center_weight': 0.5,
            'two_bridge_weight': 4.0,
        }

    def _clear_bridge_cache(self, board: Board):
        self.bridge_cache.clear()

    def _get_cache_key(self, board: Board, colour: Colour) -> int:
        return hash((self.hash_board(board), colour))

    def get_possible_moves(self, board: Board) -> list[Move]:
        valid_moves = []
        for x in range(board.size):
            for y in range(board.size):
                if board.tiles[x][y].colour is None:
                    valid_moves.append(Move(x, y))
        return valid_moves

    def is_my_stone(self, board: Board, position: Move | tuple[int, int]) -> bool:
        x, y = (position.x, position.y) if isinstance(position, Move) else position
        return (0 <= x < board.size and 0 <= y < board.size and
                board.tiles[x][y].colour == self._colour)

    def check_immediate_win(self, board: Board, position: Move | tuple[int, int], player: Colour) -> bool:
        x, y = (position.x, position.y) if isinstance(position, Move) else position
        if not self.is_valid_move(board, x, y):
            return False
        board.set_tile_colour(x, y, player)
        won = board.has_ended(player)
        board.set_tile_colour(x, y, None)
        return won

    def get_all_positions_for_colour(self, board: Board, colour: Colour) -> list[Move]:
        positions = []
        for x in range(board.size):
            for y in range(board.size):
                if board.tiles[x][y].colour == colour:
                    positions.append(Move(x, y))
        return positions

    def copy_board(self, board: Board) -> Board:
        new_board = Board(board.size)
        new_board._winner = board._winner
        for i in range(board.size):
            for j in range(board.size):
                new_board.tiles[i][j].colour = board.tiles[i][j].colour
        return new_board

    def is_valid_move(self, board: Board, x: int, y: int) -> bool:
        return 0 <= x < board.size and 0 <= y < board.size and board.tiles[x][y].colour is None

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        start_time = time.time()
        max_time = 1.9

        # Check swap logic early in the game
        if turn == 2 and opp_move and self.should_we_swap(opp_move):
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

        # Save threatened two-bridges if possible
        if opp_move:
            save_move = self.save_two_bridge(board, opp_move)
            if save_move:
                return random.choice(save_move)

        # Pure MCTS with RAVE
        root = MCTSNode(board)
        iterations = 0

        while time.time() - start_time < max_time:
            current_node = self.select_node(root)
            current_node = self.expand_node(current_node)
            played_moves, result = self.simulate(current_node.board)
            self.backpropagate(current_node, played_moves, result)
            iterations += 1

        # Choose best move by visits
        best_child = max(root.children, key=lambda c: c.visits)
        return Move(best_child.move.x, best_child.move.y)

    def select_node(self, current_node: MCTSNode) -> MCTSNode:
        """Select child node using UCT-RAVE."""
        while not current_node.unexplored_children and current_node.children:
            current_node = max(current_node.children, key=lambda child: current_node.calculate_uct_rave_score(child))
        return current_node

    def expand_node(self, node: MCTSNode) -> MCTSNode:
        """Expand a node by exploring an unexplored move."""
        if node.unexplored_children:
            _, possible_bridges = self.analyze_two_bridges(node.board, self._colour)
            candidate_moves = [m for m in node.unexplored_children if any(m.x == pb.x and m.y == pb.y for pb in possible_bridges)]

            if candidate_moves:
                move = random.choice(candidate_moves)
            else:
                move = random.choice(node.unexplored_children)

            node.unexplored_children.remove(move)
            new_board = node.apply_move(move, self._colour)
            child = MCTSNode(new_board, node, move)
            child.board_hash = self.hash_board(new_board)
            child.creates_two_bridge = any((move.x == pb.x and move.y == pb.y) for pb in possible_bridges)
            node.children.append(child)
            return child
        return node

    def simulate(self, state: Board) -> tuple[list[Move], bool]:
        """
        Run a random simulation to a terminal state.
        Return the moves played and the result (win or loss).
        """
        current_hash = self.hash_board(state)
        if current_hash in self.transposition_table:
            # We don't have a record of played moves for cached results, so return empty list
            return [], self.transposition_table[current_hash]

        simulation_board = self.copy_board(state)
        simulation_colour = self._colour
        moves = self.get_possible_moves(simulation_board)
        played_moves = []

        while not simulation_board.has_ended(Colour.RED) and not simulation_board.has_ended(Colour.BLUE):
            if not moves:
                break
            move = random.choice(moves)
            simulation_board.set_tile_colour(move.x, move.y, simulation_colour)
            played_moves.append((move, simulation_colour))
            moves.remove(move)
            simulation_colour = Colour.opposite(simulation_colour)

        result = (simulation_board._winner == self._colour)
        self.transposition_table[current_hash] = result
        return played_moves, result

    def hash_board(self, board: Board) -> int:
        h = 0
        for x in range(board.size):
            for y in range(board.size):
                color = board.tiles[x][y].colour
                if color is not None:
                    h ^= self.zobrist_table[(x, y, color)]
        return h

    def backpropagate(self, node: MCTSNode, played_moves: list, won: bool):
        """
        Backpropagation step:
        - Update node's wins and visits.
        - Update AMAF/RAVE stats for all moves encountered in the simulation.
        """
        moves_set = {(m.x, m.y) for m, c in played_moves if c == self._colour}
        while node:
            node.visits += 1
            if won:
                node.wins += 1
            # Update AMAF/RAVE stats for moves from the simulation
            for (mx, my) in moves_set:
                node.amaf_visits[(mx, my)] = node.amaf_visits.get((mx, my), 0) + 1
                if won:
                    node.amaf_wins[(mx, my)] = node.amaf_wins.get((mx, my), 0) + 1
            node = node.parent

    def analyze_two_bridges(self, board: Board, colour: Colour):
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
                if (self.is_valid_move(board, bx, by) and
                    self.is_valid_move(board, c1x, c1y) and
                    self.is_valid_move(board, c2x, c2y)):
                    possible_bridges.add(Move(bx, by))

        possible_bridges = list(possible_bridges)
        self.bridge_cache[cache_key] = (existing_bridges, possible_bridges)
        return existing_bridges, possible_bridges

    def save_two_bridge(self, board: Board, opp_move: Move) -> list[Move] | None:
        x, y = opp_move.x, opp_move.y
        original_color = board.tiles[x][y].colour

        current_existing, _ = self.analyze_two_bridges(board, self._colour)
        current_bridge_positions = {bridge[0] for bridge in current_existing}

        board.set_tile_colour(x, y, None)
        self._clear_bridge_cache(board)

        previous_existing, _ = self.analyze_two_bridges(board, self._colour)

        board.set_tile_colour(x, y, original_color)
        self._clear_bridge_cache(board)

        moves_to_save = set()

        for bridge_pos, empty_cells in previous_existing:
            if bridge_pos not in current_bridge_positions:
                cell1, cell2 = empty_cells
                if (x, y) == (cell1.x, cell1.y) and self.is_valid_move(board, cell2.x, cell2.y):
                    moves_to_save.add(Move(cell2.x, cell2.y))
                elif (x, y) == (cell2.x, cell2.y) and self.is_valid_move(board, cell1.x, cell1.y):
                    moves_to_save.add(Move(cell1.x, cell1.y))

        return list(moves_to_save) if moves_to_save else None

    def should_we_swap(self, opp_move: Move) -> bool:
        swap_moves = []
        for x in range(2, 9):
            for y in range(11):
                if not (x == 2 and y == 0) and not (x == 8 and y == 0) \
                   and not (x == 2 and y == 10) and not (x == 8 and y == 10):
                    swap_moves.append(Move(x, y))
        
        swap_moves.extend([
            Move(0, 10), Move(1, 9), Move(1, 10),
            Move(9, 0), Move(10, 0), Move(9, 1)
        ])

        if opp_move is not None and self._colour == Colour.BLUE:
            if opp_move in swap_moves:
                return True
        return False