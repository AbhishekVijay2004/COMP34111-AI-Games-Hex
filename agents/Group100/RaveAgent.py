from math import log, sqrt
import copy
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
                t = board.tiles[x][y]
                if t.colour is None:
                    valid_moves.append(Move(x, y))

        return valid_moves


    def calculate_uct_score(self, c_param=1.4) -> float:
        """ Choose the best move using Upper Confidence bounds for Trees (UCT) selection method.

            At parent node v, choose child v' which maximises:
                (Q(v') / N(v')) + (C * sqrt((2 * ln(N(v))) / N(v')))
            where 
                Q(v) = the sum of all payoffs received,
                N(v) = the number of times the node has been visited
            and
                C = a custom parameter to determine exploration-exploitation trade-off
        """
        if self.visits == 0:
            return float('inf')  # Prioritize unvisited nodes
        
        # Return the UCT score of the current node
        return ((self.wins / self.visits) + c_param * sqrt((2 * log(self.parent.visits)) / self.visits))
    

    def apply_move(self, move: Move, colour: Colour) -> Board:
        """ Creates a new board with the move applied. """
        new_board = copy.deepcopy(self.board)
        new_board.tiles[move.x][move.y].colour = colour
        return new_board
    

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

    _board_size: int = 11
    _colour: Colour
    

    def __init__(self, colour: Colour):
        super().__init__(colour)
        self._colour = colour
        
        # Initialize Zobrist hashing table
        random.seed(42)  # Fixed seed for reproducibility
        self.zobrist_table = {}
        for x in range(self._board_size):
            for y in range(self._board_size):
                self.zobrist_table[(x, y, Colour.RED)] = random.getrandbits(64)
                self.zobrist_table[(x, y, Colour.BLUE)] = random.getrandbits(64)
        self.transposition_table = {}
        

    def get_possible_moves(self, board: Board) -> list[Move]:
        """ Get all valid moves at the current game state. """
        valid_moves = []

        for x in range(board.size):
            for y in range(board.size):
                t = board.tiles[x][y]
                if t.colour is None:
                    valid_moves.append(Move(x, y))

        return valid_moves
    
    def is_my_stone(self, board: Board,  move: Move | tuple[int, int] ) -> bool:
        """Check if the stone at the given position is mine"""
        if isinstance(move, tuple):
            x, y = move
        else:
            x, y = move.x, move.y
        return 0 <= x < board.size and 0 <= y < board.size and board.tiles[x][y].colour is self._colour
    
    def check_immediate_win(self, board: Board, move: Move | tuple[int, int], player: Colour) -> bool:
        """Check if a move leads to immediate win for specified player"""
        if isinstance(move, Move):
            x, y = move.x, move.y
        else:
            x, y = move

        if not self.is_valid_move(board, (x, y)):
            return False
            
        board.set_tile_colour(x, y, player)
        won = board.has_ended(player)
        board.set_tile_colour(x, y, None)  # Undo the move
        return won

    def get_all_positions_for_colour(self, board: Board, colour: Colour) -> list[Move]:
        """ Get all nodes that are placed down, of that colour, in the current game state. """
        all_positions = []

        for x in range(board.size):
            for y in range(board.size):
                t = board.tiles[x][y]
                if t.colour is colour:
                    all_positions.append(Move(x, y))

        return all_positions
    
    def copy_board(self, board: Board) -> Board:
        """Create an efficient copy of the board state."""
        new_board = Board(board.size)
        new_board._winner = board._winner
        
        # Bulk copy colors in one step per row using list comprehension
        # This is faster than individual tile access
        for i in range(board.size):
            row = board.tiles[i]
            for j in range(board.size):
                new_board.tiles[i][j].colour = row[j].colour
                
        return new_board
    

    def is_valid_move(self, board: Board, move: Move | tuple[int, int]) -> bool:
        """ Checks if a move is within the board boundaries and does not contain a colour. """
        if isinstance(move, tuple):
            x, y = move
        else:
            x, y = move.x, move.y
        return 0 <= x < board.size and 0 <= y < board.size and board.tiles[x][y].colour is None
    

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
        max_time = 1.9

        # If opponent swaps
        if opp_move and opp_move.x == -1 and opp_move.y == -1:  # Swap move
            return Move(board.size // 2, board.size // 2)  # Center is often a good move after swap
        
        # If opponent plays center on their first move
        if turn == 2 and board.tiles[board.size // 2][board.size // 2].colour is not None:
            return Move(-1, -1)  # Swap if center is taken
        
        valid_moves = self.get_possible_moves(board)
        
        # Check for immediate win
        for move in valid_moves:
            if self.check_immediate_win(board, (move.x, move.y), self._colour):
                return move
        
        # Check for immediate loss and block
        for move in valid_moves:
            if self.check_immediate_win(board, (move.x, move.y), Colour.opposite(self._colour)):
                return move
        
        # Save two-bridge if opponent threatens
        if opp_move:
            save_move = self.save_two_bridge(board, opp_move)
            if save_move:
                # pass
                return random.choice(save_move)
        
        # test = self.get_two_bridges_with_positions(board, self._colour)
        # if test != []:
        #     print("Two-bridge positions:")
        #     print(test)
        #     print(board.print_board())
        
        root = MCTSNode(board)
            
        iterations = 0
        while time.time() - start_time < max_time:
            current_node = self.select_node(root)

            current_node = self.expand_node(current_node)

            result = self.simulate(current_node.board)

            self.backpropagate(current_node, result)

            iterations += 1

        # print(f"MCTS Iterations: {iterations}, Time Spent: {time.time() - start_time:.2f}s")

        # Return the move with the most visits (the 'best' one) because it has 
        # been selected the most after all the MCTS iterations.
        best_child = max(root.children, key=lambda c: c.visits)

        return Move(best_child.move.x, best_child.move.y)


    def select_node(self, current_node: MCTSNode) -> MCTSNode:
        """ Select a child of current node to become the next current node. """

        # When all child nodes of current node have been explored, this calculates the UCT 
        # of each child and chooses the best one to become the new current node.
        while current_node.unexplored_children == [] and current_node.children:
            current_node = max(current_node.children, key=lambda child: child.calculate_uct_score())  # Select the best child

        # If the current node has unexplored children, the current node will be returned
        # This is the node that will be expanded to explore the children

        return current_node
        

    def expand_node(self, node: MCTSNode) -> MCTSNode:
        if node.unexplored_children:
            # Get all possible two-bridge moves for current board state
            two_bridge_moves = self.get_possible_two_bridges(node.board, self.colour)
                
            # If there are two-bridge moves, prioritize them
            if two_bridge_moves:
                # Create weights list: 4.0 for two-bridge moves, 1.0 for regular moves
                weights = []
                for move in node.unexplored_children:
                    if any(bridge.x == move.x and bridge.y == move.y for bridge in two_bridge_moves):
                        weights.append(4.0)  # Higher weight for two-bridge moves
                    else:
                        weights.append(1.0)  # Normal weight for regular moves
                        
                # Use weighted random choice
                move = random.choices(node.unexplored_children, weights=weights)[0]
            else:
                # If no two-bridge moves, just pick randomly
                move = random.choice(node.unexplored_children)
                
            node.unexplored_children.remove(move)
            
            # Create new board and child node
            new_board = node.apply_move(move, self.colour)
            child = MCTSNode(new_board, node, move)
            child.board_hash = self.hash_board(new_board)
            
            # Set two-bridge property
            child.creates_two_bridge = any(two_bridge_moves) and any(
                bridge.x == move.x and bridge.y == move.y for bridge in two_bridge_moves
            )
            
            node.children.append(child)
            return child
        return node

    
    def simulate(self, state: Board) -> bool:
        """Enhanced simulation using Zobrist hashing for caching"""
        current_hash = self.hash_board(state)
        
        # Check transposition table
        if current_hash in self.transposition_table:
            return self.transposition_table[current_hash]
        
        simulation_board = self.copy_board(state)
        simulation_colour = self._colour
        current_hash = self.hash_board(simulation_board)

        moves = self.get_possible_moves(simulation_board)
        
        while not simulation_board.has_ended(Colour.RED) and not simulation_board.has_ended(Colour.BLUE):
            if not moves:
                break
                
            random_move = random.choice(moves)
            simulation_board.set_tile_colour(random_move.x, random_move.y, simulation_colour)
            current_hash = self.hash_update(current_hash, random_move.x, random_move.y, simulation_colour)
            moves.remove(random_move)
            
            # Cache intermediate positions
            if current_hash not in self.transposition_table and len(self.transposition_table) < 1000000:  # Limit cache size
                self.transposition_table[current_hash] = simulation_board._winner == self.colour
            
            # Clear cache if it reaches the limit
            if len(self.transposition_table) == 1000000:
                self.transposition_table.clear()
            
            simulation_colour = Colour.opposite(simulation_colour)
        
        result = simulation_board._winner == self.colour
        self.transposition_table[current_hash] = result
        return result

    def hash_board(self, board: Board) -> int:
        """Calculate Zobrist hash for given board state"""
        h = 0
        for x in range(board.size):
            for y in range(board.size):
                color = board.tiles[x][y].colour
                if color is not None:
                    h ^= self.zobrist_table[(x, y, color)]
        return h

    def hash_update(self, hash_val: int, x: int, y: int, color: Colour) -> int:
        """Update existing hash when making/undoing a move"""
        return hash_val ^ self.zobrist_table[(x, y, color)]

    def backpropagate(self, node: MCTSNode, won: bool):
        """ Backpropagate result to all visited nodes. """

        while node:
            node.visits += 1

            if won:
                node.wins += 1

            # Move to parent
            node = node.parent


    def get_possible_two_bridges(self, board: Board, colour: Colour) -> list[Move]:
        """ Get all possible two-bridge moves. """

        two_bridges = []
        current_nodes = self.get_all_positions_for_colour(board, colour)
    
        for node in current_nodes:
            # Pattern 1: Top-left bridge
            if (self.is_valid_move(board, Move(node.x - 1, node.y - 1)) and 
                self.is_valid_move(board, Move(node.x - 1, node.y)) and 
                self.is_valid_move(board, Move(node.x, node.y - 1))):
                two_bridges.append(
                    Move(node.x - 1, node.y - 1)  # bridge position
                )
            
            # Pattern 2: Top-right bridge
            if (self.is_valid_move(board, Move(node.x + 1, node.y - 2)) and 
                self.is_valid_move(board, Move(node.x, node.y - 1)) and 
                self.is_valid_move(board, Move(node.x + 1, node.y - 1))):
                two_bridges.append(
                    Move(node.x + 1, node.y - 2),  # bridge position
                )
    
            # Pattern 3: Right bridge
            if (self.is_valid_move(board, Move(node.x + 2, node.y - 1)) and 
                self.is_valid_move(board, Move(node.x + 1, node.y - 1)) and 
                self.is_valid_move(board, Move(node.x + 1, node.y))):
                two_bridges.append(
                    Move(node.x + 2, node.y - 1),  # bridge position
                )
    
            # Pattern 4: Bottom-right bridge
            if (self.is_valid_move(board, Move(node.x + 1, node.y + 1)) and 
                self.is_valid_move(board, Move(node.x + 1, node.y)) and 
                self.is_valid_move(board, Move(node.x, node.y + 1))):
                two_bridges.append(
                    Move(node.x + 1, node.y + 1),  # bridge position
                )
    
            # Pattern 5: Bottom-left bridge
            if (self.is_valid_move(board, Move(node.x - 1, node.y + 2)) and 
                self.is_valid_move(board, Move(node.x, node.y + 1)) and 
                self.is_valid_move(board, Move(node.x - 1, node.y + 1))):
                two_bridges.append(
                    Move(node.x - 1, node.y + 2),  # bridge position
          
                )
    
            # Pattern 6: Left bridge
            if (self.is_valid_move(board, Move(node.x - 2, node.y + 1)) and 
                self.is_valid_move(board, Move(node.x - 1, node.y + 1)) and 
                self.is_valid_move(board, Move(node.x - 1, node.y))):
                two_bridges.append(
                    Move(node.x - 2, node.y + 1),  # bridge position
                 )
    
        return two_bridges

    def get_two_bridges_with_positions(self, board: Board, colour: Colour) -> list[tuple[Move, tuple[Move, Move]]]:
        """Get all possible two-bridge moves with their associated empty cells."""
        two_bridges = []
        current_nodes = self.get_all_positions_for_colour(board, colour)
        
        
    
        for node in current_nodes:
            # Pattern 1: Top-left bridge
            if (self.is_my_stone(board, Move(node.x - 1, node.y - 1)) and 
                self.is_valid_move(board, Move(node.x - 1, node.y)) and 
                self.is_valid_move(board, Move(node.x, node.y - 1))):
                two_bridges.append((
                    (node.x - 1, node.y - 1),  # bridge position
                    (Move(node.x - 1, node.y), Move(node.x, node.y - 1))  # empty cells
                ))
            
            # Pattern 2: Top-right bridge
            if (self.is_my_stone(board, Move(node.x + 1, node.y - 2)) and 
                self.is_valid_move(board, Move(node.x, node.y - 1)) and 
                self.is_valid_move(board, Move(node.x + 1, node.y - 1))):
                two_bridges.append((
                    (node.x + 1, node.y - 2),  # bridge position
                    (Move(node.x, node.y - 1), Move(node.x + 1, node.y - 1))  # empty cells
                ))
    
            # Pattern 3: Right bridge
            if (self.is_my_stone(board, Move(node.x + 2, node.y - 1)) and 
                self.is_valid_move(board, Move(node.x + 1, node.y - 1)) and 
                self.is_valid_move(board, Move(node.x + 1, node.y))):
                two_bridges.append((
                    (node.x + 2, node.y - 1),  # bridge position
                    (Move(node.x + 1, node.y - 1), Move(node.x + 1, node.y))  # empty cells
                ))
    
            # Pattern 4: Bottom-right bridge
            if (self.is_my_stone(board, Move(node.x + 1, node.y + 1)) and 
                self.is_valid_move(board, Move(node.x + 1, node.y)) and 
                self.is_valid_move(board, Move(node.x, node.y + 1))):
                two_bridges.append((
                    (node.x + 1, node.y + 1),  # bridge position
                    (Move(node.x + 1, node.y), Move(node.x, node.y + 1))  # empty cells
                ))
    
            # Pattern 5: Bottom-left bridge
            if (self.is_my_stone(board, Move(node.x - 1, node.y + 2)) and 
                self.is_valid_move(board, Move(node.x, node.y + 1)) and 
                self.is_valid_move(board, Move(node.x - 1, node.y + 1))):
                two_bridges.append((
                    (node.x - 1, node.y + 2),  # bridge position
                    (Move(node.x, node.y + 1), Move(node.x - 1, node.y + 1))  # empty cells
                ))
    
            # Pattern 6: Left bridge
            if (self.is_my_stone(board, Move(node.x - 2, node.y + 1)) and 
                self.is_valid_move(board, Move(node.x - 1, node.y + 1)) and 
                self.is_valid_move(board, Move(node.x - 1, node.y))):
                two_bridges.append((
                    (node.x - 2, node.y + 1),  # bridge position
                    (Move(node.x - 1, node.y + 1), Move(node.x - 1, node.y))  # empty cells
                ))
    
        return two_bridges

    def save_two_bridge(self, board: Board, opp_move: Move) -> Move | None:
        """
        Checks if opponent's last move threatens a two-bridge and returns the move to save it.
        Returns None if no bridge needs saving.
        """
        x, y = opp_move.x, opp_move.y
        
        moves_to_save = set()  # Store moves to save two-bridge
        
        # Store original state
        original_color = board.tiles[x][y].colour
        current_bridges = self.get_two_bridges_with_positions(board, self._colour)
        
        # Temporarily undo opponent's move
        board.set_tile_colour(x, y, None)
        
        # Get bridges before opponent's move
        previous_bridges = self.get_two_bridges_with_positions(board, self._colour)
        
        # Restore board state
        board.set_tile_colour(x, y, original_color)
        
        # Check each bridge that existed before
        for bridge_pos, empty_cells in previous_bridges:
            # If this bridge no longer exists after opponent's move
            if bridge_pos not in [b[0] for b in current_bridges]:
                cell1, cell2 = empty_cells
                # If opponent blocked one empty cell, play in the other
                if (x, y) == (cell1.x, cell1.y) and self.is_valid_move(board, (cell2.x, cell2.y)):
                    moves_to_save.add(Move(cell2.x, cell2.y))
                elif (x, y) == (cell2.x, cell2.y) and self.is_valid_move(board, (cell1.x, cell1.y)):
                    moves_to_save.add(Move(cell1.x, cell1.y))
        
        return list(moves_to_save)