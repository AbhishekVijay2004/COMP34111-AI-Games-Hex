from src.Game import Game
from src.Colour import Colour
from src.Player import Player
from agents.Group100.RaveAgent import MCTSAgent as RaveAgentOld
from agents.Group100.RaveAgent import MCTSAgent as RaveAgent
import random
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
import time
import os

def play_single_game(game_id: int, swap_colors: bool = False) -> tuple[str, str, str, float]:
    """Play a single game between agents and return the result"""
    if swap_colors:
        p1 = Player("Basic", RaveAgentOld(Colour.RED))
        p2 = Player("New", RaveAgent(Colour.BLUE))
        colors = "(RED vs BLUE)"
    else:
        p1 = Player("New", RaveAgent(Colour.RED))
        p2 = Player("Basic", RaveAgentOld(Colour.BLUE))
        colors = "(RED vs BLUE)"
    
    game = Game(p1, p2, logDest=os.devnull, silent=True)  # Suppress output
    start_time = time.time()
    result = game.run()
    elapsed = time.time() - start_time
    
    # Enhanced output with player positions and colors
    winner_info = f"Player {'1' if result['winner'] == p1.name else '2'} ({result['winner']}) {colors}"
    print(f"Game {game_id + 1} completed - Winner: {winner_info} Time: {elapsed:.2f}s")
    
    return (p1.name, p2.name, result["winner"], "P1" if result["winner"] == p1.name else "P2")

def run_tournament(num_games: int = 100):
    """Run a tournament between agents"""
    stats = defaultdict(int)
    position_stats = {"P1": 0, "P2": 0}  # Track wins by position
    total_games = num_games * 2  # Play both as RED and BLUE
    start_time = time.time()
    
    print(f"Starting tournament: {total_games} games ({num_games} per color configuration)")
    print("=" * 50)
    
    with ProcessPoolExecutor() as executor:
        # First half: MCTS as RED, MCTSOpt as BLUE
        first_half = list(executor.map(play_single_game, 
                                     range(num_games), 
                                     [False] * num_games))
        
        # Second half: MCTSOpt as RED, MCTS as BLUE
        second_half = list(executor.map(play_single_game, 
                                      range(num_games, total_games),
                                      [True] * num_games))
    
    # Process results
    for p1, p2, winner, position in first_half + second_half:
        stats[winner] += 1
        position_stats[position] += 1
    
    # Print enhanced results
    total_time = time.time() - start_time
    print("\nTournament Results")
    print("=" * 50)
    print(f"Total Games Played: {total_games}")
    print(f"New Wins: {stats['New']} ({stats['New']/total_games*100:.1f}%)")
    print(f"Basic Wins: {stats['Basic']} ({stats['Basic']/total_games*100:.1f}%)")
    print("\nPosition Analysis:")
    print(f"Player 1 Wins: {position_stats['P1']} ({position_stats['P1']/total_games*100:.1f}%)")
    print(f"Player 2 Wins: {position_stats['P2']} ({position_stats['P2']/total_games*100:.1f}%)")
    print(f"\nTotal Time: {total_time:.1f}s")
    print(f"Average Game Time: {total_time/total_games:.1f}s")

if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    run_tournament(num_games=30)  # 50 games per color configuration = 100 total games