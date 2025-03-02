import subprocess
import os
import random
import re
from datetime import datetime

def run_game(red_team, blue_team, layout):
    """Run a single Pacman game and return the result."""
    cmd = [
        "python3", "-m", "pacai.bin.capture",
        "--red", red_team,
        "--blue", blue_team,
        "--layout", layout
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        output = result.stdout + result.stderr
        
        # Parse the output to get the winner and score
        if "The Red team wins" in output:
            winner = "Red"
            score_match = re.search(r"The Red team wins by (\d+) points", output)
            score = int(score_match.group(1)) if score_match else 0
        elif "The Blue team wins" in output:
            winner = "Blue"
            score_match = re.search(r"The Blue team wins by (\d+) points", output)
            score = int(score_match.group(1)) if score_match else 0
        elif "Tie game!" in output:
            winner = "Tie"
            score = 0
        else:
            winner = "Unknown"
            score = 0
            
        return winner, score
    except Exception as e:
        print(f"Error running game on layout {layout}: {str(e)}")
        return "Error", 0

def run_tests(count=10, output_file="test_results.txt"):
    """Run tests on random layouts."""
    my_team = "pacai.student.myTeam"
    baseline_team = "pacai.core.baselineTeam"
    
    results = []
    
    # Generate random layout seeds
    seeds = list(range(1, count + 1))
    
    total_wins = 0
    total_games = count * 2  # We play each layout twice (as red and blue)
    
    print(f"Starting tests on {count} layouts...")
    
    with open(output_file, 'w') as f:
        f.write(f"Test Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n\n")
        
        for i, seed in enumerate(seeds):
            layout = f"RANDOM{seed}"
            
            # Play as red
            print(f"[{i*2+1}/{total_games}] Testing layout {layout} - myTeam(RED) vs baseline(BLUE)")
            winner, score = run_game(my_team, baseline_team, layout)
            my_team_won = (winner == "Red")
            
            if my_team_won:
                total_wins += 1
                
            f.write(f"Layout: {layout}, myTeam as RED: ")
            if winner == "Red":
                f.write(f"WIN +{score}\n")
            elif winner == "Blue":
                f.write(f"LOSS -{score}\n")
            elif winner == "Tie":
                f.write("TIE\n")
            else:
                f.write(f"ERROR: {winner}\n")
            
            # Play as blue
            print(f"[{i*2+2}/{total_games}] Testing layout {layout} - baseline(RED) vs myTeam(BLUE)")
            winner, score = run_game(baseline_team, my_team, layout)
            my_team_won = (winner == "Blue")
            
            if my_team_won:
                total_wins += 1
                
            f.write(f"Layout: {layout}, myTeam as BLUE: ")
            if winner == "Blue":
                f.write(f"WIN +{score}\n")
            elif winner == "Red":
                f.write(f"LOSS -{score}\n")
            elif winner == "Tie":
                f.write("TIE\n")
            else:
                f.write(f"ERROR: {winner}\n")
            
            f.write("\n")
        
        # Write summary
        win_percent = (total_wins / total_games) * 100 if total_games > 0 else 0
        f.write("=" * 50 + "\n\n")
        f.write("SUMMARY:\n")
        f.write(f"Total games: {total_games}\n")
        f.write(f"Wins: {total_wins}\n")
        f.write(f"Win rate: {win_percent:.2f}%\n")
    
    print(f"\nTesting complete! Results saved to {output_file}")
    print(f"MyTeam won {total_wins} out of {total_games} games ({win_percent:.2f}%)")

if __name__ == "__main__":
    import sys
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    output_file = sys.argv[2] if len(sys.argv) > 2 else "test_results.txt"
    run_tests(count, output_file)