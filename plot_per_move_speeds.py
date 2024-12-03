import matplotlib.pyplot as plt
import os

# List of log files to process
log_files = ["game3.log"] #, "game2.log", "game3.log", "MCTS.log", "MCTS2.log"]  # Replace with your actual filenames

# Colors for each log file
colors = ['green', 'blue', 'red', 'orange', 'purple', 'brown']
labels = ["MCTS3 (Tuned)"] #, "MCTS3 (2)", "MCTS3Tuned", "MCTS", "MCTS (2)"]

# Plot data for each file
for index, file_path in enumerate(log_files):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue
    
    mcts_data = []

    # Read data from the log file
    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split(",")
            if len(parts) == 5 and "MCTS" in parts[1]:
                try:
                    x = int(parts[0])  # Move No.
                    y = int(parts[4])  # Move Speed (ns)
                    if x <= 90:  # Limit to max_moves
                        mcts_data.append((x, y / 1e9))
                except ValueError:
                    print(f"Invalid data in file {file_path}: {line.strip()}")

    # Check if mcts_data has been populated
    if not mcts_data:
        print(f"No data found for MCTS in file {file_path}.")
        continue

    # Extract x and y values
    x_values = [row[0] for row in mcts_data]
    y_values = [row[1] for row in mcts_data]

    # Plot data for this file
    color = colors[index % len(colors)]  # Cycle through colors if there are more files than colors
    label = labels[index]
    plt.scatter(x_values, y_values, color=color, label=label, marker='.')
    plt.plot(x_values, y_values, color=color, linestyle='-')

# Add labels, title, and legend
plt.xlabel("Move No.")
plt.ylabel("Move Speed (s)")
plt.title("Agent Move Speed Over Time")
plt.legend()

# Show the plot
plt.show()
