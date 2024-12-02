import matplotlib.pyplot as plt

# Function to read data from the text file
def read_data(filename):
    turns = []
    times = []
    with open(filename, 'r') as f:
        for line in f:
            turn, time = line.strip().split(',')
            turns.append(int(turn))  # Convert turn to integer
            times.append(float(time))  # Convert time to float
    return turns, times

# Read data from both files
turns1, times1 = read_data('a_star_speed.log')  # Replace with your actual file names
turns2, times2 = read_data('bfs_speed2.log')
turns3, times3 = read_data('bfs_speed.log')

# Calculate average time per move for both files
average_time1 = sum(times1) / len(times1) if times1 else 0
average_time2 = sum(times2) / len(times2) if times2 else 0
average_time3 = sum(times3) / len(times3) if times3 else 0

print(f"Average time per move for A*: {average_time1:.4f} seconds")
print(f"Average time per move for BFS2: {average_time2:.4f} seconds")
print(f"Average time per move for BFS: {average_time3:.4f} seconds")

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(turns1, times1, label='A*', color='blue', marker='o')
plt.plot(turns2, times2, label='BFS2', color='orange', marker='.')
plt.plot(turns3, times3, label='BFS', color='red', marker='x')

# Labels and title
plt.xlabel('Turn')
plt.ylabel('Time')
plt.title('_trace_path() Speed')

# Show legend
plt.legend()

# Show the plot
plt.grid(True)
plt.show()
