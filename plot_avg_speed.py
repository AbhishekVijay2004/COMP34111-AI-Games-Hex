import matplotlib.pyplot as plt
import csv

# Read data from a CSV file
groups = []
move_speed = []

with open('combined_stats.csv', mode='r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header row
    for row in reader:
        groups.append(row[0])  # Group name in first column
        move_speed.append(float(row[7]))  # Win rate in 8th column

# Specify a list of colors for each group
colors = ['#FF6347', '#4682B4', '#32CD32', '#FFD700', '#8A2BE2', '#FF4500', '#00FA9A']

# Create the bar chart
plt.figure(figsize=(10, 6))

bars = plt.bar(groups, move_speed, color=colors)

# Add labels and title
plt.xlabel('Agent')
plt.ylabel('Average Move Speed')
plt.title('Agent Average Move Speeds')

# Add value labels to each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom', fontsize=10)

# Show the plot
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
