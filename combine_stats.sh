#!/bin/bash

# Output file
OUTPUT="combined_summary.csv"

# Initialize the output with the header
echo "player,matches,wins,win_rate,total_move_time,total_moves,average_move_time,illegal_moves_loss,time_out_loss,regular_loss" > $OUTPUT

# Create associative arrays to store cumulative data
declare -A matches wins total_move_time total_moves illegal_moves_loss time_out_loss regular_loss player_count

# Read each file and process data
for file in *.csv; do
  echo "Processing $file..."
  tail -n +2 "$file" | while IFS=',' read -r player matches_val wins_val win_rate_val total_move_time_val total_moves_val average_move_time_val illegal_moves_loss_val time_out_loss_val regular_loss_val; do
    # Accumulate data for the player
    matches[$player]=$((matches[$player] + matches_val))
    wins[$player]=$((wins[$player] + wins_val))
    total_move_time[$player]=$(echo "${total_move_time[$player]:-0} + $total_move_time_val" | bc)
    total_moves[$player]=$((total_moves[$player] + total_moves_val))
    illegal_moves_loss[$player]=$((illegal_moves_loss[$player] + illegal_moves_loss_val))
    time_out_loss[$player]=$((time_out_loss[$player] + time_out_loss_val))
    regular_loss[$player]=$((regular_loss[$player] + regular_loss_val))
    player_count[$player]=$((player_count[$player] + 1))
  done
done

# Calculate aggregated values and write to the output
for player in "${!matches[@]}"; do
  total_matches=${matches[$player]}
  total_wins=${wins[$player]}
  avg_win_rate=$(echo "$total_wins / $total_matches" | bc -l)
  avg_move_time=$(echo "${total_move_time[$player]} / ${player_count[$player]}" | bc -l)
  avg_illegal_moves=$(echo "${total_move_time[$player]} / ${total_moves[$player]}" | bc -l)
  total_illegal_moves=${illegal_moves_loss[$player]}
  total_timeouts=${time_out_loss[$player]}
  total_regular_loss=${regular_loss[$player]}

  echo "$player,$total_matches,$total_wins,$avg_win_rate,${total_move_time[$player]},${total_moves[$player]},$avg_move_time,$total_illegal_moves,$total_timeouts,$total_regular_loss" >> $OUTPUT
done

echo "Data combined into $OUTPUT."
