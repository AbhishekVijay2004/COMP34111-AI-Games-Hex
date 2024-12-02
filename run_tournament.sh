#!/bin/bash

# Path to your program
PROGRAM="python ./HexTournament.py"

# Number of times to run the program
RUNS=9

# Loop to run the program 10 times
for ((i=1; i<=RUNS; i++))
do
  echo "Running the program: Attempt $i of $RUNS"
  $PROGRAM
  if [ $? -ne 0 ]; then
    echo "Program failed during attempt $i. Exiting."
    exit 1
  fi
  echo "Attempt $i completed successfully."
done

echo "All $RUNS runs completed successfully."
