"""Parse leave-one-subject-out fold results and print the aggregated table.

Kept as the named entry point for the evaluate.sh workflow. Reads the JSON-line
result files written by main.py --out and prints the seed-averaged, per
input-type metrics. Delegates to aggregate.py so there is a single implementation.

Usage:
  python parse_logs.py results/power-spectral-difference.jsonl [more.jsonl ...]
"""
import aggregate

if __name__ == "__main__":
    aggregate.main()
