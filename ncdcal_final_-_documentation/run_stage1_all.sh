#!/bin/bash
# Stage 1: run every panel in sequence (each panel uses 2 worker processes).
cd "$(dirname "$0")"
for p in calib test topo scale real; do
  echo "=== panel $p $(date)" 
  python3 run_experiments.py stage1 --panel $p --workers 2
done
echo "=== ALL DONE $(date)"
