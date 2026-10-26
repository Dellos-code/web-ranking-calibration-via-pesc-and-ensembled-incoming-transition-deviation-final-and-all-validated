#!/bin/bash
# Stage 2: replay the Stage 1 instances and run the alpha sweeps (2 worker processes).
cd "$(dirname "$0")"
for p in calib test,topo scale,real; do
  echo "=== panel $p $(date)"
  python3 run_experiments.py stage2 --panel $p --workers 2
done
echo "=== ALL DONE $(date)"
