#!/usr/bin/env bash
# Headless PPO training using all CPU cores.  Usage: ./run_training.sh [timesteps]   (default 30M)
N=$(python -c "import os; print(max(1, (os.cpu_count() or 2) - 1))")
python main.py --headless --train --n-envs "$N" --timesteps "${1:-30000000}"
