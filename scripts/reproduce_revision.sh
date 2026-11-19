#!/usr/bin/env bash
# Experiments added in the first revision (reverse-osmosis case study, budget and population
# sensitivity) and the extended analysis. Run from the repository root with the package installed:
#     JOBS=24 bash scripts/reproduce_revision.sh
# Included results that can be recomputed separately:
#     python scripts/structure_robustness.py                  # synthetic robustness study
#     python scripts/run_experiment.py --model otto --seeds 10 --gens 400 --jobs $JOBS \
#         --algs MOEA/D-LWS2x,MOEA/D-LWS4x --tag lwscone       # LWS cone-width sensitivity
#     python scripts/run_experiment.py --model otto --seeds 10 --gens 400 --jobs $JOBS \
#         --algs MOEA/D-LWS0.5x,MOEA/D-LWS0.25x --tag lwsnarrow
set -euo pipefail
JOBS=${JOBS:-24}
mkdir -p logs
python -m pytest -q

echo "[0/5] Otto: extra decomposition baselines (PBI, LWS, L2, L4)"
python scripts/run_experiment.py --model otto --seeds 30 --gens 400 --jobs "$JOBS" \
    --algs MOEA/D-PBI,MOEA/D-LWS,MOEA/D-L2,MOEA/D-L4 --tag scal | tee logs/ot_scal.log

echo "[1/5] reverse-osmosis plant: 10 algorithms x 30 runs x 2 formulations"
python scripts/run_experiment.py --model ro --seeds 30 --gens 400 --jobs "$JOBS" \
    --algs NSGA-II,NSGA-III,RVEA,MOEA/D,MOEA/D-AWA,MOEA/D-WS,MOEA/D-PBI,MOEA/D-LWS,MOEA/D-L2,MOEA/D-L4 \
    | tee logs/ro_runs.log

echo "[2/5] Otto: five-times larger budget"
python scripts/run_experiment.py --model otto --seeds 10 --gens 2000 --jobs "$JOBS" \
    --algs MOEA/D-WS,MOEA/D-AWA --tag long | tee logs/ot_long.log

echo "[3/5] Otto: larger population (N = 231)"
python scripts/run_experiment.py --model otto --seeds 10 --gens 400 --partitions 20 --jobs "$JOBS" \
    --algs MOEA/D-WS,MOEA/D-AWA --tag bigpop | tee logs/ot_bigpop.log

echo "[4/5] extended analysis (support tests are single-threaded, ~10 min)"
python scripts/analyze_extended.py --model otto | tee logs/ot_extended.log
python scripts/analyze_extended.py --model ro | tee logs/ro_extended.log

echo "[5/5] paper figures"
python paper/make_figures.py
echo "done: results/ot_extended.md and results/ro_extended.md"
