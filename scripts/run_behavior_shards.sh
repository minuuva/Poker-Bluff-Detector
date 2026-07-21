#!/bin/bash
# Sharded extract-behavior across cores. Usage: run_behavior_shards.sh VIDEO HANDS_JSONL SEATS_YAML NSHARDS
# Hands split round-robin into shard files; each worker appends to its own
# CSV with per-decision resume, so rerunning after a kill continues.
set -e
cd ~/pokertell
VIDEO=$1; HANDS=$2; SEATS=$3; N=$4
UV=~/.local/bin/uv
STEM=$(basename "$HANDS" .hands.jsonl)
mkdir -p logs

# Warm-up: serial MediaPipe model download before workers race for it.
$UV run python -c "
from pokertell.behavior.models import ensure_model
ensure_model('face_landmarker.task'); ensure_model('pose_landmarker_lite.task')
print('models ready')" > logs/warmup_behavior.log 2>&1

N=$N STEM=$STEM HANDS=$HANDS $UV run python -c "
import os
n = int(os.environ['N']); stem = os.environ['STEM']
lines = [l for l in open(os.environ['HANDS']) if l.strip()]
shards = [[] for _ in range(n)]
for i, l in enumerate(lines):
    shards[i % n].append(l)
for i, s in enumerate(shards):
    open(f'data/hands/{stem}_bs{i:02d}.hands.jsonl', 'w').writelines(s)
print(f'{len(lines)} hands into {n} shards')"

for i in $(seq 0 $((N-1))); do
  SHARD=$(printf %02d "$i")
  OMP_NUM_THREADS=2 $UV run pokertell extract-behavior "$VIDEO" \
    "data/hands/${STEM}_bs${SHARD}.hands.jsonl" "$SEATS" \
    > "logs/behavior_${STEM}_${SHARD}.log" 2>&1 &
done
wait
echo ALL_SHARDS_DONE
