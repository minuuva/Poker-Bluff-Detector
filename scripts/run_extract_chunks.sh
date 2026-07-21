#!/bin/bash
# Chunked extract-state across cores. Usage: run_extract_chunks.sh VIDEO STEM NCHUNKS
# Each chunk writes its own JSONL with a checkpoint sidecar, so rerunning this
# script after a spot reclaim resumes every chunk where it stopped.
set -e
cd ~/pokertell
VIDEO=$1; STEM=$2; N=$3
UV=~/.local/bin/uv
DUR=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$VIDEO")
DUR=${DUR%.*}
STEP=$(( (DUR + N - 1) / N ))
mkdir -p logs

# Warm-up: one tiny serial run so 16 parallel workers never race on the
# first-use PaddleOCR model download.
OMP_NUM_THREADS=2 $UV run pokertell extract-state "$VIDEO" --t-start 0 --t-end 2 \
  --out /tmp/warmup.snapshots.jsonl > logs/warmup_${STEM}.log 2>&1
rm -f /tmp/warmup.snapshots.jsonl /tmp/warmup.snapshots.jsonl.progress

for i in $(seq 0 $((N-1))); do
  S=$((i * STEP)); E=$(( (i+1) * STEP )); [ "$E" -gt "$DUR" ] && E=$DUR
  OUT=data/hands/${STEM}.chunk$(printf %02d "$i").snapshots.jsonl
  OMP_NUM_THREADS=2 $UV run pokertell extract-state "$VIDEO" \
    --t-start "$S" --t-end "$E" --out "$OUT" \
    > "logs/extract_${STEM}_$(printf %02d "$i").log" 2>&1 &
done
wait
echo ALL_CHUNKS_DONE
