#!/bin/bash
# Segment-parallel AV1 -> H.264 transcode (cv2 wheel cannot decode AV1).
# Usage: transcode_parallel.sh IN OUT NSEG
set -e
IN=$1; OUT=$2; N=$3
DUR=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$IN")
DUR=${DUR%.*}; DUR=$((DUR + 1))
STEP=$(( (DUR + N - 1) / N ))
TMP=/tmp/transcode_$(basename "$OUT" .mp4)
mkdir -p "$TMP"
: > "$TMP/list.txt"
pids=()
for i in $(seq 0 $((N-1))); do
  S=$((i * STEP))
  ffmpeg -hide_banner -loglevel error -ss "$S" -t "$STEP" -i "$IN" \
    -an -c:v libx264 -preset veryfast -crf 19 -threads 3 \
    -y "$TMP/seg$(printf %03d $i).mp4" &
  pids+=($!)
  echo "file '$TMP/seg$(printf %03d $i).mp4'" >> "$TMP/list.txt"
done
for p in "${pids[@]}"; do wait "$p"; done
ffmpeg -hide_banner -loglevel error -f concat -safe 0 -i "$TMP/list.txt" -c copy -y "$OUT"
rm -rf "$TMP"
echo TRANSCODE_OK "$OUT"
