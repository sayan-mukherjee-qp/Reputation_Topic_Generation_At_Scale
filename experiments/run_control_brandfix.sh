#!/usr/bin/env bash
# Control arm re-run after the stabilize_topics() brand fix (IDs never cross brands).
#
# Same code path, flags and interpreter as `run_experiments.sh base:control stream:control`,
# but into its own directory so exp_control stays as the comparator:
#
#   out/exp_control_brandfix/   fresh base_200k, stream_1..6, buf_base, buf_stream
#
# Reused on purpose, so the brand fix is the only clustering-side difference:
#   - exp_control's embedding cache (same model, same text: identical vectors)
#   - exp_control's UMAP model (50k reference, fitted on the 200k corpus), copied in
# Never reused: rolling buffers (a stale buffer leaked future records in gpu_run_2
# and the Laya run).
#
# Usage: experiments/run_control_brandfix.sh [base] [stream]   (default: both)
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$ROOT/data"
PY="$ROOT/reputation_topic_gpu_bertopic/.venv/bin/python"
CODE="$ROOT/reputation_topic_gpu"
SRC="$CODE/out/exp_control"
O="$CODE/out/exp_control_brandfix"
CACHE="$SRC/cache/embed"
UMAP="$O/models/umap_v5_perbrand.pkl"
LOGS="$ROOT/experiments/logs_brandfix"
mkdir -p "$LOGS" "$O/models"

ENV_FILE="${ENV_FILE:-/home/sayan/Desktop/Reputation/reputation_topic_gpu/.env}"
set -a; . "$ENV_FILE"; set +a
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

if [ ! -f "$UMAP" ]; then
  cp "$SRC/models/umap_v5_perbrand.pkl" "$UMAP"
  [ -f "$SRC/models/umap_v5_perbrand.json" ] && cp "$SRC/models/umap_v5_perbrand.json" "$O/models/"
fi

BASE_FLAGS=(
  --device=cpu
  --min-similarity=0.50 --min-cluster-size=30 --min-samples=10
  --brand-scoped-assignment --merge-duplicate-topics --duplicate-similarity=0.92
  --flag-junk-topics --junk-coherence-veto=0.10 --junk-coherence-min-size=30
  --micro-clusters --tweets-per-topic=-1
  --label-method=llm --label-llm-samples=10 --label-llm-workers=12
  --candidate-margin=0.05
  --alert-min-coherence=0.05 --alert-min-size=250 --residual-ack-ratio=0.40
  --event-recall --recover-unassigned --recover-match-existing
  --max-live-topics=600 --consolidate-min-similarity=0.75
  --reducer=umap --umap-components=5 --umap-neighbors=30 --umap-min-dist=0.0
  --umap-per-brand --umap-model="$UMAP" --umap-reference-size=50000
  --cluster-selection-epsilon=0.2 --split-max-size=2500 --split-max-child-similarity=0.92
)

run_step() {  # $1 = step name, rest = command
  local name=$1; shift
  echo "=== [$name] $(date -Is) ==="
  { echo "start $(date -Is)"; uptime; free -m | sed -n 2p; } >> "$LOGS/$name.machine"
  /usr/bin/time -v -o "$LOGS/$name.time" "$@" > "$LOGS/$name.log" 2>&1
  local rc=$?
  echo "end $(date -Is) rc=$rc" >> "$LOGS/$name.machine"
  grep -E 'Elapsed|Maximum resident' "$LOGS/$name.time" | sed 's/^\s*/    /'
  if [ $rc -ne 0 ]; then echo "    FAILED rc=$rc -- tail of $LOGS/$name.log:"; tail -20 "$LOGS/$name.log"; fi
  return $rc
}

base() {
  rm -rf "$O/base_200k" "$O/buf_base"
  cd "$CODE" && run_step base_control_brandfix "$PY" reputation_topic_detection.py \
    "$DATA/twcs_subset_200k.csv" --out="$O/base_200k" \
    --model=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
    "${BASE_FLAGS[@]}" --buffer="$O/buf_base" --embed-cache="$CACHE"
}

stream() {
  rm -rf "$O"/stream_[0-9]* "$O/buf_stream"
  cd "$CODE" && run_step stream_control_brandfix "$PY" run_stream.py \
    --base-run="$O/base_200k" --model="$UMAP" \
    --chunks "$DATA"/stream300/chunk_{1..6}.csv --window=3 \
    --out-prefix="$O/stream" --buffer="$O/buf_stream" \
    --device=cpu --embed-cache="$CACHE"
}

STEPS=("$@"); [ ${#STEPS[@]} -eq 0 ] && STEPS=(base stream)
for s in "${STEPS[@]}"; do
  "$s" || { echo "    $s failed; stopping (the stream needs a good base)"; exit 1; }
done
echo "=== ALL DONE $(date -Is) ==="
