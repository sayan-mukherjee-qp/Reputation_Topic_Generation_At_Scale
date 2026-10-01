#!/usr/bin/env bash
# Slice test of the tuned 768-dim copy: 20k base run + 30k stream (6 chunks,
# window 3), made by make_slice.py from the same 200k / 300k data.
#
#   control      reputation_topic_gpu      MiniLM-384, original thresholds
#   e768_old     reputation_topic_gpu_768  mpnet-768, MiniLM-tuned thresholds
#   e768_cal     reputation_topic_gpu_768  mpnet-768, calibrated thresholds
#   e768_margin  reputation_topic_gpu_768  calibrated + --secondary-margin 0.03
#   e768_floors  reputation_topic_gpu_768  only the assignment floors calibrated (0.53)
#
# All four carry the same-brand ID inheritance fix. The three 768 arms share
# one embedding cache, so the encoder runs once per text.
#
# Usage: experiments/slice_test.sh [arm ...]   (default: all four, in order)
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$ROOT/data"
LOGS="$ROOT/experiments/logs_slice"
mkdir -p "$LOGS"
ENV_FILE="${ENV_FILE:-/home/sayan/Desktop/Reputation/reputation_topic_gpu/.env}"
set -a; . "$ENV_FILE"; set +a
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

CTL="$ROOT/reputation_topic_gpu"
E768="$ROOT/reputation_topic_gpu_768"

# Shared base flags: docker-compose `base`, minus GPU-only and similarity flags.
BASE=(
  --device=cpu --min-cluster-size=30 --min-samples=10
  --brand-scoped-assignment --merge-duplicate-topics
  --flag-junk-topics --junk-coherence-veto=0.10 --junk-coherence-min-size=30
  --micro-clusters --tweets-per-topic=-1
  --label-method=llm --label-llm-samples=10 --label-llm-workers=12
  --reducer=umap --umap-components=5 --umap-neighbors=30 --umap-min-dist=0.0
  --umap-per-brand --umap-reference-size=50000
  --cluster-selection-epsilon=0.2 --candidate-margin=0.05 --split-max-size=2500
  --alert-min-coherence=0.05 --alert-min-size=250 --residual-ack-ratio=0.40
  --event-recall --recover-unassigned --recover-match-existing --max-live-topics=600
)
THR_384=(--min-similarity=0.50 --candidate-similarity=0.55 --duplicate-similarity=0.92
         --split-max-child-similarity=0.92 --label-merge-similarity=0.70
         --label-reuse-similarity=0.92 --consolidate-min-similarity=0.75)
THR_FLOORS=(--min-similarity=0.53 --history-min-similarity=0.53 --candidate-similarity=0.55
            --duplicate-similarity=0.92 --split-max-child-similarity=0.92
            --label-merge-similarity=0.70 --label-reuse-similarity=0.92
            --consolidate-min-similarity=0.75)
THR_CAL=(--min-similarity=0.53 --history-min-similarity=0.53 --candidate-similarity=0.60
         --duplicate-similarity=0.93 --split-max-child-similarity=0.93
         --label-merge-similarity=0.75 --label-reuse-similarity=0.93
         --consolidate-min-similarity=0.80)

step() {  # name, dir, command...
  local name=$1 dir=$2; shift 2
  echo "=== [$name] $(date -Is) ==="
  { echo "start $(date -Is)"; uptime; } > "$LOGS/$name.machine"
  ( cd "$dir" && /usr/bin/time -v -o "$LOGS/$name.time" "$@" > "$LOGS/$name.log" 2>&1 )
  local rc=$?
  echo "end $(date -Is) rc=$rc" >> "$LOGS/$name.machine"
  grep -E 'Elapsed|Maximum resident' "$LOGS/$name.time" | sed 's/^\s*/    /'
  [ $rc -eq 0 ] || { echo "    FAILED rc=$rc"; tail -15 "$LOGS/$name.log"; }
  return $rc
}

arm() {  # arm name -> base 20k then stream 30k
  local a=$1 dir py model o cache thr margin streamthr
  case $a in
    control)     dir=$CTL;  py=$CTL/.venv/bin/python; model=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
                 thr=("${THR_384[@]}"); margin=0; cache=$CTL/out/slice_control/cache ;;
    e768_old)    dir=$E768; thr=("${THR_384[@]}"); margin=0;    streamthr=original ;;
    e768_cal)    dir=$E768; thr=("${THR_CAL[@]}"); margin=0;    streamthr=calibrated ;;
    e768_margin) dir=$E768; thr=("${THR_CAL[@]}"); margin=0.03; streamthr=calibrated ;;
    e768_floors) dir=$E768; thr=("${THR_FLOORS[@]}"); margin=0; streamthr=floors ;;
  esac
  if [ "$dir" = "$E768" ]; then
    py=$ROOT/reputation_topic_gpu_bertopic/.venv/bin/python
    model=sentence-transformers/paraphrase-multilingual-mpnet-base-v2
    cache=$E768/out/slice_cache
  fi
  o=$dir/out/slice_$a
  rm -rf "$o"; mkdir -p "$o/models"
  step "base_$a" "$dir" "$py" reputation_topic_detection.py "$DATA/twcs_subset_20k.csv" \
    --out="$o/base_20k" --model="$model" "${BASE[@]}" "${thr[@]}" \
    --secondary-margin="$margin" \
    --umap-model="$o/models/umap_perbrand.pkl" --buffer="$o/buf_base" --embed-cache="$cache" || return 1
  local sargs=(--base-run="$o/base_20k" --model="$o/models/umap_perbrand.pkl"
               --chunks "$DATA"/stream30/chunk_{1..6}.csv --window=3
               --out-prefix="$o/stream" --buffer="$o/buf_stream"
               --device=cpu --embed-cache="$cache")
  [ "$dir" = "$E768" ] && sargs+=(--thresholds="$streamthr" --secondary-margin="$margin")
  step "stream_$a" "$dir" "$py" run_stream.py "${sargs[@]}"
}

ARMS=("$@"); [ ${#ARMS[@]} -eq 0 ] && ARMS=(control e768_old e768_cal e768_margin)
for a in "${ARMS[@]}"; do arm "$a" || echo "    ($a failed; continuing)"; done
echo "=== ALL DONE $(date -Is) ==="
