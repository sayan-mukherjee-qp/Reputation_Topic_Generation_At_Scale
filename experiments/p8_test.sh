#!/usr/bin/env bash
# P8 near-duplicate experiments on the 20k / 30k slice, one idea per arm.
#
#   baseline     no P8 flags (current behaviour)
#   wordset      idea 1: label merge on the same content words
#   semantic     idea 1: label merge on label-embedding similarity >= 0.90
#   overlap      idea 2: merge when >= 50% of the smaller topic's records are shared
#   review       idea 3: LLM reviews close pairs and merges the ones it calls the same
#   disamb       idea 4: LLM relabels colliding pairs apart (no merging)
#   groups       idea 5: parent groups, complete linkage >= 0.75 (no merging)
#   aliases      idea 6: overlap merge + alias IDs (compare with `overlap`)
#   combo        review + disambiguation + aliases, lifetime cap 3 absorbed IDs
#   combo_cheap  overlap + semantic label merge + disambiguation + aliases, cap 3
#
# Every arm reuses ONE frozen UMAP artifact (copied from the slice control run)
# and the cached MiniLM embeddings, so discovery is identical across arms and
# only the idea under test differs.
#
# Usage: experiments/p8_test.sh [arm ...]
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP="$ROOT/reputation_topic_gpu"
PY="$APP/.venv/bin/python"
DATA="$ROOT/data"
LOGS="$ROOT/experiments/logs_p8"
OUT="$APP/out/p8"
UMAP_SRC="$APP/out/slice_control/models/umap_perbrand"
CACHE="$APP/out/slice_control/cache"
mkdir -p "$LOGS" "$OUT"
ENV_FILE="${ENV_FILE:-/home/sayan/Desktop/Reputation/reputation_topic_gpu/.env}"
set -a; . "$ENV_FILE"; set +a
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

BASE=(
  --device=cpu --model=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
  --min-similarity=0.50 --min-cluster-size=30 --min-samples=10
  --brand-scoped-assignment --merge-duplicate-topics --duplicate-similarity=0.92
  --flag-junk-topics --junk-coherence-veto=0.10 --junk-coherence-min-size=30
  --micro-clusters --tweets-per-topic=-1
  --label-method=llm --label-llm-samples=10 --label-llm-workers=12
  --reducer=umap --umap-components=5 --umap-neighbors=30 --umap-min-dist=0.0
  --umap-per-brand --umap-reference-size=50000
  --cluster-selection-epsilon=0.2 --candidate-margin=0.05
  --split-max-size=2500 --split-max-child-similarity=0.92
  --alert-min-coherence=0.05 --alert-min-size=250 --residual-ack-ratio=0.40
  --event-recall --recover-unassigned --recover-match-existing
  --max-live-topics=600 --consolidate-min-similarity=0.75
)

flags() {
  case $1 in
    baseline) echo "" ;;
    wordset)  echo "--label-merge-mode wordset" ;;
    semantic) echo "--label-merge-mode semantic --label-merge-text-similarity 0.90" ;;
    overlap)  echo "--overlap-merge-ratio 0.5 --overlap-merge-similarity 0.70" ;;
    review)   echo "--llm-merge-review" ;;
    disamb)   echo "--disambiguate-labels" ;;
    groups)   echo "--topic-groups --group-min-similarity 0.75" ;;
    aliases)  echo "--overlap-merge-ratio 0.5 --overlap-merge-similarity 0.70 --keep-aliases" ;;
    combo)       echo "--llm-merge-review --disambiguate-labels --keep-aliases --merge-max-aliases 3" ;;
    combo_cheap) echo "--overlap-merge-ratio 0.5 --overlap-merge-similarity 0.70 --label-merge-mode semantic --disambiguate-labels --keep-aliases --merge-max-aliases 3" ;;
  esac
}

step() {
  local name=$1; shift
  echo "=== [$name] $(date -Is) ==="
  ( cd "$APP" && /usr/bin/time -v -o "$LOGS/$name.time" "$@" > "$LOGS/$name.log" 2>&1 )
  local rc=$?
  grep -E 'Elapsed' "$LOGS/$name.time" | sed 's/^\s*/    /'
  [ $rc -eq 0 ] || { echo "    FAILED rc=$rc"; tail -15 "$LOGS/$name.log"; }
  return $rc
}

arm() {
  local a=$1 o="$OUT/$1" f; f=$(flags "$a")
  rm -rf "$o"; mkdir -p "$o/models"
  cp "$UMAP_SRC.pkl" "$o/models/umap.pkl"; cp "$UMAP_SRC.json" "$o/models/umap.json"
  # shellcheck disable=SC2086
  step "base_$a" "$PY" reputation_topic_detection.py "$DATA/twcs_subset_20k.csv" \
    --out="$o/base_20k" "${BASE[@]}" $f \
    --umap-model="$o/models/umap.pkl" --buffer="$o/buf_base" --embed-cache="$CACHE" || return 1
  step "stream_$a" "$PY" run_stream.py --base-run="$o/base_20k" --model="$o/models/umap.pkl" \
    --chunks "$DATA"/stream30/chunk_{1..6}.csv --window=3 \
    --out-prefix="$o/stream" --buffer="$o/buf_stream" \
    --device=cpu --embed-cache="$CACHE" --extra-args="$f"
}

ARMS=("$@"); [ ${#ARMS[@]} -eq 0 ] && ARMS=(baseline wordset semantic overlap review disamb groups aliases)
for a in "${ARMS[@]}"; do arm "$a" || echo "    ($a failed; continuing)"; done
echo "=== ALL DONE $(date -Is) ==="
