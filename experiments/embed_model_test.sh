#!/usr/bin/env bash
# Which 768-dim encoder should the laya copy use? Each arm runs the laya copy's
# cold-start base pipeline (the normal pipeline) of reputation_topic_gpu_gpt on the 20k slice with LLM
# labels, flags as in its docker-compose `base`, only --model differing.
#
#   minilm  paraphrase-multilingual-MiniLM-L12-v2  384-dim reference (current default)
#   mpnet   paraphrase-multilingual-mpnet-base-v2  768
#   e5      intfloat/multilingual-e5-base          768, prompt "query: "
#   nomic   nomic-ai/nomic-embed-text-v2-moe       768, prompt "clustering: ", remote code
#
# Usage: experiments/embed_model_test.sh [arm ...]   (NOMIC_PY=<python with einops> for nomic)
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP="$ROOT/reputation_topic_gpu_gpt"
PY="${PY:-$ROOT/reputation_topic_gpu/.venv/bin/python}"
OUT="$APP/out/model_eval"
LOGS="$ROOT/experiments/logs_models"
mkdir -p "$OUT" "$LOGS"
set -a; . "${ENV_FILE:-/home/sayan/Desktop/Reputation/reputation_topic_gpu/.env}"; set +a
export TOKENIZERS_PARALLELISM=false

BASE=(
  --device=cpu --min-similarity=0.50 --min-cluster-size=30 --min-samples=10
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

arm() {
  local a=$1 model cache py=$PY
  case $a in
    minilm) model=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
            cache=$ROOT/reputation_topic_gpu/out/slice_control/cache ;;
    mpnet)  model=sentence-transformers/paraphrase-multilingual-mpnet-base-v2
            cache=$ROOT/reputation_topic_gpu_768/out/slice_cache ;;
    e5)     model=intfloat/multilingual-e5-base; cache=$OUT/cache ;;
    nomic)  model=nomic-ai/nomic-embed-text-v2-moe; cache=$OUT/cache; py=${NOMIC_PY:?set NOMIC_PY} ;;
  esac
  local o="$OUT/$a"; rm -rf "$o"; mkdir -p "$o/models"
  echo "=== [$a] $(date -Is) ==="
  ( cd "$APP" && /usr/bin/time -v -o "$LOGS/$a.time" "$py" reputation_topic_detection.py \
      "$ROOT/data/twcs_subset_20k.csv" --out="$o/base_20k" --model="$model" "${BASE[@]}" \
      --umap-model="$o/models/umap.pkl" --buffer="$o/buf" --embed-cache="$cache" \
      > "$LOGS/$a.log" 2>&1 )
  local rc=$?
  grep -E 'Elapsed' "$LOGS/$a.time" | sed 's/^\s*/    /'
  [ $rc -eq 0 ] || { echo "    FAILED rc=$rc"; tail -12 "$LOGS/$a.log"; }
}

ARMS=("$@"); [ ${#ARMS[@]} -eq 0 ] && ARMS=(minilm mpnet e5)
for a in "${ARMS[@]}"; do arm "$a"; done
echo "=== ALL DONE $(date -Is) ==="
