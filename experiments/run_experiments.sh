#!/usr/bin/env bash
# Embedding-width and BERTopic experiments, run one after another on one machine.
#
#   control   ../reputation_topic_gpu            unchanged code, 384-dim MiniLM-L12
#   emb768    ../reputation_topic_gpu_768        768-dim paraphrase-multilingual-mpnet-base-v2
#   bertopic  ../reputation_topic_gpu_bertopic   BERTopic replaces frozen UMAP + HDBSCAN
#
# Phase 1: the 200k base run for each arm (same flags as `docker compose run base`).
# Phase 2: the 300k stream (stream300/chunk_1..6, window 3), as `docker compose run stream`.
#
# All arms share one interpreter (the bertopic copy's venv, a strict superset of
# the others' locked dependencies), so library versions cannot differ between
# arms. The control embeds cold and fills its own cache; the bertopic arm uses
# the same model on the same text, so it reuses that cache (identical vectors).
#
# Usage: experiments/run_experiments.sh [step ...]   (default: every step, in order)
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$ROOT/data"
PY="$ROOT/reputation_topic_gpu_bertopic/.venv/bin/python"
LOGS="$ROOT/experiments/logs"
mkdir -p "$LOGS"

# AI Router credentials for --label-method llm, read from the prototype's .env
# (never copied into any of the three code directories).
ENV_FILE="${ENV_FILE:-/home/sayan/Desktop/Reputation/reputation_topic_gpu/.env}"
set -a; . "$ENV_FILE"; set +a
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

declare -A DIR=( [control]="$ROOT/reputation_topic_gpu"
                 [emb768]="$ROOT/reputation_topic_gpu_768"
                 [bertopic]="$ROOT/reputation_topic_gpu_bertopic" )
declare -A MODEL=( [control]=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
                   [emb768]=sentence-transformers/paraphrase-multilingual-mpnet-base-v2
                   [bertopic]=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 )
out()   { echo "${DIR[$1]}/out/exp_$1"; }
cache() { case $1 in bertopic) echo "$(out control)/cache/embed";; *) echo "$(out $1)/cache/embed";; esac; }

# Flags from docker-compose.yml `base`, minus the GPU-only ones.
BASE_COMMON=(
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
)
UMAP_FLAGS() {  # $1 = arm
  echo --reducer=umap --umap-components=5 --umap-neighbors=30 --umap-min-dist=0.0 \
       --umap-per-brand --umap-model="$(out $1)/models/umap_v5_perbrand.pkl" \
       --umap-reference-size=50000 --cluster-selection-epsilon=0.2 \
       --split-max-size=2500 --split-max-child-similarity=0.92
}
BERTOPIC_FLAGS="--reducer=bertopic --cluster-selection-epsilon=0 --split-max-size=0"

stamp() {  # machine state at the start of a step, for reading timings later
  { echo "start $(date -Is)"; uptime; free -m | sed -n 2p; } >> "$LOGS/$1.machine"
}

run_step() {  # $1 = step name, rest = command
  local name=$1; shift
  echo "=== [$name] $(date -Is) ==="
  stamp "$name"
  /usr/bin/time -v -o "$LOGS/$name.time" "$@" > "$LOGS/$name.log" 2>&1
  local rc=$?
  echo "end $(date -Is) rc=$rc" >> "$LOGS/$name.machine"
  grep -E 'Elapsed|Maximum resident' "$LOGS/$name.time" | sed 's/^\s*/    /'
  if [ $rc -ne 0 ]; then echo "    FAILED rc=$rc -- tail of $LOGS/$name.log:"; tail -20 "$LOGS/$name.log"; fi
  return $rc
}

base() {  # $1 = arm
  local arm=$1 o; o=$(out $arm); mkdir -p "$o/models" "$o/cache"
  rm -rf "$o/base_200k" "$o/buf_base"
  local extra; if [ "$arm" = bertopic ]; then extra=$BERTOPIC_FLAGS; else extra=$(UMAP_FLAGS $arm); fi
  cd "${DIR[$arm]}" && run_step "base_$arm" "$PY" reputation_topic_detection.py \
    "$DATA/twcs_subset_200k.csv" --out="$o/base_200k" --model="${MODEL[$arm]}" \
    "${BASE_COMMON[@]}" $extra --buffer="$o/buf_base" --embed-cache="$(cache $arm)"
}

stream() {  # $1 = arm
  local arm=$1 o; o=$(out $arm)
  rm -rf "$o"/stream_[0-9]* "$o/buf_stream"
  local model=(); [ "$arm" = bertopic ] || model=(--model="$o/models/umap_v5_perbrand.pkl")
  cd "${DIR[$arm]}" && run_step "stream_$arm" "$PY" run_stream.py \
    --base-run="$o/base_200k" "${model[@]}" \
    --chunks "$DATA"/stream300/chunk_{1..6}.csv --window=3 \
    --out-prefix="$o/stream" --buffer="$o/buf_stream" \
    --device=cpu --embed-cache="$(cache $arm)"
}

STEPS=("$@")
[ ${#STEPS[@]} -eq 0 ] && STEPS=(base:control base:emb768 base:bertopic
                                 stream:control stream:emb768 stream:bertopic)
{ echo "host $(hostname) | $(nproc) cpus | $(lscpu | sed -n 's/^Model name:\s*//p')";
  free -g | sed -n 2p; "$PY" -c 'import torch,umap,sklearn,bertopic;print("torch",torch.__version__,"threads",torch.get_num_threads(),"umap",umap.__version__,"sklearn",sklearn.__version__,"bertopic",bertopic.__version__)'; } > "$LOGS/machine.txt"

for s in "${STEPS[@]}"; do
  "${s%%:*}" "${s##*:}" || echo "    (continuing with the next step)"
done
echo "=== ALL DONE $(date -Is) ==="
