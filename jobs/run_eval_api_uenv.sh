#!/usr/bin/env bash
#SBATCH --job-name=accuracy-sched
#SBATCH --partition=normal
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --mem=800G
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --account=a0179

set -Eeuo pipefail

srun \
  --uenv=prgenv-gnu/25.6:v2 \
  --view=default \
  bash -c '
set -Eeuo pipefail

echo "$(date --iso-8601=seconds) \
host=$(hostname) \
job=${SLURM_JOB_ID:-unset} \
array_job=${SLURM_ARRAY_JOB_ID:-unset} \
array_task=${SLURM_ARRAY_TASK_ID:-unset} \
step=${SLURM_STEP_ID:-unset} \
cuda=${CUDA_VISIBLE_DEVICES:-unset}"

nvidia-smi

REPO=/capstor/scratch/cscs/ftrindad/ParetoCompass
OC_DIR="${REPO}/third_party/opencompass"

OC_VENV="${REPO}/.venv_opencompass"
VLLM_VENV="${REPO}/.venv_vllm"

if [[ ! -d "${VLLM_VENV}" ]]; then
  VLLM_VENV="${OC_VENV}"
fi

if [[ ! -x "${VLLM_VENV}/bin/python" ]]; then
  echo "Erro: Python do venv do vLLM não encontrado: ${VLLM_VENV}/bin/python" >&2
  exit 2
fi

if [[ ! -x "${VLLM_VENV}/bin/vllm" ]]; then
  echo "Erro: executável do vLLM não encontrado: ${VLLM_VENV}/bin/vllm" >&2
  exit 2
fi

if [[ ! -x "${OC_VENV}/bin/opencompass" ]]; then
  echo "Erro: executável do OpenCompass não encontrado: ${OC_VENV}/bin/opencompass" >&2
  exit 2
fi

export PYTHONPATH="${PYTHONPATH:-}:${REPO}:${OC_DIR}"
export COMPASS_DATA_CACHE="${OC_DIR}"

# Mantém os modelos e datasets no armazenamento compartilhado.
export HF_HOME=/capstor/scratch/cscs/ftrindad/
export NLTK_DATA="${REPO}"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
export PYTHONUNBUFFERED=1

NODE_ID="${SLURM_ARRAY_TASK_ID}"
MODELS_PER_NODE="${OC_MODELS_PER_NODE:-1}"
START_INDEX=$((NODE_ID * MODELS_PER_NODE))

# Diretório local ao nó para temporários, caches e locks.
LOCAL_ROOT="${SLURM_TMPDIR:-/tmp/${USER}/accuracy-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}-${NODE_ID}}"

mkdir -p \
  "${LOCAL_ROOT}/tmp" \
  "${LOCAL_ROOT}/rpc" \
  "${LOCAL_ROOT}/vllm" \
  "${LOCAL_ROOT}/torchinductor" \
  "${LOCAL_ROOT}/triton"

export LOCAL_ROOT

export TMPDIR="${LOCAL_ROOT}/tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"

export VLLM_RPC_BASE_PATH="${LOCAL_ROOT}/rpc"
export VLLM_CACHE_ROOT="${LOCAL_ROOT}/vllm"
export TORCHINDUCTOR_CACHE_DIR="${LOCAL_ROOT}/torchinductor"
export TRITON_CACHE_DIR="${LOCAL_ROOT}/triton"

echo "REPO=${REPO}"
echo "OC_DIR=${OC_DIR}"
echo "OC_VENV=${OC_VENV}"
echo "VLLM_VENV=${VLLM_VENV}"
echo "HF_HOME=${HF_HOME}"
echo "LOCAL_ROOT=${LOCAL_ROOT}"
echo "TMPDIR=${TMPDIR}"
echo "VLLM_RPC_BASE_PATH=${VLLM_RPC_BASE_PATH}"
echo "VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT}"
echo "TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR}"
echo "TRITON_CACHE_DIR=${TRITON_CACHE_DIR}"

if [[ -n "${SLURM_TMPDIR:-}" ]]; then
  echo "Usando SLURM_TMPDIR=${SLURM_TMPDIR}"
else
  echo "SLURM_TMPDIR não definido; usando fallback local ${LOCAL_ROOT}"
fi

LOG_DIR="${OC_DIR}/logs/${SLURM_ARRAY_JOB_ID}/node_${NODE_ID}"
RUNTIME_DIR="${OC_DIR}/runtime/${SLURM_ARRAY_JOB_ID}/node_${NODE_ID}"
EXPERIMENT_DIR="${OC_DIR}/experiments/pareto_v0/${SLURM_ARRAY_JOB_ID}/node_${NODE_ID}"

mkdir -p \
  "${LOG_DIR}" \
  "${RUNTIME_DIR}" \
  "${EXPERIMENT_DIR}"

cd "${OC_DIR}"

export SSL_CERT_FILE="$("${VLLM_VENV}/bin/python" -m certifi)"

exec "${VLLM_VENV}/bin/python" \
  "${OC_DIR}/examples/pareto_v0/pareto_gpu_scheduler.py" \
  --registry "${OC_DIR}/examples/pareto_v0/load_vars.py" \
  --opencompass-config "${OC_DIR}/examples/pareto_v0/pareto_v0_accuracy_eval_api.py" \
  --opencompass-cwd "${OC_DIR}" \
  --vllm-bin "${VLLM_VENV}/bin/vllm" \
  --opencompass-bin "${OC_VENV}/bin/opencompass" \
  --runtime-dir "${RUNTIME_DIR}" \
  --log-dir "${LOG_DIR}" \
  --host 127.0.0.1 \
  --base-port "$((8000 + NODE_ID * 5))" \
  --api-key "${VLLM_API_KEY}" \
  --total-gpus 4 \
  --start-index "${START_INDEX}" \
  --max-models "${MODELS_PER_NODE}" \
  --sort-by-gpus desc \
  --reuse 20260409_135608 \
  --work-root experiments/pareto_v0/ \
  --max-model-len 131072 \
  --max-num-batched-tokens 16000 \
  --gpu-memory-utilization 0.95 \
  --oc-infer-workers 1 \
  --oc-api-workers 1 \
  --oc-concurrent-users 1 \
  --oc-qps 1000 \
  --oc-eval-workers 18
'