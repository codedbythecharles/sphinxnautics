#!/bin/bash

MAX_LEN=${1:-16384}
TP_SIZE=${2:-2}
PORT=${3:-8000}
CUDA_DEVICES=${4:-0,1}
MAIN_MODEL=${5:-"Qwen/Qwen2.5-Coder-32B-Instruct"}
MAIN_DTYPE=${6:-"bfloat16"}
MAIN_QUANT=${7:-""}

DRAFT_MODEL=${8:-""}
DRAFT_DTYPE=${9:-$MAIN_DTYPE}
DRAFT_QUANT=${10:-""}

echo "Starting all vLLM servers with:"
echo "  max_model_len = $MAX_LEN"
echo "  tensor_parallel_size = $TP_SIZE"
echo "  PORT = $PORT"
echo "  CUDA_DEVICES = $CUDA_DEVICES"
echo "  MAIN_MODEL = $MAIN_MODEL"
echo "  MAIN_QUANT = $MAIN_QUANT"
echo "  DRAFT_MODEL = $DRAFT_MODEL"
echo "  DRAFT_QUANT = $DRAFT_QUANT"

# Build command as an array
CMD=(CUDA_VISIBLE_DEVICES=$CUDA_DEVICES vllm serve)

# Main model block
CMD+=("$MAIN_MODEL" --dtype $MAIN_DTYPE)
if [[ -n "$MAIN_QUANT" ]]; then
    CMD+=(--quantization $MAIN_QUANT)
fi

# Draft model block (if provided)
if [[ -n "$DRAFT_MODEL" ]]; then
    CMD+=("$DRAFT_MODEL" --dtype $DRAFT_DTYPE)
    if [[ -n "$DRAFT_QUANT" ]]; then
        CMD+=(--quantization $DRAFT_QUANT)
    fi
fi

# Global options
#CMD+=(--port $PORT --max-model-len $MAX_LEN --tensor-parallel-size $TP_SIZE --trust-remote-code)
CMD=(vllm serve)
# Print for debugging
echo "Launching: CUDA_VISIBLE_DEVICES=$CUDA_DEVICES ${CMD[@]}"

# Actually launch
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES "${CMD[@]}" &

#CUDA_VISIBLE_DEVICES=2,3 vllm serve $MODEL_NAME \
 # --dtype $DTYPE --port 8002 --max-model-len $MAX_LEN --tensor-parallel-size $TP_SIZE &

#CUDA_VISIBLE_DEVICES=4,5 vllm serve $MODEL_NAME \
 # --dtype $DTYPE --port 8004 --max-model-len $MAX_LEN --tensor-parallel-size $TP_SIZE &

#CUDA_VISIBLE_DEVICES=6,7 vllm serve $MODEL_NAME \
 # --dtype $DTYPE --port 8006 --max-model-len $MAX_LEN --tensor-parallel-size $TP_SIZE &

wait
