# Default values
MAX_LEN=${1:-16384}
TP_SIZE=${2:-2}
PORT=${3:-8000}
CUDA_DEVICES=${4:-0,1}
MODEL_NAME=${5:-"Qwen/Qwen2.5-Coder-32B-Instruct"}
DTYPE="bfloat16"

echo "Starting all vLLM servers with:"
echo "  max_model_len = $MAX_LEN"
echo "  tensor_parallel_size = $TP_SIZE"
echo "  PORTS= $PORT"
echo "  CUDA_DEVICES = $CUDA_DEVICES"

CUDA_VISIBLE_DEVICES=$CUDA_DEVICES vllm serve $MODEL_NAME \
  --dtype $DTYPE --port $PORT --max-model-len $MAX_LEN --tensor-parallel-size $TP_SIZE --trust-remote-code &

#CUDA_VISIBLE_DEVICES=2,3 vllm serve $MODEL_NAME \
 # --dtype $DTYPE --port 8002 --max-model-len $MAX_LEN --tensor-parallel-size $TP_SIZE &

#CUDA_VISIBLE_DEVICES=4,5 vllm serve $MODEL_NAME \
 # --dtype $DTYPE --port 8004 --max-model-len $MAX_LEN --tensor-parallel-size $TP_SIZE &

#CUDA_VISIBLE_DEVICES=6,7 vllm serve $MODEL_NAME \
 # --dtype $DTYPE --port 8006 --max-model-len $MAX_LEN --tensor-parallel-size $TP_SIZE &

wait
