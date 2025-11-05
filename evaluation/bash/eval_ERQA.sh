# 传入配置参数
MODEL_URL="$1"
MODEL_NAME="$2"
RESULT_DIR="${3:-./results}"

# 必须传入MODEL_URL和MODEL_NAME，RESULT_DIR可选，默认为./results路径

if [ -z "$MODEL_URL" ] || [ -z "$MODEL_NAME" ]; then
    echo "传入参数不完整，正确用法: $0 <MODEL_URL> <MODEL_NAME> [RESULT_DIR]"
    exit 1
fi

V1_URL="${MODEL_URL}/v1"
CHAT_COMPLETIONS_URL="${V1_URL}/chat/completions"

# 获取当前脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DATA_DIR="${SCRIPT_DIR}/../../test_data/ERQA/data/erqa.tfrecord"
SRC_CODE_PATH="${SCRIPT_DIR}/../src/ERQA-main" 
echo "数据路径: $BASE_DATA_DIR"
echo "评测代码路径: $SRC_CODE_PATH"

# --- 创建保存目录 ---
RESULT_DIR="$(realpath "$RESULT_DIR")"
ERQA_DIR="${RESULT_DIR}/${MODEL_NAME}/SpatialBenchmarks/ERQA"
mkdir -p "$ERQA_DIR"
echo "评测代码路径: $SRC_CODE_PATH"

# 环境
cd "$SRC_CODE_PATH"

echo "正在启动 ERQA 推理... 推理结果保存路径: ${ERQA_DIR}"
# 遍历任务并执行 api_eval.py


python eval_harness_vllm.py \
    --url "$CHAT_COMPLETIONS_URL" \
    --model "$MODEL_NAME" \
    --tfrecord_path "$BASE_DATA_DIR" \
    > "${ERQA_DIR}/Eval_${MODEL_NAME}.log" 2>&1

