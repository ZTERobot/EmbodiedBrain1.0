# MODEL_URL="http://0.0.0.0:31450"
# MODEL_NAME="Qwen2.5-VL-7B-Instruct"

# 传入配置参数
MODEL_URL="$1"
MODEL_NAME="$2"
RESULT_DIR="${3:-./results}"

# 必须传入MODEL_URL和MODEL_NAME，RESULT_DIR可选，默认为./results路径

if [ -z "$MODEL_URL" ] || [ -z "$MODEL_NAME" ]; then
    echo "传入参数不完整，正确用法: $0 <MODEL_URL> <MODEL_NAME> [RESULT_DIR]"
    exit 1
fi

# --- 构建不同服务所需的URL ---
V1_URL="${MODEL_URL}/v1"
CHAT_COMPLETIONS_URL="${V1_URL}/chat/completions"

# --- 创建空间评测目录 ---
Spatial_DIR="${RESULT_DIR}/${MODEL_NAME}/SpatialBenchmarks"
mkdir -p "$Spatial_DIR"

# --- 打印解析信息 ---
echo "=================================================="
echo "正在启动空间能力评估... 保存路径: ${Spatial_DIR}"
echo "基础URL: $MODEL_URL"
echo "解析出的模型名称: $MODEL_NAME"
echo "Blink and API URL (v1): $V1_URL"
echo "Chat Completions URL: $CHAT_COMPLETIONS_URL"
echo "=================================================="
echo


# 获取当前脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DATA_DIR="${SCRIPT_DIR}/../../test_data/BLINK"
SRC_CODE_PATH="${SCRIPT_DIR}/../src/BLINK_Benchmark-main/eval/my_test_benchmark_cmd.py" 

echo "数据路径: $BASE_DATA_DIR"
echo "评测代码路径: $SRC_CODE_PATH"

# --- 按顺序执行评估命令 ---
echo "正在启动 BLINK 评估... 日志: ${Spatial_DIR}/blink.log"
mkdir -p "${Spatial_DIR}/BLINK"
echo "BLINK 评估结果保存路径: ${Spatial_DIR}/BLINK"

python "$SRC_CODE_PATH" --base_url "$V1_URL" --model_name "$MODEL_NAME" --data_dir "${BASE_DATA_DIR}" --output_dir "${Spatial_DIR}/BLINK" > "${Spatial_DIR}/blink.log" 2>&1
echo "BLINK 评估完成."
echo "--------------------------------------------------"
echo