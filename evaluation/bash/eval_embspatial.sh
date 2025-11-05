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
echo "API URL (v1): $V1_URL"
echo "Chat Completions URL: $CHAT_COMPLETIONS_URL"
echo "=================================================="
echo


# 获取当前脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DATA_DIR="${SCRIPT_DIR}/../../test_data/EmbSpatial"
SRC_CODE_PATH="${SCRIPT_DIR}/../src/EmbSpatial_async/run_evaluation_async.sh" 

echo "数据路径: $BASE_DATA_DIR"
echo "评测代码路径: $SRC_CODE_PATH"

# --- 按顺序执行评估命令 ---
echo "正在启动 EmbSpatial 评估... 日志: ${Spatial_DIR}/embspatial.log"
"$SRC_CODE_PATH" --server_url "$MODEL_URL" --model_name "$MODEL_NAME" > "${Spatial_DIR}/embspatial.log" 2>&1
echo "EmbSpatial 评估完成."
echo "--------------------------------------------------"
echo
