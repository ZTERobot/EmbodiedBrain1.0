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

# 获取当前脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DATA_DIR="${SCRIPT_DIR}/../../test_data/EgoThink/data"
SRC_CODE_PATH="${SCRIPT_DIR}/../src/EgoThink-main/api_eval.py" 

echo "数据路径: $BASE_DATA_DIR"
echo "评测代码路径: $SRC_CODE_PATH"

# 任务列表（任务名 => 相对路径）
declare -A TASKS=(
    ["Activity"]="${BASE_DATA_DIR}/Activity/"
    ["Forecasting"]="${BASE_DATA_DIR}/Forecast/"
    ["Localization_location"]="${BASE_DATA_DIR}/Localization/location/"
    ["Localization_spatial"]="${BASE_DATA_DIR}/Localization/spatial/"
    ["Object_affordance"]="${BASE_DATA_DIR}/Object/affordance/"
    ["Object_attribute"]="${BASE_DATA_DIR}/Object/attribute/"
    ["Object_existence"]="${BASE_DATA_DIR}/Object/existence/"
    ["Planning_assistance"]="${BASE_DATA_DIR}/Planning/assistance/"
    ["Planning_navigation"]="${BASE_DATA_DIR}/Planning/navigation/"
    ["Reasoning_comparing"]="${BASE_DATA_DIR}/Reasoning/comparing/"
    ["Reasoning_counting"]="${BASE_DATA_DIR}/Reasoning/counting/"
    ["Reasoning_situated"]="${BASE_DATA_DIR}/Reasoning/situated/"
)


# --- 创建保存目录 ---
EgoThink_DIR="${RESULT_DIR}/${MODEL_NAME}/PlanningBenchmarks/EgoThink"
mkdir -p "$EgoThink_DIR"

echo "正在启动 EgoThink 推理... 推理结果保存路径: ${EgoThink_DIR}"
# 遍历任务并执行 api_eval.py
for TASK in "${!TASKS[@]}"; do
    ANNOTATION_PATH="${TASKS[$TASK]}annotations.json"
    ANSWER_PATH="${EgoThink_DIR}/${TASK}"

    echo "正在运行任务: $TASK"
    mkdir -p "$ANSWER_PATH"

    python "$SRC_CODE_PATH" \
        --model_name "$MODEL_NAME" \
        --model_url "$V1_URL" \
        --annotation_path "$ANNOTATION_PATH" \
        --answer_path "$ANSWER_PATH" 
        > "${ANSWER_PATH}/Eval_${TASK}.log" 2>&1
done
