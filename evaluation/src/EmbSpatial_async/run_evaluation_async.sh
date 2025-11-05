#!/bin/bash

# EmbSpatial Benchmark 异步并行评测脚本
# 使用vllm服务进行EmbSpatial任务的并行评测

set -e  # 遇到错误时退出

# =============================================================================
# 配置参数 - 根据实际情况修改或通过命令行参数覆盖
# 用法:
#   ./run_evaluation_async.sh
#   ./run_evaluation_async.sh --server_url http://another_host:port --model_name your_model_name
# =============================================================================

# vLLM服务器配置 (默认值)
VLLM_SERVER_URL="http://127.0.0.1:8000"
# 模型名称 (默认值)
MODEL_NAME="qwen2_vl-7b-tulu5k-ultra10k-embspatial15k-lrv30k"   # 与vllm部署时设置的模型名称一致

# 解析命令行参数来覆盖默认值
while [[ $# -gt 0 ]]; do
  case "$1" in
    --server_url)
      VLLM_SERVER_URL="$2"
      shift 2
      ;;
    --model_name)
      MODEL_NAME="$2"
      shift 2
      ;;
    *)
      # 未知参数，可以忽略或报错
      shift
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 评测数据文件
EVALUATION_DATA_DIR="${SCRIPT_DIR}/../../../test_data/EmbSpatial"

# 输出目录
OUTPUT_DIR="${SCRIPT_DIR}/outputs"
LOG_DIR="${SCRIPT_DIR}/logs"

# 并发配置 - 可根据服务器性能调整
MAX_CONCURRENT=10  # 默认并发数为10，可根据需要调整

# 创建必要的目录
mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR

# =============================================================================
# 清理现有结果文件
# =============================================================================
cleanup_existing_results() {
    local model_output_dir="$OUTPUT_DIR/$MODEL_NAME"
    
    echo "检查并清理现有结果文件..."
    
    if [[ -d "$model_output_dir" ]]; then
        echo "发现现有结果目录: $model_output_dir"
        
        # 清理进度文件（断点续传文件）
        if [[ -f "$model_output_dir/progress.json" ]]; then
            echo "删除现有进度文件: progress.json"
            rm -f "$model_output_dir/progress.json"
        fi
        
        # 清理结果文件
        if [[ -f "$model_output_dir/embspatial_results.json" ]]; then
            echo "删除现有结果文件: embspatial_results.json"
            rm -f "$model_output_dir/embspatial_results.json"
        fi
        
        # 清理详细结果文件
        if [[ -f "$model_output_dir/detailed_results.jsonl" ]]; then
            echo "删除现有详细结果文件: detailed_results.jsonl"
            rm -f "$model_output_dir/detailed_results.jsonl"
        fi
        
        echo "现有结果文件清理完成"
    else
        echo "未发现现有结果目录，将创建新的结果目录"
    fi
    
    echo
}


# =============================================================================
# 运行评测
# =============================================================================
run_benchmark() {
    local log_file="$LOG_DIR/${MODEL_NAME}_embspatial_benchmark_async_c${MAX_CONCURRENT}.log"
    
    echo "开始EmbSpatial异步并行评测"
    echo "日志文件: $log_file"
    
    python3 "${SCRIPT_DIR}/eval_embspatial_async.py" \
        --model_name "$MODEL_NAME" \
        --server_url "$VLLM_SERVER_URL" \
        --data_dir "$EVALUATION_DATA_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --log_dir "$LOG_DIR" \
        --max_concurrent "$MAX_CONCURRENT" \
        2>&1 | tee "$log_file"
    
    if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
        echo "EmbSpatial异步评测完成"
    else
        echo "EmbSpatial异步评测失败"
        return 1
    fi
}

# =============================================================================
# 主函数
# =============================================================================
main() {
    echo "======================================================================"
    echo "                 EmbSpatial Benchmark 异步并行评测脚本"
    echo "======================================================================"
    echo "模型名称: $MODEL_NAME"
    echo "服务器地址: $VLLM_SERVER_URL"
    echo "评测数据文件: $EVALUATION_DATA_DIR"
    echo "输出目录: $OUTPUT_DIR"
    echo "日志目录: $LOG_DIR"
    echo "最大并发数: $MAX_CONCURRENT"
    echo "======================================================================"
    
    # 记录开始时间
    start_time=$(date +%s)
    
    
    # 检查Python依赖
    echo "检查Python依赖..."
    python3 -c "import asyncio, aiohttp, tqdm" 2>/dev/null || {
        echo "错误: 缺少必要的Python依赖包"
        echo "请安装: pip install aiohttp tqdm"
        exit 1
    }
    
    # 清理现有结果文件
    cleanup_existing_results
    
    # 运行评测
    if run_benchmark; then
        echo "评测成功完成"
    else
        echo "评测失败"
        exit 1
    fi
    
    # 计算总耗时
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    echo
    echo "======================================================================"
    echo "评测完成!"
    echo "总耗时: ${duration}秒"
    echo "并发数: $MAX_CONCURRENT"
    echo "结果文件位于: $OUTPUT_DIR/$MODEL_NAME/"
    echo "日志文件位于: $LOG_DIR/"
    
    # 显示结果摘要
    echo
    echo "评测结果摘要:"
    result_file="$OUTPUT_DIR/$MODEL_NAME/embspatial_results.json"
    if [[ -f "$result_file" ]]; then
        echo "结果文件: $result_file"
        # 显示准确率信息
        if command -v jq &> /dev/null; then
            accuracy=$(jq -r '.accuracy' "$result_file" 2>/dev/null || echo "N/A")
            total_samples=$(jq -r '.total_samples' "$result_file" 2>/dev/null || echo "N/A")
            correct_predictions=$(jq -r '.correct_predictions' "$result_file" 2>/dev/null || echo "N/A")
            max_concurrent=$(jq -r '.max_concurrent' "$result_file" 2>/dev/null || echo "N/A")
            echo "- 总样本数: $total_samples"
            echo "- 正确预测数: $correct_predictions"
            echo "- 准确率: $accuracy%"
            echo "- 最大并发数: $max_concurrent"
        else
            echo "- 结果文件已生成，请查看: $result_file"
            echo "- 提示: 安装jq命令可显示详细结果摘要"
        fi
    fi
    
    echo "======================================================================"
}

main 