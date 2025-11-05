import json
from collections import defaultdict

# 假设数据存储在jsonl文件中
def calculate_accuracy(file_path):
    stats = defaultdict(lambda: {"correct": 0, "total": 0})
    
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            task_type = data["task_type"]
            is_correct = data["is_correct"]
            
            stats[task_type]["total"] += 1
            if is_correct:
                stats[task_type]["correct"] += 1
    
    # 计算并输出各类型准确率
    accuracy_results = {}
    for task_type, counts in stats.items():
        accuracy = (counts["correct"] / counts["total"]) * 100
        accuracy_results[task_type] = f"{accuracy:.2f}%"
        print(f"{task_type} 准确率: {accuracy:.2f}%")
    
    return accuracy_results

# 调用函数（替换为实际文件路径）
results = calculate_accuracy("/mnt/tenant-home_speed/ywr/cv_bench/evaluation_results/qwen2.5-VL-32B_results.jsonl")