import os
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'
import json
import time
import requests
import numpy as np
from tqdm import tqdm
import base64
# 系统提示词：让模型只返回选项
# SYSTEM_NEW = "你是一个视觉推理助手，请根据提供的问题和选项，严格只返回正确选项的字母（如(A)或B），不需要任何额外解释。"
SYSTEM_NEW = "You are a visual reasoning assistant. Based on the questions and options provided, please return only the correct option letter (such as (A) or (B)) without any additional explanation."
def read_jsonl_file(file_path):
    """读取JSONL文件并返回数据列表"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data
def encode_image_to_base64(image_path):
    """Encode image to Base64"""
    file_format = image_path.split('.')[-1]
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return f"data:image/{file_format};base64,{encoded_string}"
def construct_message(prompt,image_base64):
    """构造模型输入的message"""
    return [
        {"role": "system", "content": SYSTEM_NEW},
        {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_base64}}
                ]
        }
    ]

def call_model(prompt, image_base64,model_url='http://localhost:7803/v1/chat/completions'):
    """调用模型API获取回答"""
    intent_input = {
        "model": "",
        "messages": construct_message(prompt,image_base64)
    }
    headers = {
        "Content-Type": "application/json"
    }
    
    start_time = time.perf_counter()
    try:
        response = requests.post(model_url, headers=headers, data=json.dumps(intent_input), timeout=30)
        response.raise_for_status()
        
        # 解析模型返回结果
        result = response.json()['choices'][-1]['message']['content']
        print(result)
        # 清理结果中的多余字符，只保留选项字母
        if "(" in result and ")" in result:
            answer = result[result.find("(")+1:result.find(")")]
        else:
            # 处理没有括号的情况，取第一个大写字母
            answer = ''.join([c for c in result if c.isupper()])[:1]
        return answer
    except requests.exceptions.RequestException as e:
        print(f"API调用错误: {e}")
        return None
    except KeyError as e:
        print(f"解析响应错误: {e}，响应内容: {response.text}")
        return None

def evaluate_model_2d(data_2d, data_3d, model_url,model_name, results_dir, data_dir):
    """评估模型在2D和3D任务上的表现"""
    os.makedirs(results_dir, exist_ok=True)
    results_file_path = os.path.join(results_dir, f"{model_name}_results.jsonl")
    # 初始化统计变量
    total_2d = len(data_2d)
    total_3d = len(data_3d)
    correct_2d = 0
    correct_3d = 0
    correct_2d_count = 0
    correct_2d_relation = 0
    count_2d_count = 0
    count_2d_relation = 0
    
    all_evaluation_results = []

    print(f"开始评测: 2D任务 {total_2d}条, 3D任务 {total_3d}条")
    
    # 评测2D任务
    print("\n=== 评测2D任务 ===")
    for item in tqdm(data_2d):
        prompt = item["prompt"]
        correct_answer = item["answer"]
        img_path = item['image']
        img_path = os.path.join(data_dir,img_path)
        image_base64 = encode_image_to_base64(img_path)
        task_type = item.get("task", "unknown")
        
        # 调用模型获取回答
        model_answer = call_model(prompt,image_base64, model_url)
        if not model_answer:
            continue
        print("##################")
        print(f"model_answer:{model_answer}")
        print(f"correct_answer:{correct_answer}")
        # 判断回答是否正确
        is_correct = model_answer.upper() == correct_answer[1] if correct_answer.startswith("(") else model_answer.upper() == correct_answer
        print(is_correct)
        all_evaluation_results.append({
            "task_type": task_type,
            "prompt": prompt,
            "image_path": img_path,
            "model_answer": model_answer,
            "correct_answer": correct_answer, # Store original correct answer
            "is_correct": is_correct
        })
        # 按任务类型统计
        if task_type == "Count":
            count_2d_count += 1
            if is_correct:
                correct_2d_count += 1
        elif task_type == "Relation":
            count_2d_relation += 1
            if is_correct:
                correct_2d_relation += 1
    
    # 评测3D任务
    print("\n=== 评测3D任务 ===")
    for item in tqdm(data_3d):
        prompt = item["prompt"]
        correct_answer = item["answer"]
        img_path = item['image']
        img_path = os.path.join(data_dir,img_path)
        image_base64 = encode_image_to_base64(img_path)
        # 调用模型获取回答
        model_answer = call_model(prompt, image_base64,model_url)
        if not model_answer:
            continue
        
        # 判断回答是否正确
        is_correct = model_answer.upper() == correct_answer[1] if correct_answer.startswith("(") else model_answer.upper() == correct_answer
        if is_correct:
            correct_3d += 1
        all_evaluation_results.append({
            "task_type": "3D",
            "prompt": prompt,
            "image_path": img_path,
            "model_answer": model_answer,
            "correct_answer": correct_answer, # Store original correct answer
            "is_correct": is_correct
        })
    with open(results_file_path, 'w', encoding='utf-8') as outfile:
        for record in all_evaluation_results:
            outfile.write(json.dumps(record, ensure_ascii=False) + '\n')
    # 计算准确率
    accuracy_3d = correct_3d / total_3d * 100 if total_3d > 0 else 0
    accuracy_2d_count = correct_2d_count / count_2d_count * 100 if count_2d_count > 0 else 0
    accuracy_2d_relation = correct_2d_relation / count_2d_relation * 100 if count_2d_relation > 0 else 0
    accuracy_2d = (accuracy_2d_count + accuracy_2d_relation) /2 
    accuracy_overall = (accuracy_2d + accuracy_3d) / 2
    
    # 输出评测结果
    print("\n=== 评测结果 ===")
    if count_2d_count > 0:
        print(f"2D任务-计数(Count)准确率: {accuracy_2d_count:.2f}% ({correct_2d_count}/{count_2d_count})")
    if count_2d_relation > 0:
        print(f"2D任务-关系(Relation)准确率: {accuracy_2d_relation:.2f}% ({correct_2d_relation}/{count_2d_relation})")
    print(f"3D任务准确率: {accuracy_3d:.2f}% ({correct_3d}/{total_3d})")
    print(f"2D任务准确率: {accuracy_2d:.2f}")
    print(f"整体平均准确率: {accuracy_overall:.2f}%")
    
    return {
        "accuracy_2d": accuracy_2d,
        "accuracy_3d": accuracy_3d,
        "accuracy_2d_count": accuracy_2d_count,
        "accuracy_2d_relation": accuracy_2d_relation,
        "accuracy_overall": accuracy_overall
    }
import argparse
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate a visual reasoning model.")
    parser.add_argument("--model_url", type=str, required=True,
                        help="The URL of the model API (e.g., http://localhost:31415/v1/chat/completions)")
    parser.add_argument("--model_name", type=str, required=True,
                        help="The name of the model (e.g., qwen2_vl-7b-ultraif)")
    parser.add_argument('--data_dir', type=str,
                    default="/mnt/shared/wireless_robot/benchmark/cv_bench")
    parser.add_argument('--output_dir', type=str,
                    default="./evaluation_results",
                    help='输出目录')
    
    args = parser.parse_args()

    # File paths
    file_path_2d = os.path.join(args.data_dir,"test_2d_val.jsonl")
    file_path_3d = os.path.join(args.data_dir,"test_3d_val.jsonl")
    
    # Model API URL and Name
    model_url = args.model_url
    model_name = args.model_name
    
    # Results directory
    results_dir = args.output_dir

    # Read data
    print("Reading data...")
    data_2d = read_jsonl_file(file_path_2d)
    data_3d = read_jsonl_file(file_path_3d)
    
    # Evaluate model
    evaluation_summary = evaluate_model_2d(data_2d, data_3d, model_url, model_name, results_dir, args.data_dir)
    print("\nEvaluation completed.")
    print("Summary:", evaluation_summary)

if __name__ == "__main__":
    main()

#  python eval.py --model_url http://localhost:31415/v1/chat/completions --model_name fuse_grpo_ckpt100