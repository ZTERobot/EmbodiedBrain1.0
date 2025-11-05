import json
import base64
from io import BytesIO
from PIL import Image
import traceback

# ===== 异步编程关键导入 =====
import asyncio    # Python异步编程核心库，提供事件循环、协程等功能
import aiohttp    # 异步HTTP客户端，替代同步的requests库
# ===== 异步编程关键导入结束 =====

import time
import logging
from tqdm.asyncio import tqdm as async_tqdm  # 异步版本的进度条，支持异步任务进度显示
import argparse
import os
import re
from datetime import datetime


def setup_logging(log_dir, model_name):
    """设置日志配置"""
    log_file = os.path.join(log_dir, f"{model_name}_embspatial_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return log_file


def encode_image_to_base64(image_path):
    """将图片编码为base64格式"""
    file_format = image_path.split('.')[-1]
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return f"data:image/{file_format};base64,{encoded_string}"


def parse_answer_choice(answer_text):
    """
    从模型返回的文本中解析出ABCD选项，并映射为数字索引0-3
    原则：挑选最后一个匹配的选项
    
    Args:
        answer_text (str): 模型返回的答案文本
    
    Returns:
        int: 选项索引 (A->0, B->1, C->2, D->3)，如果解析失败返回-1
    """
    if not answer_text:
        return -1
    
    # 使用正则表达式匹配所有 (A), (B), (C), (D) 模式，取最后一个
    pattern = r'\(([ABCD])\)'
    matches = re.findall(pattern, answer_text.upper())
    
    if matches:
        # 取最后一个匹配的选项
        choice = matches[-1]
        # 映射 A->0, B->1, C->2, D->3
        choice_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        return choice_mapping.get(choice, -1)
    
    # 如果没有匹配到括号模式，尝试匹配所有单独的字母，取最后一个
    pattern_simple = r'\b([ABCD])\b'
    matches_simple = re.findall(pattern_simple, answer_text.upper())
    
    if matches_simple:
        # 取最后一个匹配的选项
        choice = matches_simple[-1]
        choice_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        return choice_mapping.get(choice, -1)
    
    return -1


def build_case(image_base64, obj_names, question, answer_options, model_name):
    """
    构建API请求体
    注意：这个函数保持同步，因为只是数据组装，不涉及I/O操作
    """
    prompt = f"<image>\nAssume you are a viewer seeing current observation. You are supposed to understand the spatial relationships among\
                several objects. The spatial relationships should be described in the viewer's perspective.\
                You need to select the option to answer the question below: \n \
                Question:{question}\n \
                Options: (A){answer_options[0]}, (B){answer_options[1]}, (C){answer_options[2]}, (D){answer_options[3]}\
                1. Please first describe the position of {obj_names} respectively in the image.\
                2. Please choose the option to answer the question above. Your answer should be one of (A), (B), (C), (D)."
    
    return {
        "model": model_name,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user",
                "content": [{
                    "type": "text",
                    "text": prompt  
                },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_base64
                        }
                    }
                ]}
            ]
        }


# ===== 核心异步函数 =====
async def async_model_inference(session, sem, idx, item, progress, server_url, model_name, progress_file, data_dir):
    """
    异步模型推理函数 - 这是最重要的改动！
    
    关键概念解释：
    1. async def: 声明这是一个"协程函数"，可以暂停和恢复执行
    2. await: 在等待操作完成时，让出控制权给其他协程
    3. sem (Semaphore): 信号量，控制同时运行的协程数量，避免过载
    
    """
    # 检查是否已经处理过（断点续传功能）
    if str(idx) in progress:
        return progress[str(idx)]
    
    try:
        # ===== 信号量控制并发数 =====
        # async with sem: 这行代码的作用：
        # 1. 获取信号量许可（如果当前并发数已满，会在这里等待）
        # 2. 进入代码块后，并发数+1
        # 3. 离开代码块时，并发数-1，释放许可给其他等待的协程
        async with sem:
            question = item.get("question")
            image_path = item.get("image")
            answer_options = item.get("answer_options")
            correct_answer_index = item.get("answer")
            objects = item.get("objects")
            
            full_image_path = os.path.join(data_dir, image_path)
            
            # 数据完整性检查
            if not all([question, image_path, answer_options is not None, correct_answer_index is not None]):
                logging.warning(f"第 {idx + 1} 行数据不完整或缺少关键字段，跳过。")
                return None
            
            if not isinstance(answer_options, list) or not answer_options:
                logging.warning(f"第 {idx + 1} 行 'answer_options' 格式不正确或为空，跳过。")
                return None
            
            if not isinstance(correct_answer_index, int) or not (0 <= correct_answer_index < len(answer_options)):
                logging.warning(f"第 {idx + 1} 行 'answer' 索引无效或格式不正确，跳过。")
                return None
            
            # 生成对象名称列表
            obj_names = ", ".join([obj["name"] for obj in objects]) if objects else ""
            
            # 编码图片（同步操作，因为是本地文件读取）
            image_base64 = encode_image_to_base64(full_image_path)
            
            # 构建API请求
            payload = build_case(image_base64, obj_names, question, answer_options, model_name)
            
            # ===== 异步HTTP请求 - 核心改动！=====
            # 原版使用：response = requests.post(url, json=payload)  # 同步，会阻塞
            # 异步版使用：async with session.post() as resp:         # 异步，不阻塞
            # 
            # 关键差异：
            # - 同步版本：发送请求后，线程被阻塞，直到服务器返回响应
            # - 异步版本：发送请求后，立即让出控制权，可以处理其他请求
            #           当服务器响应到达时，再恢复这个协程的执行
            async with session.post(f"{server_url}/v1/chat/completions", json=payload, timeout=60) as resp:
                result = await resp.json()  # await: 等待响应解析为JSON，期间可以处理其他任务
                model_answer = result["choices"][0]["message"]["content"]
                
                # 解析答案（与原版逻辑相同）
                predicted_answer_index = parse_answer_choice(model_answer)
                
                # 判断是否正确
                is_correct = predicted_answer_index == correct_answer_index
                
                # 构建结果（比原版更详细，便于调试和分析）
                result_item = {
                    "line_num": idx + 1,
                    "question": question,
                    "answer_options": answer_options,
                    "correct_answer_index": correct_answer_index,
                    "correct_answer_text": answer_options[correct_answer_index] if 0 <= correct_answer_index < len(answer_options) else "N/A",
                    "model_raw_answer": model_answer,
                    "predicted_answer_index": predicted_answer_index,
                    "predicted_answer_text": answer_options[predicted_answer_index] if 0 <= predicted_answer_index < len(answer_options) else "N/A",
                    "is_correct": is_correct,
                    "image_path": image_path,
                    "objects": objects
                }
                
                # ===== 进度保存机制 =====
                # 原版没有这个功能，异步版本添加了断点续传
                # 每完成一个样本，就保存进度，防止程序中断导致重新开始
                progress[str(idx)] = result_item
                save_progress(progress, progress_file)
                
                return result_item
                
    except Exception as e:
        logging.error(f"第 {idx + 1} 行处理失败: {e}")
        return None


def load_progress(progress_file):
    """
    加载进度文件 - 断点续传功能
    原版没有这个功能，这是异步版本的新增功能
    """
    if os.path.exists(progress_file):
        with open(progress_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_progress(progress_dict, progress_file):
    """
    保存进度到文件 - 断点续传功能
    原版没有这个功能，这是异步版本的新增功能
    """
    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump(progress_dict, f, indent=2, ensure_ascii=False)


# ===== 主评测函数 - 核心架构改动 =====
async def evaluate_model_async(data_dir: str, model_name: str, server_url: str, output_dir: str, log_dir: str, max_concurrent: int = 10):
    """
    异步并行评测多模态大模型的性能
    
    关键架构变化：
    1. 原版：逐个处理，for循环串行执行
    2. 异步版：批量并发，创建多个协程同时执行
    
    """
    log_file = setup_logging(log_dir, model_name)
    
    model_output_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    
    # 创建进度文件路径
    progress_file = os.path.join(model_output_dir, "progress.json")
    
    # 结果文件路径
    results_file = os.path.join(model_output_dir, "embspatial_results.json")
    detailed_results_file = os.path.join(model_output_dir, "detailed_results.jsonl")
    
    data_path = os.path.join(data_dir, 'embspatial_bench_new.jsonl')
    print(f"开始异步评测，数据文件: {data_path}")
    print(f"模型名称: {model_name}")
    print(f"服务器地址: {server_url}")
    print(f"最大并发数: {max_concurrent}")  # 新增：显示并发配置
    print(f"输出目录: {model_output_dir}")
    print(f"日志文件: {log_file}")
    print("-" * 60)
    
    logging.info("开始异步评测，数据文件: %s", data_path)
    logging.info("模型名称: %s", model_name)
    logging.info("服务器地址: %s", server_url)
    logging.info("最大并发数: %d", max_concurrent)
    logging.info("输出目录: %s", model_output_dir)
    logging.info("日志文件: %s", log_file)
    logging.info("-" * 60)
    
    # ===== 数据加载方式改变 =====
    # 原版：边读边处理（流式处理）
    # 异步版：一次性加载所有数据到内存，便于批量并发处理
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data_items = [json.loads(line.strip()) for line in f if line.strip()]
        
        if not data_items:
            print("警告: 评测文件为空。")
            logging.warning("警告: 评测文件为空。")
            return
        
        print(f"加载了 {len(data_items)} 个评测样本")
        logging.info("加载了 %d 个评测样本", len(data_items))
        
    except FileNotFoundError:
        print(f"错误: 文件 '{data_path}' 未找到。")
        logging.error("错误: 文件 '%s' 未找到。", data_path)
        return
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        logging.error("读取文件时发生错误: %s", e)
        return
    
    # 加载进度（断点续传）
    progress = load_progress(progress_file)
    
    # ===== 并发控制机制 =====
    # Semaphore（信号量）：控制同时运行的协程数量
    sem = asyncio.Semaphore(max_concurrent)
    
    # ===== 异步HTTP会话管理 =====
    # aiohttp.ClientSession：管理HTTP连接池，复用连接提高效率
    async with aiohttp.ClientSession() as session:
        tasks = []
        
        # ===== 创建协程任务 - 关键架构差异！=====
        for i, item in enumerate(data_items):
            # asyncio.create_task：将协程封装为任务，加入事件循环
            # 此时任务还没开始执行，只是创建了任务对象
            task = asyncio.create_task(
                async_model_inference(session, sem, i, item, progress, server_url, model_name, progress_file, data_dir)
            )
            tasks.append(task)
        
        # ===== 并发执行所有任务 - 核心性能提升点！=====
        # asyncio.as_completed：按完成顺序返回结果，而非按创建顺序
        print("开始异步处理...")
        results = []
        async for task in async_tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="处理中"):
            result = await task  # 等待单个任务完成
            if result is not None:
                results.append(result)
    
    total_predictions = len(results)
    correct_predictions = sum(1 for r in results if r["is_correct"])
    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    
    # 保存结果
    final_results = {
        "model_name": model_name,
        "server_url": server_url,
        "data_file": data_path,
        "total_samples": total_predictions,
        "correct_predictions": correct_predictions,
        "accuracy": round(accuracy, 2),
        "max_concurrent": max_concurrent,  # 新增：记录并发配置
        "evaluation_time": datetime.now().isoformat(),
        "log_file": log_file
    }
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    
    # 保存详细结果
    with open(detailed_results_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print("\n" + "=" * 60)
    print("评测结果:")
    print(f"总样本数: {total_predictions}")
    print(f"正确预测数: {correct_predictions}")
    print(f"准确率: {accuracy:.2f}%")
    print(f"最大并发数: {max_concurrent}")
    print(f"结果文件: {results_file}")
    print(f"详细结果文件: {detailed_results_file}")
    print("=" * 60)
    
    logging.info("\n" + "=" * 60)
    logging.info("评测结果:")
    logging.info("总样本数: %d", total_predictions)
    logging.info("正确预测数: %d", correct_predictions)
    logging.info("准确率: %.2f%%", accuracy)
    logging.info("最大并发数: %d", max_concurrent)
    logging.info("结果文件: %s", results_file)
    logging.info("详细结果文件: %s", detailed_results_file)
    logging.info("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(description='EmbSpatial Benchmark Async Evaluation')
    parser.add_argument('--model_name', type=str, default='qwen2.5vl_7b', 
                       help='模型名称 (默认: qwen2.5vl_7b)')
    parser.add_argument('--server_url', type=str, default='http://127.0.0.1:7807',
                       help='vLLM服务器URL (默认: http://127.0.0.1:7807)')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--output_dir', type=str,
                       default='/mnt/shared/wireless_robot/fansiyuan/benchmark/EmbSpatial_async/outputs',
                       help='输出目录')
    parser.add_argument('--log_dir', type=str,
                       default='/mnt/shared/wireless_robot/fansiyuan/benchmark/EmbSpatial_async/logs',
                       help='日志目录')
    parser.add_argument('--max_concurrent', type=int, default=10,
                       help='最大并发数 (默认: 10)')  # 新增：并发数配置
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # ===== 异步程序入口 =====
    # asyncio.run()：启动异步事件循环，执行主协程
    # 这是异步程序的标准启动方式
    asyncio.run(evaluate_model_async(
        data_dir=args.data_dir,
        model_name=args.model_name,
        server_url=args.server_url,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        max_concurrent=args.max_concurrent
    )) 