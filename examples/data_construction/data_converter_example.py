import os
import json
import re
import asyncio
import aiohttp
import argparse
from tqdm import tqdm
from copy import deepcopy
from openai import AsyncOpenAI


def load_existing_data(output_path):
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                return data if isinstance(data, list) else []
            except json.JSONDecodeError:
                print(f"警告: {output_path} JSON格式错误，将重新创建文件")
                return []
    return []


def reformat_data(data):
    for conv in data['conversations']:
        if conv['from'] == 'human':
            conv['role'] = 'user'
        elif conv['from'] == 'gpt':
            conv['role'] = 'assistant'
        del conv['from']
        conv['content'] = conv.pop('value')
    return data


async def request_model(client, query: str, sys_prompt: str, model_name: str, max_retries=3) -> str:
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": query}
                ],
                max_tokens=1024,
                temperature=0.5,
                timeout=30,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}}
            )
            content = response.choices[0].message.content.strip()
            return content
        except Exception as e:
            print(f"API错误 (尝试 {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(5)
    return ""


async def writer_task(output_path: str, result_queue: asyncio.Queue, total_count: int, initial_count: int):
    output_data = load_existing_data(output_path)
    pbar = tqdm(total=total_count, initial=initial_count, desc="写入进度")

    try:
        while True:
            item = await result_queue.get()
            if item is None:
                break
            output_data.append(item)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            pbar.update(1)
            result_queue.task_done()
    finally:
        pbar.close()


async def process_item(client, item, semaphore, result_queue, sys_prompt, model_name):
    async with semaphore:
        try:
            for conversation in item["conversations"]:
                if conversation.get("from") == "gpt":
                    gpt_value = conversation.get("value", "")
                    match = re.search(r"response:\s*(.*?)\s*\n\s*plans:\s*(.*)", gpt_value, re.DOTALL)
                    if match:
                        response_text = match.group(1).strip()
                        plans_text = match.group(2).strip()
                        model_output = await request_model(client, plans_text, sys_prompt, model_name)
                        if not model_output:
                            print("模型返回为空，跳过")
                            return

                        reformated_item = deepcopy(item)
                        reformated_item = reformat_data(reformated_item)
                        reformated_item['conversations'][1]['content'] = '<response>' + response_text + '</response>' + model_output
                        await result_queue.put(reformated_item)
                        return
        except Exception as e:
            print(f"处理项出错: {e}")


# 原子动作列表
PRIME_ACTIONS = [
    "Pick", "Pick up", "Place", "Pick and Place", "Grasp", "Release",
    "Grasp and Release", "Fold", "UnFold", "Open", "Close", "Press",
    "HandOver", "Catch", "Pull", "push", "Insert", "Point", "Screw",
    "Unscrew", "Lift", "Turn（Twist）", "Peel", "Stack", "Flip",
    "Slide / Drag", "Tear", "Pinch", "Wipe", "Scoop", "Cut", "Sweep",
    "Stir", "Throw", "Shake", "Wash（Rinse）", "Beat", "Hammer", "Hang",
    "Spread", "Pour", "Navigate", "Rotate in place", "Follow", "Search", "Find"
]


def build_system_prompt(prime_actions):
    return f"""
你是一个专业的数据格式转换专家，需要将一个规划数据描述的数据转换为我要求的格式。请严格按照以下要求和示例进行操作：

### 原始数据结构
1. 一段文字任务规划（plan）流程

### 转换任务
你需要执行以下转换：
1. plans增强要求：
    -- 为每个plan步骤添加原子动作标签（格式：[动作]）
    -- 从以下原子动作列表中选择最合适的动作：{", ".join(prime_actions)}
    -- 示例："1. Navigate to the First Cargo Section" 转化为 "1.[Navigate] Navigate to the First Cargo Section"

2. actions生成规则：
    -- 格式：Python风格的二维列表 [['动作', '对象1'], ['动作', '对象1', '对象2'], ...]
    -- 动作选择：必须从提供的原子动作列表中选择，如果原子动作列表中没有完全一样的动作，那么选择一个最接近的动作即可
    -- 对象规范：1. 使用简洁英文名词短语（如"Cargo Straps"）2. 单对象动作：['动作', '对象']（如['Find', 'Cargo Straps']）3. 双对象动作：['动作', '对象', '目标']（如['Put', 'Bread', 'Basket']）

### **严格禁止**如下操作
1. 修改原始plans的核心语义
2. 添加解释性文本或注释
3. 使用原子动作列表之外的动作
4. 改变对象的核心指代

一个正确的转换后的数据示例：
"<plans>1.[Navigate] Navigate to the First Cargo Section\\n2.[Find] Visually Inspect the Cargo Straps for Tightness\\n3.[Adjust] Adjust any Loose Straps\\n4.[Navigate] Move to the Next Cargo Section</plans><actions>[['Navigate', 'First Cargo Section'], ['Find', 'Cargo Straps'], ['Adjust', 'Loose Straps'], ['Navigate', 'Next Cargo Section']]</actions>"

现在请处理以下数据，直接输出最终的转化结果
"""


async def main(args):
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # 构建系统提示
    sys_prompt = build_system_prompt(PRIME_ACTIONS)

    # 初始化 OpenAI 客户端（从参数传入）
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)

    # 加载输入数据
    with open(args.input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 支持从断点续跑
    output_data = load_existing_data(args.output_path)
    processed_count = len(output_data)
    total_count = len(data)

    print(f"已处理: {processed_count}/{total_count}")

    # 队列与并发控制
    result_queue = asyncio.Queue()
    semaphore = asyncio.Semaphore(args.concurrency)

    # 启动写入任务
    writer = asyncio.create_task(writer_task(args.output_path, result_queue, total_count, processed_count))

    # 创建处理任务
    tasks = [
        process_item(client, item, semaphore, result_queue, sys_prompt, args.model_name)
        for item in data[processed_count:]
    ]

    await asyncio.gather(*tasks)

    # 结束写入
    await result_queue.put(None)
    await writer

    print("✅ 所有数据处理完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="数据格式转换工具：将原始对话数据中的 plans 转换为带原子动作标签和结构化 actions 的格式。")
    parser.add_argument("--input_path", type=str, required=True, help="输入 JSON 文件路径")
    parser.add_argument("--output_path", type=str, required=True, help="输出 JSON 文件路径")
    parser.add_argument("--api_key", type=str, required=True, help="LLM API 密钥")
    parser.add_argument("--base_url", type=str, required=True, help="LLM 服务的 base URL（例如 http://localhost:8000/v1）")
    parser.add_argument("--model_name", type=str, default="default-model", help="使用的模型名称（默认: default-model）")
    parser.add_argument("--concurrency", type=int, default=10, help="并发请求数（默认: 10）")

    args = parser.parse_args()
    asyncio.run(main(args))