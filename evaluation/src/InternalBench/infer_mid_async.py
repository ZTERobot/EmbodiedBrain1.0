# python infer_mid_async.py --api http://localhost:31452/v1/chat/completions --name dapo_sft_plan_rule7_rm3-ckpt700 --data_path sharerobot_benchmark_set.json --save_folder ./results

import json
import base64
import os
import asyncio
import aiohttp
import argparse
import sys

# Create a lock to ensure thread-safe writing to the output file
lock = asyncio.Lock()

MAX_CONCURRENT = 4  # Limit the number of concurrent requests

# Encode image to base64
def encode_images_to_base64(image_paths):
    encoded_images = []
    for image_path in image_paths:
        file_format = image_path.split('.')[-1]
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        encoded_images.append(f"data:image/{file_format};base64,{encoded_string}")
    return encoded_images

# Load QA data from the json file
def extract_qa_image_from_json(file_path):
    tasks, gt_answers, images = [], [], []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        for entry in data:
            # print(entry)
            conversations = entry.get('conversations', [])
            user_content = conversations[0]['content']
            # 使用正则表达式去除 <image>\n 标签
            task = re.sub(r'<image>\n', '', user_content)
            tasks.append(task)
            
            # 提取助手回答 (GT answer)
            assistant_content = conversations[1]['content']
            gt_answers.append(assistant_content)

            # 提取图像路径
            image_paths = entry.get('images', [])  # 获取所有图片路径列表
            images.append(image_paths)

    except Exception as e:
        print(f"Error loading data: {e}")
    return tasks, gt_answers, images

# Load existing progress from a file
def load_progress():
    if os.path.exists(SAVE_FILE):
        with open(SAVE_FILE, 'r') as f:
            return json.load(f)
    return {}

# Save new results to the progress file
def save_progress(progress_dict):
    with open(SAVE_FILE, 'w') as f:
        json.dump(progress_dict, f, indent=2, ensure_ascii=False)
import re


# Build the API request body
def build_case(image_base64_list, question, model_name):
    content = []
    for image_base64 in image_base64_list:
        content.append({"type": "image_url", "image_url": {"url": image_base64}})
    content.append({"type": "text", "text": question})
    
    return {
        "model": model_name,
        "max_tokens": 1024,
        "messages": [
            {
                "role": "system",
                "content": """You are an advanced robot task planner that converts natural language commands into executable structured plans. All operations must be based on the content of provided images, but you do not need to explicitly analyze or describe the images. You must strictly follow these rules:\n\n## Output Specifications\n### Required XML Tags\n1. <response>\n   - Single sentence confirming the task, implicitly based on image content\n   - Example: <response>I will place the book on the table</response>\n\n2. <plans>\n   - Multiple steps (no strict limit on count) formatted as:\n     [Step_Type] Action with [object] and/or [location]\n   - All steps must be based on objects and locations visible in the provided images\n   - Step types are strictly limited to [Navigate] and [Manipulate] (all actions except navigation use [Manipulate])\n   - Step type tags must be capitalized: [Manipulate], [Navigate]\n   - Example:\n     <plans>\n     1.[Manipulate] Locate the book on the shelf\n     2.[Navigate] Move to the shelf\n     3.[Manipulate] Pick up the book\n     4.[Navigate] Move to the table\n     5.[Manipulate] Place the book on the table\n     </plans>\n\n3. <actions>\n   - JSON-style action sequence derived from the plans\n   - Format: [['Verb','Object'], ['Verb','Object','Location'], ...] (must match predicates and objects from <plans> as closely as possible)\n   - All words must be capitalized correctly with first letter uppercase\n   - Verbs must be selected exclusively from the predefined atomic action set (no exceptions)\n   - Example:\n     <actions>\n     [['Search','Book'],['Navigate','Shelf'],['Pick','Book'],['Navigate','Table'],['Place','Book','Table']]\n     </actions>\n\n## Strict Requirements\n1. All planning must be based solely on visible content in the provided images - do not assume existence of objects not shown\n2. Use only these step types in <plans> (must be capitalized):\n   - [Navigate], [Manipulate]\n3. Verbs in <plans> should, when possible, be selected from the atomic action set (listed below)\n4. If verbs in <plans> cannot follow the atomic action set (due to contextual necessity), the corresponding verbs in <actions> must be replaced with the closest matching verb from the atomic action set\n5. Action verbs in <actions> must be selected exclusively from this atomic action set (capitalized correctly with first letter uppercase, no exceptions):\n   - HandShake, Clap, Wave, Bow, Dance, Like, Give a thumbs-up, Stretch, HandOver, Catch, Screw, Unscrew, Peel, Stack, Flip, Heart sign, Make a heart with hands, Tear, Pinch, Search, Find, Scoop, Cut, Chop, Sweep, Stir, Throw, Slide, Drag, Shake, Hammer, Spread, Pour, Navigate, Point, Pick, Place, Pick and Place, Fold, UnFold, Press, Open, Close, Pull, Push, Insert, Grasp and Release, Grasp, Release, Lift, Turn, Twist, Wash, Rinse, Beat, Hang, Follow, Rotate in place\n6. Maintain consistent object naming across all tags, using names that reflect how objects appear in the images\n7. All words in <actions> must have first letter capitalized (e.g., 'Dirty Clothes', 'Washing Machine')\n8. If a predicate and objects/locations are specified in <plans> (e.g., \"Place the Box on the Desk\"), the corresponding <actions> entry must explicitly include them (e.g., ['Place', 'Box', 'Desk'])\n\n## Example Input/Output\nExample 1:\nUser: \"<image><image><image><image>\nPut the dirty clothes in the washing machine\"\nAssistant:\n<response>I will put the dirty clothes in the washing machine</response>\n<plans>\n1.[Manipulate] Locate the dirty clothes in the basket\n2.[Navigate] Navigate to the basket\n3.[Manipulate] Pick up the dirty clothes\n4.[Navigate] Navigate to the washing machine\n5.[Manipulate] Place the dirty clothes in the washing machine\n</plans>\n<actions>\n[['Search','Dirty Clothes'],['Navigate','Basket'],['Pick','Dirty Clothes'],['Navigate','Washing Machine'],['Place','Dirty Clothes','Washing Machine']]\n</actions>\n\n## Error Handling\nFor commands that cannot be executed based on image content or are invalid, return:\n<response>Error: cannot complete task based on image content or invalid planning task.</response><plans></plans><actions></actions>
                """
            },
            {
                "role": "user",
                "content": content
            }
        ]
    }

# Async API call task
async def async_connect_api(session, sem, idx, question, answer, image_paths, progress, api_url, model_name):
    # 检查进度文件中是否已存在该索引的结果
    if str(idx) in progress:
        # 如果存在，则打印提示信息并跳过
        print(f"Skipping index {idx}: Result already exists.")
        # 返回一个值以表示此任务已处理（无论是跳过还是新推理）
        return 1

    async with sem:
        try:
            image_base64_list = encode_images_to_base64(image_paths)
            payload = build_case(image_base64_list, question, model_name)
            
            async with session.post(api_url, json=payload, timeout=60) as resp:
                resp.raise_for_status()  # 检查HTTP请求是否成功
                result = await resp.json()
                
                # 检查API返回的JSON结构是否符合预期
                if "choices" in result and len(result["choices"]) > 0 and "message" in result["choices"][0] and "content" in result["choices"][0]["message"]:
                    model_answer = result["choices"][0]["message"]["content"]
                else:
                    print(f"Error processing index {idx}: Unexpected API response format.")
                    model_answer = "Error: Invalid response format"

                progress[str(idx)] = {
                    "question": question,
                    "model_answer": model_answer,
                    "real_answer": answer,
                    "image_paths": image_paths
                }
                save_progress(progress)  # 保存进度
                score = 1
                return score
        except aiohttp.ClientError as e:
            print(f"An HTTP error occurred for index {idx}: {e}")
            return 0  # 返回0或None表示失败
        except Exception as e:
            print(f"An unexpected error occurred for index {idx}: {e}")
            return 0 # 返回0或None表示失败

sft_user_prompt = "You’re a robot task planner. Convert natural language commands to structured plans based on provided images. Follow these rules:\n## Output Specifications\n### Required XML Tags\n1. <response>\n   - Single sentence confirming the task, implicitly based on image content\n2. <plans>\n   - Multiple steps (no strict limit on count) formatted as:\n     [step_type] Action with [object] and/or [location]\n   - All steps must be based on objects and locations visible in the provided images\n   - Step types are strictly limited to [navigate] and [manipulate] (all actions except navigation use [manipulate])\n3. <actions>\n   - JSON-style action sequence derived from the plans\n   - Format: [['Verb','Object'], ['Verb','Object','Location'], ...] (must match predicates and objects from <plans> as closely as possible)\n   - Verbs must be selected exclusively from the predefined atomic action set (no exceptions)\n## Strict Requirements\n1. All planning must be based solely on visible content in the provided images - do not assume existence of objects not shown\n2. Use only these step types in <plans>:\n   - [navigate], [manipulate]\n3. Verbs in <plans> should, when possible, be selected from the atomic action set (listed below)\n4. If verbs in <plans> cannot follow the atomic action set (due to contextual necessity), the corresponding verbs in <actions> must be replaced with the closest matching verb from the atomic action set\n5. Action verbs in <actions> must be selected exclusively from this atomic action set (capitalized correctly, no exceptions):\n   - HandShake, Clap, Wave, Bow, Dance, Like, Give a thumbs-up, Stretch, HandOver, Catch, Screw, Unscrew, Peel, Stack, Flip, Heart sign, Make a heart with hands, Tear, Pinch, Search, Scoop, Cut, Chop, Sweep, Stir, Throw, Slide, Drag, Shake, Hammer, Spread, Pour, Navigate, Point, Pick, Place, Pick and Place, Fold, UnFold, Press, Open, Close, Pull, Push, Insert, Grasp and Release, Grasp, Release, Lift, Turn, Twist, Wash, Rinse, Beat, Hang, Follow, Rotate in place\n6. Maintain consistent object naming across all tags, using names that reflect how objects appear in the images\n7. If a predicate and objects/locations are specified in <plans> (e.g., \"Place the Box on the Desk\"), the corresponding <actions> entry must explicitly include them (e.g., ['Place', 'Box', 'Desk'])\n## Example Input/Output\nExample 1:\nUser: \"Put the dirty clothes in the washing machine\"\nAssistant:\n<response>I will put the dirty clothes in the washing machine</response><plans>1.[manipulate] Locate the dirty clothes in the basket\n2.[navigate] Navigate to the basket\n3.[manipulate] Pick up the dirty clothes\n4.[navigate] Navigate to the washing machine\n5.[manipulate] Place the dirty clothes in the washing machine</plans><actions>[['Search','Dirty clothes'],['Navigate','Basket'],['Pick','Dirty clothes'],['Navigate','Washing machine'],['Place','Dirty clothes','Washing machine']]</actions>\n## Error Handling\nFor commands that cannot be executed based on image content or are invalid, return:\n<response>Error: cannot complete task based on image content or invalid planning task.</response><plans></plans><actions></actions>\nThe current natural language command is as follows:Please perform the task:"

# Main function
async def main(api_url, model_name, data_path):
    questions, answers, images = extract_qa_image_from_json(data_path)

    progress = load_progress()
    sem = asyncio.Semaphore(MAX_CONCURRENT)

    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(len(questions)):
            task = async_connect_api(session, sem, i, questions[i], answers[i], images[i], progress, api_url, model_name)
            tasks.append(task)

        scores = await asyncio.gather(*tasks)
        print('success inferences:', scores)

# Entry point of the script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the script with API URL and model name.")
    parser.add_argument("--api", type=str, help="API URL", required=True)
    parser.add_argument("--name", type=str, help="Model name", required=True)
    parser.add_argument("--data_path", type=str, help="benchmark set path", required=True)
    parser.add_argument("--save_folder", type=str, help="save path", required=True)
    # parser.add_argument("--timestamp", type=str, help="subfolder", required=True)
    args = parser.parse_args()
    save_path = os.path.join(args.save_folder, args.name)
    os.makedirs(save_path, exist_ok=True)
    SAVE_FILE = os.path.join(save_path, "inference_"+os.path.basename(args.data_path))
    asyncio.run(main(args.api, args.name, args.data_path))
