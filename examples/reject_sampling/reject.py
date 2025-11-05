import os
import json
import aiohttp
import asyncio
import argparse
import tqdm
from typing import List, Dict, Any
import hashlib
import base64
import re

def image_to_base64(image_path: str) -> str:
    if not image_path or not os.path.exists(image_path):
        return None
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        print(f"Image encoding error: {e}")
        return None

def generate_entry_id(entry: Dict) -> str:
    return hashlib.md5(entry["original_question"].encode()).hexdigest()

def load_existing_data(output_file: str) -> Dict[str, Any]:
    processed_ids = {}
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if "entry_id" in data:
                            processed_ids[data["entry_id"]] = True
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Error loading existing data: {e}")
    return processed_ids

def save_data_incrementally(data: Dict[str, Any], file_path: str):
    try:
        with open(file_path, 'a') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    except Exception as e:
        print(f"Save error: {e}")

async def generate_answer(
    session: aiohttp.ClientSession,
    question: str,
    image_path: str,
    model_name: str,
    api_url: str,
    max_retries: int,
    request_timeout: int
) -> str:
    for attempt in range(max_retries):
        try:
            messages = [{"role": "system", "content": "You are a helpful assistant."}]
            text = question.replace("<image>", "").strip()
            user_content = [{"type": "text", "text": text}]
            if image_path:
                b64 = image_to_base64(image_path)
                if b64:
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                    })
            messages.append({"role": "user", "content": user_content})
            payload = {
                "model": model_name,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 2048,
            }
            async with session.post(api_url, json=payload, timeout=request_timeout) as resp:
                if resp.status != 200:
                    raise aiohttp.ClientError(f"HTTP {resp.status}")
                result = await resp.json()
                return result['choices'][0]['message']['content'].strip()
        except (aiohttp.ClientError, asyncio.TimeoutError, KeyError) as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
            else:
                return f"Error: Generation failed after {max_retries} retries."
    return "Error"

def construct_scoring_prompt(entry: Dict, answer: str) -> List[Dict]:
    original_question = entry["original_question"]
    original_gpt_answer = ""
    if "original_answers" in entry:
        for ans in entry["original_answers"]:
            if ans.get("from") == "gpt":
                original_gpt_answer = ans.get("value", "")
                break
    system_prompt = (
        "You are a strict evaluator. Score the answer as: "
        "10 = correct and confident, "
        "5 = correct but hesitant (uses 'maybe', 'I think', etc.), "
        "0 = factually incorrect. "
        "Output ONLY 0, 5, or 10."
    )
    user_prompt = (
        f"Original Question:\n{original_question}\n\n"
        f"Original Answer:\n{original_gpt_answer}\n\n"
        f"Generated Answer:\n{answer}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

async def score_answer(
    session: aiohttp.ClientSession,
    entry: Dict,
    answer: str,
    model_name: str,
    api_url: str,
    max_retries: int,
    request_timeout: int
) -> float:
    for attempt in range(max_retries):
        try:
            payload = {
                "model": model_name,
                "messages": construct_scoring_prompt(entry, answer),
                "temperature": 0.1,
                "max_tokens": 10,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            async with session.post(api_url, json=payload, timeout=request_timeout) as resp:
                if resp.status != 200:
                    raise aiohttp.ClientError(f"HTTP {resp.status}")
                result = await resp.json()
                content = result['choices'][0]['message']['content'].strip()
                match = re.search(r'\b(10|[0-9])\b', content)
                if match:
                    return float(match.group(0))
                return 5.0
        except (aiohttp.ClientError, asyncio.TimeoutError, KeyError, json.JSONDecodeError) as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
            else:
                return 0.0
    return 0.0

async def process_entry(
    session: aiohttp.ClientSession,
    entry: Dict,
    semaphore: asyncio.Semaphore,
    gen_model: str,
    gen_url: str,
    scorer_model: str,
    scorer_url: str,
    answers_per_q: int,
    max_retries: int,
    request_timeout: int,
    cache_file: str
) -> Dict:
    async with semaphore:
        q = entry["original_question"]
        img = entry.get("image_path")
        gen_tasks = [generate_answer(session, q, img, gen_model, gen_url, max_retries, request_timeout) for _ in range(answers_per_q)]
        answers = await asyncio.gather(*gen_tasks)
        score_tasks = [score_answer(session, entry, a, scorer_model, scorer_url, max_retries, request_timeout) for a in answers]
        scores = await asyncio.gather(*score_tasks)
        avg_score = sum(scores) / len(scores) if scores else 0
        result = {
            "entry_id": generate_entry_id(entry),
            "original_question": q,
            "image_path": img,
            "original_answers": entry.get("original_answers", []),
            "scored_candidates": [{"answer": a, "score": s} for a, s in zip(answers, scores)],
            "average_score": avg_score
        }
        cache_data = {
            "entry_id": result["entry_id"],
            "original_question": q,
            "image_path": img,
            "generated_answers": answers,
            "generated_scores": scores
        }
        save_data_incrementally(cache_data, cache_file)
        return result

async def main(args):
    processed_ids = load_existing_data(args.output_file)
    print(f"Loaded {len(processed_ids)} processed entries.")

    entries_to_process = []
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
            for item in all_data:
                human_prompt = next((conv["value"] for conv in item["conversations"] if conv["from"] == "human"), None)
                if not human_prompt:
                    continue
                image_paths = item.get("images")
                image_path = image_paths[0] if image_paths else None
                entry = {
                    "original_question": human_prompt,
                    "original_answers": [{"from": c["from"], "value": c["value"]} for c in item["conversations"]],
                    "image_path": image_path
                }
                eid = generate_entry_id(entry)
                if eid not in processed_ids:
                    entries_to_process.append(entry)
    except Exception as e:
        print(f"Input loading error: {e}")
        return

    if not entries_to_process:
        print("No new entries to process.")
        return

    print(f"Processing {len(entries_to_process)} entries.")
    semaphore = asyncio.Semaphore(args.concurrency)
    pbar = tqdm.tqdm(total=len(entries_to_process), desc="Reject Sampling")

    gen_url = f"http://localhost:{args.generator_port}/v1/chat/completions"
    scorer_url = f"http://localhost:{args.scorer_port}/v1/chat/completions"

    async with aiohttp.ClientSession() as session:
        tasks = [
            process_entry(
                session, entry, semaphore,
                args.generator_model, gen_url,
                args.scorer_model, scorer_url,
                args.answers_per_question,
                args.max_retries,
                args.request_timeout,
                args.cache_file
            )
            for entry in entries_to_process
        ]
        for future in asyncio.as_completed(tasks):
            try:
                result = await future
                save_data_incrementally(result, args.output_file)
                pbar.update(1)
            except Exception as e:
                print(f"Entry processing error: {e}")
    pbar.close()
    print(f"Done. Output saved to: {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform rejection sampling with generation + scoring.")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSON file with conversations")
    parser.add_argument("--output_file", type=str, required=True, help="Final output JSONL file")
    parser.add_argument("--cache_file", type=str, required=True, help="Intermediate cache JSONL file")
    parser.add_argument("--generator_model", type=str, default="Qwen2.5-VL-32B-Instruct")
    parser.add_argument("--scorer_model", type=str, default="Qwen3-30B-A3B")
    parser.add_argument("--generator_port", type=int, default=31415)
    parser.add_argument("--scorer_port", type=int, default=31416)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--answers_per_question", type=int, default=4)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--request_timeout", type=int, default=180)
    args = parser.parse_args()

    asyncio.run(main(args))