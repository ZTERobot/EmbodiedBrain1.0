import os
import json
import aiohttp
import asyncio
import argparse
import tqdm
from typing import List, Dict, Any
import hashlib

def generate_entry_id(entry: Dict) -> str:
    content = entry["original_question"] + "".join(entry["generated_answers"][:2])
    return hashlib.md5(content.encode()).hexdigest()

def load_existing_data(output_file: str) -> Dict[str, Any]:
    processed_ids = {}
    if not os.path.exists(output_file):
        return processed_ids
    try:
        with open(output_file, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if "entry_id" in entry:
                        processed_ids[entry["entry_id"]] = True
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Error loading existing data: {e}")
    return processed_ids

def save_data_incrementally(data: Dict[str, Any], output_file: str):
    try:
        with open(output_file, 'a') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    except Exception as e:
        print(f"Error saving data: {e}")

def extract_gpt_answer(original_answers: List[Dict]) -> str:
    for answer in original_answers:
        if answer.get("from") == "gpt":
            return answer.get("value", "")
    return ""

def construct_prompt(entry: Dict, answer: str) -> List[Dict]:
    original_question = entry["original_question"]
    original_answers_str = extract_gpt_answer(entry["original_answers"])
    system_prompt = (
        "You are an expert evaluator of LLM responses. Score the generated answer on a scale of 0–10 based on: "
        "instruction following (40%), accuracy (30%), completeness (20%), relevance (10%). "
        "Fluency is a bonus only if all above are satisfied. "
        "Output ONLY an integer between 0 and 10, nothing else."
    )
    user_prompt = (
        f"Original Question:\n{original_question}\n\n"
        f"Original Answer:\n{original_answers_str}\n\n"
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
                "messages": construct_prompt(entry, answer),
                "temperature": 0.1,
                "max_tokens": 10
            }
            async with session.post(api_url, json=payload, timeout=request_timeout) as response:
                if response.status != 200:
                    raise Exception(f"HTTP {response.status}")
                result = await response.json()
                content = result['choices'][0]['message']['content'].strip()
                for word in content.split():
                    try:
                        score = float(word)
                        if 0 <= score <= 10:
                            return score
                    except ValueError:
                        continue
                return 5.0
        except (aiohttp.ClientError, asyncio.TimeoutError, KeyError, json.JSONDecodeError) as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
            else:
                print(f"Scoring failed: {e}")
                return 0.0
    return 0.0

async def process_entry(
    session: aiohttp.ClientSession,
    entry: Dict,
    semaphore: asyncio.Semaphore,
    model_name: str,
    api_url: str,
    max_retries: int,
    request_timeout: int
) -> Dict:
    async with semaphore:
        scores = []
        for answer in entry["generated_answers"]:
            score = await score_answer(session, entry, answer, model_name, api_url, max_retries, request_timeout)
            scores.append(score)
        result = entry.copy()
        result["generated_scores"] = scores
        result["entry_id"] = generate_entry_id(entry)
        return result

async def main(args):
    processed_ids = load_existing_data(args.output_file)
    print(f"Loaded {len(processed_ids)} processed entries.")

    entries_to_process = []
    try:
        with open(args.input_file, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    entry_id = generate_entry_id(entry)
                    if entry_id not in processed_ids:
                        entries_to_process.append(entry)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Error loading input file: {e}")
        return

    print(f"Processing {len(entries_to_process)} entries.")
    semaphore = asyncio.Semaphore(args.concurrency)
    pbar = tqdm.tqdm(total=len(entries_to_process), desc="Scoring")

    api_url = f"http://localhost:{args.api_port}/v1/chat/completions"
    async with aiohttp.ClientSession() as session:
        for i in range(0, len(entries_to_process), args.concurrency):
            batch = entries_to_process[i:i+args.concurrency]
            tasks = [
                process_entry(
                    session, entry, semaphore,
                    args.model_name, api_url,
                    args.max_retries, args.request_timeout
                )
                for entry in batch
            ]
            for future in asyncio.as_completed(tasks):
                try:
                    result = await future
                    save_data_incrementally(result, args.output_file)
                    pbar.update(1)
                except Exception as e:
                    print(f"Error processing entry: {e}")
    pbar.close()
    print(f"Scoring completed. Results saved to: {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score generated answers using an LLM judge.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input JSONL file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output JSONL file")
    parser.add_argument("--model_name", type=str, default="Qwen3-32B", help="Scoring model name")
    parser.add_argument("--api_port", type=int, default=31415, help="Local API port")
    parser.add_argument("--concurrency", type=int, default=10, help="Number of concurrent requests")
    parser.add_argument("--max_retries", type=int, default=3, help="Max retry attempts per request")
    parser.add_argument("--request_timeout", type=int, default=120, help="Request timeout in seconds")
    args = parser.parse_args()

    os.environ['NO_PROXY'] = 'localhost,127.0.0.1'
    asyncio.run(main(args))