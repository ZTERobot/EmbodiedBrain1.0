import os
import json
import asyncio
import aiohttp
import argparse
import jsonlines
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

def load_processed_prompts(output_file: str):
    processed = set()
    if os.path.exists(output_file):
        try:
            with jsonlines.open(output_file) as reader:
                for item in reader:
                    processed.add(item["original_question"])
        except Exception as e:
            print(f"Error reading output file: {e}")
    return processed

def append_processed_data(output_file: str, item: dict):
    try:
        with jsonlines.open(output_file, mode='a') as writer:
            writer.write(item)
    except Exception as e:
        print(f"Write error: {e}")
        with open(f"{output_file}.bak", "a") as f:
            f.write(json.dumps(item) + "\n")

@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(5),
       retry=retry_if_exception_type(aiohttp.ClientError))
async def get_llm_response(session, prompt_text, model_name, base_url):
    url = f"{base_url}/chat/completions"
    data = {
        "model": model_name,
        "max_tokens": 1024,
        "temperature": 0.9,
        "messages": [
            {"role": "system", "content": "You are an assistant."},
            {"role": "user", "content": [{"type": "text", "text": prompt_text}]}
        ]
    }
    async with session.post(url, json=data) as response:
        response.raise_for_status()
        res = await response.json()
        return res["choices"][0]["message"]["content"]

async def process_conversations(args):
    processed_prompts = load_processed_prompts(args.output_file)
    print(f"Loaded {len(processed_prompts)} processed prompts.")

    if not os.path.exists(args.output_file):
        open(args.output_file, 'w').close()

    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    semaphore = asyncio.Semaphore(args.concurrent_requests)
    base_url = f"http://localhost:{args.port}/v1"

    async with aiohttp.ClientSession() as session:
        tasks = []
        for entry in data:
            human_prompt = next((conv["value"] for conv in entry["conversations"] if conv["from"] == "human"), None)
            if not human_prompt or human_prompt in processed_prompts:
                continue

            async def worker(prompt, orig):
                async with semaphore:
                    answers = []
                    for i in range(args.answers_per_question):
                        try:
                            ans = await get_llm_response(session, prompt, args.model_name, base_url)
                            answers.append(ans)
                        except Exception as e:
                            answers.append(f"Error: {e}")
                    new_entry = {
                        "original_question": prompt,
                        "original_answers": [{"from": c["from"], "value": c["value"]} for c in orig["conversations"]],
                        "generated_answers": answers
                    }
                    append_processed_data(args.output_file, new_entry)

            tasks.append(worker(human_prompt, entry))
        await asyncio.gather(*tasks)
    print("Sampling completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate multiple answers per question using LLM.")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSON file")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSONL file")
    parser.add_argument("--model_name", type=str, default="Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--port", type=int, default=31415)
    parser.add_argument("--concurrent_requests", type=int, default=10)
    parser.add_argument("--answers_per_question", type=int, default=10)
    args = parser.parse_args()

    os.environ['NO_PROXY'] = 'localhost,127.0.0.1'
    asyncio.run(process_conversations(args))