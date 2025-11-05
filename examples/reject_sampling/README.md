# Rejection Sampling Pipeline

This repository provides a complete pipeline for **rejection sampling** with large language models (LLMs), consisting of three stages:

1. **Sampling**: Generate multiple candidate answers per question.
2. **Scoring**: Evaluate each candidate using a judge LLM.
3. **Selection**: (Optional) Select the best answer based on scores.

All scripts support **resumable execution** (skip already processed entries) and **concurrent requests** for efficiency.

---

## 📦 Requirements

- Python ≥ 3.8
- Required packages:
  ```bash
  pip install aiohttp tqdm tenacity jsonlines

## 📦 Usage

1. Generate Candidate Answers (sampling.py)
python sampling.py \
  --input_file data/input.json \
  --output_file data/sampled.jsonl \
  --model_name "your-model-name" \
  --port 31415 \
  --concurrent_requests 10 \
  --answers_per_question 10

```
Input format: JSON list of conversation objects with "conversations" field containing "from" and "value".
```

2. Score Generated Answers (score.py)
python score.py \
  --input_file data/sampled.jsonl \
  --output_file data/scored.jsonl \
  --model_name "judge-model-name" \
  --api_port 31415 \
  --concurrency 10

3. End-to-End Rejection Sampling (reject.py)

This script combines generation and scoring in one step:

python reject.py \
  --input_file data/input.json \
  --output_file data/final_results.jsonl \
  --cache_file data/sampling_cache.jsonl \
  --generator_model "gen-model" \
  --scorer_model "judge-model" \
  --generator_port 31415 \
  --scorer_port 31416 \
  --concurrency 5 \
  --answers_per_question 4

##  Input/Output Formats
Input (input.json)
```json
[
  {
    "conversations": [
      {"from": "human", "value": "What is 2+2?"},
      {"from": "gpt", "value": "4"}
    ],
    "images": ["/path/to/image.jpg"]  // optional
  }
]
```

Output (*.jsonl)
```json
{
  "original_question": "...",
  "original_answers": [...],
  "generated_answers": ["...", "..."],
  "generated_scores": [8.0, 6.5, ...],
  "entry_id": "md5_hash"
}
```

## Notes
* All paths and model names are configurable via CLI.
* The pipeline is resumable: already processed entries (by entry_id) are skipped.
* Set environment variable NO_PROXY=localhost,127.0.0.1 if using local APIs behind a proxy.
