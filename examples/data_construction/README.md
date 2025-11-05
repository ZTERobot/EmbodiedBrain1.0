# Data Format Conversion Tool

This tool automatically converts dialogue data containing task plans into a standardized format with **atomic action tags** and a **structured action list**, suitable for robotic instruction understanding, task planning, and related applications.

## 📦 Features

- Automatically annotates each `plan` step with an `[Action]` tag (selected from a predefined list of atomic actions)
- Generates a structured `actions` list (e.g., `[['Navigate', 'Kitchen']]`)
- Supports resumable processing (skips already processed items)
- Uses asynchronous concurrent requests for improved efficiency

## ⚙️ Dependencies

- Python ≥ 3.8  
- Install required packages:
  ```bash
  pip install openai aiohttp tqdm

##  Usage
python data_converter.py \
--input_path ./input.json \
--output_path ./output.json \
--api_key "your-api-key-here" \
--base_url "http://localhost:8000/v1" \
--model_name "Qwen3-30B-A3B" \
--concurrency 10


## 输出格式示例
<response>Okay, I'll get you a cup.</response><plans>1.[Navigate] Go to the kitchen\n2.[Find] Find the cup\n3.[Pick up] Pick up the cup</plans><actions>[['Navigate', 'kitchen'], ['Find', 'cup'], ['Pick up', 'cup']]</actions>