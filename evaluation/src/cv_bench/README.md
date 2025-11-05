---
license: apache-2.0
task_categories:
- visual-question-answering
language:
- en
pretty_name: Cambrian Vision-Centric Benchmark (CV-Bench)
configs:
- config_name: default
  data_files:
  - split: test
    path: "test*.parquet"
- config_name: 2D
  data_files:
  - split: test
    path: "test_2d.parquet"
- config_name: 3D
  data_files:
  - split: test
    path: "test_3d.parquet"
---

<p>
    <a href="https://arxiv.org/abs/2406.16860" target="_blank" style="display: inline-block; margin-right: 10px;">
        <img alt="arXiv" src="https://img.shields.io/badge/arXiv-Cambrian--1-red?logo=arxiv" />
    </a>
    <a href="https://cambrian-mllm.github.io/" target="_blank" style="display: inline-block; margin-right: 10px;">
        <img alt="Website" src="https://img.shields.io/badge/🌎_Website-cambrian--mllm.github.io-blue.svg" />
    </a>
    <a href="https://github.com/cambrian-mllm/cambrian" target="_blank" style="display: inline-block; margin-right: 10px;">
        <img alt="GitHub Code" src="https://img.shields.io/badge/Code-cambrian--mllm/cambrian-white?&logo=github&logoColor=white" />
    </a>
    <a href="https://huggingface.co/collections/nyu-visionx/cambrian-1-models-666fa7116d5420e514b0f23c" target="_blank" style="display: inline-block; margin-right: 10px;">
        <img alt="Hugging Face" src="https://img.shields.io/badge/🤗_Model-Cambrian--1-ffc107?color=ffc107&logoColor=white" />
    </a>
    <a href="https://huggingface.co/collections/nyu-visionx/cambrian-data-6667ce801e179b4fbe774e11" target="_blank" style="display: inline-block; margin-right: 10px;">
        <img alt="Hugging Face" src="https://img.shields.io/badge/🤗_Data-Cambrian--10M-ffc107?color=ffc107&logoColor=white" />
    </a>
</p>


# Cambrian Vision-Centric Benchmark (CV-Bench)

This repository contains the Cambrian Vision-Centric Benchmark (CV-Bench), introduced in [Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs](https://arxiv.org/pdf/2406.16860).


## Files
The `test*.parquet` files contain the dataset annotations and images pre-loaded for processing with HF Datasets.

These can be loaded in 3 different configurations using `datasets` as follows:

```python
from datasets import load_dataset

# default: both 2D and 3D tasks
cv_bench = load_dataset("nyu-visionx/CV-Bench")

# 2D tasks only
cv_bench_2d = load_dataset("nyu-visionx/CV-Bench", "2D")

# 3D tasks only
cv_bench_3d = load_dataset("nyu-visionx/CV-Bench", "3D")
```

Additionally, we provide the raw images and annotations separately.

- `test_2d.jsonl`: 2D text annotations
- `test_3d.jsonl`: 3D text annotations
- `img/` dir: images corresponding to the `filename` field in the annotations


## Dataset Description

CV-Bench addresses the limited size of existing vision-centric benchmarks, containing `2638` *manually-inspected* examples. By repurposing standard vision benchmarks, `ADE20k`, `COCO` and `OMNI3D`, we assess models at classic vision tasks within a multimodal context. Leveraging the rich ground truth annotations from the benchmarks, we formulate natural language questions that probe the fundamental 2D and 3D understanding of the models. CV-Bench evaluates 2D understanding via spatial relationships & object counting, and 3D understanding via depth order & relative distance.

The dataset contains the following fields:

| Field Name | Description |
| :--------- | :---------- |
| `idx` | Global index of the entry in the dataset |
| `type` | Type of task: `2D` or `3D` |
| `task` | The task associated with the entry |
| `image` | Image object |
| `question` | Question asked about the image |
| `choices` | Answer choices for the question |
| `answer` | Correct answer to the question |
| `prompt` | Prompt with question and choices pre-formatted |
| `filename` | Path to the image in the `img/` directory |
| `source` | Source of the image: `ADE20K`, `COCO`, or `Omni3D` |
| `source_dataset` | More detailed source of the image |
| `source_filename` | Filename of the image in the source dataset |
| `target_class` | Target class of the image (only for `COCO` images) |
| `target_size` | Target size of the image (only for `COCO` images) |
| `bbox` | Bounding box of the image (only for `Omni3D` images) |


<br>

## Accuracy


We calculate the accuracy for each task and compute a combined accuracy as specified in the following formula:

$$\text{CV-Bench Accuracy} = \frac 1 2 \left( \frac{\text{accuracy}_{2D_{ade}} + \text{accuracy}_{2D_{coco}}}{2} + \text{accuracy}_{3D_{omni}} \right)$$

### Example Code

```python
import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('cv_bench_results.csv')

# Define a function to calculate accuracy for a given source
def calculate_accuracy(df, source):
    source_df = df[df['source'] == source]
    accuracy = source_df['result'].mean()  # Assuming 'result' is 1 for correct and 0 for incorrect
    return accuracy

# Calculate accuracy for each source
accuracy_2d_ade = calculate_accuracy(df, 'ADE20K')
accuracy_2d_coco = calculate_accuracy(df, 'COCO')
accuracy_3d_omni = calculate_accuracy(df, 'Omni3D')

# Calculate the accuracy for each type
accuracy_2d = (accuracy_2d_ade + accuracy_2d_coco) / 2
accuracy_3d = accuracy_3d_omni

# Compute the combined accuracy as specified
combined_accuracy = (accuracy_2d + accuracy_3d) / 2

# Print the results
print(f"CV-Bench Accuracy: {combined_accuracy:.4f}")
print()
print(f"Type Accuracies:")
print(f"2D Accuracy: {accuracy_2d:.4f}")
print(f"3D Accuracy: {accuracy_3d:.4f}")
print()
print(f"Source Accuracies:")
print(f"ADE20K Accuracy: {accuracy_2d_ade:.4f}")
print(f"COCO Accuracy: {accuracy_2d_coco:.4f}")
print(f"Omni3D Accuracy: {accuracy_3d_omni:.4f}")
```

## Citation

```bibtex
@misc{tong2024cambrian1,
      title={Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs},
      author={Shengbang Tong and Ellis Brown and Penghao Wu and Sanghyun Woo and Manoj Middepogu and Sai Charitha Akula and Jihan Yang and Shusheng Yang and Adithya Iyer and Xichen Pan and Austin Wang and Rob Fergus and Yann LeCun and Saining Xie},
      year={2024},
      eprint={2406.16860},
}
```
