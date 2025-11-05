from datasets import load_dataset
import os

# --- 1. 下载一个公共数据集 (以 nyu-visionx/CV-Bench 为例) ---
print("--- Downloading nyu-visionx/CV-Bench dataset ---")

# 定义你希望数据下载到的本地路径
# 例如，下载到当前脚本所在目录下的 'cv_bench_data' 文件夹
local_download_path = "./cv_bench_data"

# 确保目标文件夹存在，如果不存在则创建
if not os.path.exists(local_download_path):
    os.makedirs(local_download_path)
print(f"Dataset will be downloaded and cached in: {local_download_path}")

dataset_name = "nyu-visionx/CV-Bench"

try:
    # 使用 load_dataset 下载数据集，并通过 cache_dir 参数指定本地路径
    # 默认会下载所有可用的分割（如 'train', 'validation', 'test' 等），并根据数据集脚本的定义来组织
    # 如果你只需要某个特定的分割，可以添加 split='train' 等参数
    print(f"Loading '{dataset_name}' to '{local_download_path}'...")
    
    # 假设你只想下载 default 配置，并且加载所有分割 (根据数据集配置，可能会有多个)
    # 如果数据集有多个配置，你可能需要指定 config_name，例如 load_dataset("nyu-visionx/CV-Bench", "default", cache_dir=local_download_path)
    # 对于 CV-Bench，它默认就是包含 2D 和 3D 任务，无需额外指定 config_name
    cv_bench_dataset = load_dataset(dataset_name, cache_dir=local_download_path)
    
    print(f"'{dataset_name}' dataset loaded successfully.")
    print("Dataset structure:")
    print(cv_bench_dataset) # 这会显示所有下载的分割 (e.g., DatasetDict({'train': Dataset(...), 'validation': Dataset(...)}))

    # 访问其中一个分割的示例
    if 'train' in cv_bench_dataset:
        print(f"\nFirst example from 'train' split:")
        print(cv_bench_dataset['train'][0])
        print(f"Number of examples in 'train' split: {len(cv_bench_dataset['train'])}")
    else:
        print("\n'train' split not found in the loaded dataset.")

except Exception as e:
    print(f"Error downloading dataset '{dataset_name}': {e}")