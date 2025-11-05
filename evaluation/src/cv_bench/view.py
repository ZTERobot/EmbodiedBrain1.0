import json
from collections import defaultdict
import os

def calculate_accuracy_by_task_type(file_path: str):
    """
    Reads a JSONL file, groups data by 'task_type', and calculates
    the accuracy (is_correct == true percentage) for each task type.

    Args:
        file_path (str): The path to the input JSONL file.

    Returns:
        dict: A dictionary where keys are task types and values are their accuracies.
              Also includes overall accuracy.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        return {}

    task_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    overall_total = 0
    overall_correct = 0

    print(f"--- Reading data from: {file_path} ---")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    data = json.loads(line.strip())
                    
                    task_type = data.get("task_type", "unknown_task_type")
                    is_correct = data.get("is_correct", False) # Default to False if not present

                    task_stats[task_type]["total"] += 1
                    overall_total += 1

                    if is_correct:
                        task_stats[task_type]["correct"] += 1
                        overall_correct += 1

                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping malformed JSON on line {line_num + 1}: {e}")
                except Exception as e:
                    print(f"Warning: An error occurred processing line {line_num + 1}: {e}")

        # Calculate accuracies
        accuracies = {}
        print("\n--- Accuracy Results by Task Type ---")
        for task_type, stats in task_stats.items():
            total = stats["total"]
            correct = stats["correct"]
            accuracy = (correct / total) * 100 if total > 0 else 0
            accuracies[task_type] = accuracy
            print(f"Task Type: '{task_type}' | Accuracy: {accuracy:.2f}% ({correct}/{total})")
        
        # Calculate overall accuracy
        overall_accuracy = (overall_correct / overall_total) * 100 if overall_total > 0 else 0
        accuracies["overall"] = overall_accuracy
        print(f"\nOverall Accuracy: {overall_accuracy:.2f}% ({overall_correct}/{overall_total})")

        return accuracies

    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return {}

# --- Configuration ---
jsonl_file_path = "/mnt/tenant-home_speed/ywr/cv_bench/evaluation_results/robobrain2-7B_results.jsonl"

# --- Run the accuracy calculation ---
results = calculate_accuracy_by_task_type(jsonl_file_path)

if results:
    print("\nCalculation complete.")