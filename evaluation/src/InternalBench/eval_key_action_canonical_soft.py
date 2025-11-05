import os
import json
import re
import requests
import numpy as np
import more_itertools as mit
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Any
import argparse
import concurrent.futures # 1️⃣ Add this import

# GPT调用类
class GPT:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        'Authorization': 'xxxxx',
        'Content-Type': 'application/json'
    }
    def __init__(self, model="openai/gpt-5-mini", name="benchmark") -> None:
        self.name = name
        self.model = model
        self.count = 0

    def generate(self, text, temperature=0.1):
        self.count += 1
        try:
            data = {
                "model": self.model,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "user", "content": text}],
                "max_tokens": 4096,
                "temperature": temperature,
                "stream": False
            }
            response = requests.post(self.url, headers=self.headers, data=json.dumps(data), timeout=60)
            response.raise_for_status() # Will raise an exception for HTTP error codes
            result = response.json()
            return json.loads(result['choices'][0]['message']['content'])

        except requests.exceptions.RequestException as e:
            print(f"GPT API request failed: {e}")
            return None
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            print(f"Failed to parse GPT response: {e}")
            # Try to get raw content for debugging
            raw_content = response.text
            print(f"Raw response: {raw_content}")
            return {"error": "Failed to parse response", "raw_content": raw_content}
        except Exception as e:
            print(f"An unexpected error occurred during GPT call: {e}")
            return None

# 每行的第一个词是标准动作，其余是同义词
ACTION_SYNONYM_GROUPS = [
    ["pick", "pick up"],
    ["place", "pick and place"],
    ["handover", "catch"],
    ["drag", "slide"],
    ["find", "search"],
    ["twist", "turn"],
    ["like", "give a thumbs-up"],
    ["heart sign", "make a heart with hands"],
]

# 构建映射表：所有同义词 → 标准动作
ACTION_CANONICAL_MAP = {}
for group in ACTION_SYNONYM_GROUPS:
    canonical = group[0].lower()
    for synonym in group:
        ACTION_CANONICAL_MAP[synonym.lower()] = canonical

ATOMIC_ACTION_SET = {
    "handshake", "clap", "wave", "bow", "dance", "like", "give a thumbs-up",
    "stretch", "handover", "catch", "screw", "unscrew", "peel", "stack",
    "flip", "heart sign", "make a heart with hands", "tear", "pinch",
    "scoop", "cut", "chop", "sweep", "stir", "throw", "slide", "drag", "shake",
    "hammer", "spread", "pour", "point", "pick", "place", "pick up",
    "pick and place", "fold", "unfold", "press", "open", "close", "pull",
    "push", "insert", "grasp and release", "grasp", "release", "lift", "turn",
    "twist", "wash", "rinse", "beat", "hang", "follow", "rotate in place", "turn on", "turn off"
}

#  忽略的动作类型
IGNORE_VERBS = {'find', 'search', 'navigate'}

def build_pred_to_gt_prompt(pred_action, unmatched_gt_actions):
    return f"""
You are an advanced, common-sense evaluator for embodied AI. Your primary mission is to determine if a predicted action aligns with the **high-level intent** of any ground truth (GT) action. Your evaluation should be highly flexible and focus on **what the action accomplishes**, not the exact words used.

---

Predicted Action:
{json.dumps(pred_action, ensure_ascii=False)}

Unmatched GT Actions:
{json.dumps(unmatched_gt_actions, ensure_ascii=False)}

---

Rules for High-Level Intent Matching:
- **Prioritize the Goal**: The most important rule is whether the predicted action achieves the same overall goal as a GT action.
- **Treat Synonyms as Identical**: You **must** treat the following as functionally or semantically identical. Do not reject a match based on these differences.
  - **Objects**: "Mug" and "Cup" are the same type of drinking vessel.
  - **Locations**: "Trash Can," "GarbageCan," and "Waste Bin" all refer to the same type of receptacle.
  - **Actions**: "turn on Faucet" is the same as "turn on Water" because one is the direct means to accomplish the other.
- **Consider a Match a "Reasonable Alternative"**: If the predicted action is a reasonable and common-sense way for a human to perform the GT action, it's a match.
- **Ignore Trivial Differences**: Ignore variations in case (e.g., "SoapBar" vs. "soapbar") and pluralization.
- **Distinguish Opposites**: While synonyms are equivalent, verbs with opposite intents are NOT. For example, "turn on" and "turn off" are distinct intents.

Your Task:
Return a single JSON object with the following format:

If a match is found:
{{
  "matched_gt_index": 2,
  "matched_gt_action": ["verb", "object", "location"],
  "match": true,
  "reasoning": "Explain the match by explicitly stating which components are equivalent. For example: 'This is a match because 'Mug' is a synonym for 'Cup' (object equivalence) and 'place' is a verb match.' or 'This is a match because 'turn on Faucet' is a functional equivalent for 'turn on Water' (action equivalence).'"
}}

If no match is found:
{{
  "match": false,
  "reasoning": "Explain why no match was found. You must explicitly state why the high-level intent differs, citing which component (verb, object, or location) is fundamentally different."
}}
""".strip()


def build_pred_to_gt_prompt_new(pred_action, gt_actions):
    return f"""
You are an advanced, common-sense-enabled embodied intelligence evaluator. Your primary task is to determine whether a **Predicted Action** aligns with the **high-level intent** of any one or more ground truth (GT) actions in the list of **GT Actions**. Your evaluation should be highly flexible and focus on **the goal achieved by the action**, rather than minor differences in wording.

---

Predicted Action:
{json.dumps(pred_action, ensure_ascii=False)}

GT Actions:
{json.dumps(gt_actions, ensure_ascii=False)}

---

Rules for High-Level Intent Matching:
- **Prioritize the Goal**: The most important rule is whether the predicted action achieves the same overall goal as a GT action.
- **Treat Synonyms as Identical**: You **must** treat the following as functionally or semantically identical. Do not reject a match based on these differences.
  - **Objects**: "Mug" and "Cup" are the same type of drinking vessel.
  - **Locations**: "Trash Can," "GarbageCan," and "Waste Bin" all refer to the same type of receptacle.
  - **Actions**: "turn on Faucet" is the same as "turn on Water" because one is the direct means to accomplish the other.
- **Consider a Match a "Reasonable Alternative"**: If the predicted action is a reasonable and common-sense way for a human to perform the GT action, it's a match.
- **Ignore Trivial Differences**: Ignore variations in case (e.g., "SoapBar" vs. "soapbar") and pluralization.
- **Distinguish Opposites**: While synonyms are equivalent, verbs with opposite intents are NOT. For example, "turn on" and "turn off" are distinct intents.
- **Match One or More**: There may be one or more ground truth actions in the list that are equivalent to the predicted action. All equivalent matching items should be identified and output.

Your Task:
Return a single JSON object with the following format:

If a match or multiple matches are found:
```json
{{"result": 
[
{{
  "matched_gt_index": 2,
  "matched_gt_action": ["verb", "object", "location"],  # Ground Truth Action Match
  "matched_pred_action": ["verb", "object", "location"],   # Predicted Action Match
  "match": true,
  "reasoning": "Explain the match by explicitly stating which components are equivalent. For example: 'This is a match because 'Mug' is a synonym for 'Cup' (object equivalence) and 'place' is a verb match.' or 'This is a match because 'turn on Faucet' is a functional equivalent for 'turn on Water' (action equivalence).'"
}},
{{
  "matched_gt_index": 4,
  "matched_gt_action": ["verb", "object"],  # Ground Truth Action Match
  "matched_pred_action": ["verb", "object"],   # Predicted Action Match
  "match": true,
  "reasoning": "Explain the match by explicitly stating which components are equivalent. For example: 'This is a match because 'Mug' is a synonym for 'Cup' (object equivalence) and 'place' is a verb match.' or 'This is a match because 'turn on Faucet' is a functional equivalent for 'turn on Water' (action equivalence).'"
}}
]
}}
```

If no match is found:
```json
{{"result": 
[
{{
  "match": false,
  "reasoning": "Explain why no match was found. You must explicitly state why the high-level intent differs, citing which component (verb, object, or location) is fundamentally different."
}}
]
}}
```
""".strip()



# 只转换同义词和忽略下肢动作，不筛除原子动作集以外的词
def canonicalize_actions(actions: List[List[str]]) -> List[List[str]]:
    """
    Standardizes the verb in each action based on the ACTION_CANONICAL_MAP.
    Does not filter out any actions.
    """
    canonicalized = []
    if not isinstance(actions, list):
        return []
    for action in actions:
        if not action or not isinstance(action, list):
            continue
        verb = action[0].lower()
        # Map synonym to its canonical form
        if verb in IGNORE_VERBS:
            continue
        canonical_verb = ACTION_CANONICAL_MAP.get(verb, verb)
        new_action = [canonical_verb] + action[1:]
        canonicalized.append(new_action)
    return canonicalized


def match_pred_to_gt(pred_actions, gt_actions, gpt):

    response_match_detail = []
    matched_pairs_gt_pred = []
    match_matrix = np.zeros((len(pred_actions), len(gt_actions)))
    for pred_i in range(len(pred_actions)):
        prompt = build_pred_to_gt_prompt_new(pred_actions[pred_i], gt_actions)
        response = gpt.generate(prompt)

        if isinstance(response, str):
            try:
                result = json.loads(response.strip())
            except Exception as e:
                result = {"result": [{"match": False, "reasoning": f"解析失败: {response}"}]}
        elif isinstance(response, dict):
            result = response
        else:
            result = {"result": [{"match": False, "reasoning": f"未知响应类型: {type(response)}"}]}
        
        try:
            # ##TODO(已完成) 使用评估模型输出的 index 参数，后续若出现不准确情况，再遍历循环匹配（检查后发现缺失有问题）
            # for res_item in result["result"]:
            #     if res_item.get("match") is True:
            #         match_matrix[pred_i, res_item["matched_gt_index"]] = 1
            # response_match_detail.append(result)

            history_gt_action = ''
            for res_item in result["result"]:
                if res_item.get("match") is True:
                    matched_pairs_gt_pred.append([res_item.get("matched_gt_action"), res_item.get("matched_pred_action")])
                    if res_item.get("matched_gt_action") == history_gt_action:
                        continue
                    indices = list(mit.locate(gt_actions, pred=lambda x: x == res_item["matched_gt_action"]))
                    for index in indices:
                        match_matrix[pred_i, index] = 1

                history_gt_action = res_item.get("matched_gt_action")
            response_match_detail.append(result)

        except Exception as e:
            response_match_detail.append(f'---- GET GPT response error!! {str(e)} ----')

    return response_match_detail, match_matrix, matched_pairs_gt_pred


#  提取 <actions> 字段中的动作列表
def extract_actions(text: str) -> List[List[str]]:
    try:
        match = re.search(r"<actions>\s*(.*?)\s*</actions>", text, re.DOTALL)
        if not match:
            return []
        actions_str = match.group(1)
        actions = eval(actions_str.strip())
        return actions if isinstance(actions, list) else []
    except Exception:
        return []

## 获取最长公共子序列长度
def lcs_from_match_matrix(match, pred, gt):
    m = len(match)
    if m == 0:
        return 0
    n = len(match[0])

    # 初始化 dp 数组
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if match[i - 1][j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # return dp, dp[m][n]
    i, j = m, n
    lcs = []

    while i > 0 and j > 0:
        if match[i - 1][j - 1]:
            lcs.append([pred[i - 1], gt[j - 1]])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return dp, dp[m][n], list(reversed(lcs))

## 获取最大匹配数量
def count_max_matching(match_matrix):
    """
    输入: match_matrix 是一个二维数组，值为 0 或 1
    输出: 最大一对一匹配的数量
    """
    # 将匹配矩阵转换为代价矩阵（我们要求最大匹配，所以取负）
    cost_matrix = -np.array(match_matrix)

    # 使用匈牙利算法求解最优分配
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # 过滤掉不匹配的情况（即 match_matrix[row][col] == 0 的情况）
    valid_matches = [(r, c) for r, c in zip(row_ind, col_ind) if match_matrix[r][c] == 1]

    return len(valid_matches)


def get_precison_single(matched_num, pred_num):
    return matched_num / pred_num if pred_num != 0 else 0

def get_recall_single(matched_num, gt_num):
    return matched_num / gt_num if gt_num != 0 else 0

def get_f1_single(precison, recall):
    if precison !=0 and recall!=0:
        return 2 * (precison * recall) / (precison + recall)
    else:
        return 0

# 主评估函数
def evaluate_all(json_path: str, output_path: str):

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results_data = {}
    total_matched_presion = 0
    total_matched_recall = 0
    total_matched_f1 = 0

    total_order_presion = 0
    total_order_recall = 0
    total_order_f1 = 0

    total_items = len(data)
    valid_items = 0

    MAX_WORKERS = 4 

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks to the thread pool
        future_to_key = {executor.submit(process_item, key, item, GPT()): key for key, item in data.items()}
       
        # Use tqdm to show progress for the concurrent tasks
        for future in tqdm(concurrent.futures.as_completed(future_to_key), total=len(data), desc="评估中"):
            key, item_result = future.result()
            results_data[key] = item_result

            if "skip" not in item_result:
                # 指标累加
                total_matched_presion += item_result["match_indicator"]["matched_presion"]
                total_matched_recall += item_result["match_indicator"]["matched_recall"]
                total_matched_f1 += item_result["match_indicator"]["matched_f1"]

                total_order_presion += item_result["order_indicator"]["order_presion"]
                total_order_recall += item_result["order_indicator"]["order_recall"]
                total_order_f1 += item_result["order_indicator"]["order_f1"]

                valid_items += 1

    avg_matched_presion = round(total_matched_presion / valid_items, 3) if valid_items else 0
    avg_matched_recall = round(total_matched_recall / valid_items, 3) if valid_items else 0
    avg_matched_f1 = round(total_matched_f1 / valid_items, 3) if valid_items else 0
    avg_order_presion = round(total_order_presion / valid_items, 3) if valid_items else 0
    avg_order_recall = round(total_order_recall / valid_items, 3) if valid_items else 0
    avg_order_f1 = round(total_order_f1 / valid_items, 3) if valid_items else 0

    print(f"✅ 平均 动作匹配 精确率: {avg_matched_presion}（共评估 {valid_items} 条）")
    print(f"✅ 平均 动作匹配 召回率: {avg_matched_recall}（共评估 {valid_items} 条）")
    print(f"✅ 平均 动作匹配 F1: {avg_matched_f1}（共评估 {valid_items} 条）")

    print(f"✅ 平均 动作顺序匹配 精确率: {avg_order_presion}（共评估 {valid_items} 条）")
    print(f"✅ 平均 动作顺序匹配 召回率: {avg_order_recall}（共评估 {valid_items} 条）")
    print(f"✅ 平均 动作顺序匹配 F1: {avg_order_f1}（共评估 {valid_items} 条）")

    results_data["__benchmark_summary__"] = {
        "judge_model": "GPT-5-mini",
        "valid_items": valid_items,
        "total_items": total_items,
        "average_matched_indicator": {
                "average_matched_presion": avg_matched_presion,
                "average_matched_recall": avg_matched_recall,
                "average_matched_f1": avg_matched_f1},
        "average_oeder_indicator": {
                "average_order_presion": avg_order_presion,
                "average_order_recall": avg_order_recall,
                "average_order_f1": avg_order_f1},
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)

def process_item(key, item, gpt):
    # for key in tqdm(data.keys(), desc="评估中"):
    #     if key == "__benchmark_summary__":
    #         total_items -= 1
    #         continue
    if not key.isdigit():
        return key, {"skip": True}

    gt_actions_raw = extract_actions(item.get("real_answer", ""))
    pred_actions_raw = extract_actions(item.get("model_answer", ""))

    if not pred_actions_raw:
        res = {}
        res["evaluation_details"] = "预测-动作提取失败."
        res["match_indicator"] ={"matched_presion": 0, "matched_recall": 0, "matched_f1": 0}
        res["order_indicator"] = { "order_presion": 0, "order_recall": 0, "order_f1": 0}
        return key, res
    if not gt_actions_raw:
        item["evaluation_details"] = "标准-动作提取失败"
        res["match_indicator"] ={"matched_presion": 0, "matched_recall": 0, "matched_f1": 0}
        res["order_indicator"] = { "order_presion": 0, "order_recall": 0, "order_f1": 0}
        res["skip"] = True
        return key, res

    gt_actions = canonicalize_actions(gt_actions_raw)
    pred_actions = canonicalize_actions(pred_actions_raw)

    response_match_detail, match_matrix, matched_pairs_gt_pred = match_pred_to_gt(pred_actions, gt_actions, gpt)

    matched_num = count_max_matching(match_matrix)
    dp_matrix, dp_num, dp_matched_list = lcs_from_match_matrix(match_matrix, pred_actions, gt_actions)

    # 计算单一动作匹配指标
    item_matched_presion = get_precison_single(matched_num, len(pred_actions))
    item_matched_recal = get_recall_single(matched_num, len(gt_actions))
    item_matched_f1 = get_f1_single(item_matched_presion, item_matched_recal)

    # 计算整体动作顺序匹配指标
    item_order_presion = get_precison_single(dp_num, len(pred_actions))
    item_order_recal = get_recall_single(dp_num, len(gt_actions))
    item_order_f1 = get_f1_single(item_order_presion, item_order_recal)

    # ✅ 写入日志字段
    item["gt_canonical_actions"] = gt_actions
    item["pred_canonical_actions"] = pred_actions
    item["match_indicator"] = {
        "matched_presion": round(item_matched_presion, 3),
        "matched_recall": round(item_matched_recal, 3),
        "matched_f1": round(item_matched_f1, 3)}
    item["order_indicator"] = {
        "order_presion": round(item_order_presion, 3),
        "order_recall": round(item_order_recal, 3),
        "order_f1": round(item_order_f1, 3)}
    item["matched_pairs_gt_pred"] = matched_pairs_gt_pred
    item["order_matched_list"] = dp_matched_list
    item["response_match_detail"] = response_match_detail
    return key, item


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate planning accuracy using GPT-4o-mini.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input JSON file.")
    parser.add_argument("--output_file", type=str, default=None, help="Path to output JSON log file.")

    args = parser.parse_args()
    input_file = args.input_file
    output_file = args.output_file or input_file.replace('inference', 'soft_multi_turn_eval_log')

    # input_file = '/home/10311584@zte.intra/文档/5-工作相关/01_2025/3-具身智能/04_具身规划/002_中间态数据bench建设/mid_bench_gpt_0919_1/code/0924_v1/test_json_file/test.json'
    # output_file = '/home/10311584@zte.intra/文档/5-工作相关/01_2025/3-具身智能/04_具身规划/002_中间态数据bench建设/mid_bench_gpt_0919_1/code/0924_v1/test_json_file/test_res_4.json'

    evaluate_all(input_file, output_file)