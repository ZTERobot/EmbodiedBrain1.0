import asyncio
import re
import textwrap
from copy import deepcopy
from typing import Dict, List, Optional
from typing import List, Dict, Tuple, Union


import json
import torch

from swift.llm import PtEngine, RequestConfig, Template, to_device
from swift.llm.infer.protocol import ChatCompletionResponse
from swift.plugin import ORM, orms, rm_plugins
from swift.plugin.rm_plugin import DefaultRMPlugin
from swift.utils import get_logger

import ast
from word2number import w2n
import regex as regext
from collections import defaultdict


logger = get_logger()
"""
Step 1: Define a Reward Class
    Implement your custom reward calculation logic within the __call__ method.
    The method accepts the model's output completions and dataset columns (passed as kwargs) as input parameters.

Step 2: Register the Reward Class in orms
    For example:
    python orms['external_math_acc'] = MathAccuracy

Step 3: Configure the Arguments
    Use the following arguments when running the script:
    bash --plugin /path/to/plugin.py --reward_funcs external_math_acc
"""


# Code borrowed from plugin/orm.py
class MathAccuracy(ORM):

    def __init__(self):
        import importlib.util
        assert importlib.util.find_spec('math_verify') is not None, (
            "The math_verify package is required but not installed. Please install it using 'pip install math_verify'.")

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        from latex2sympy2_extended import NormalizationConfig
        from math_verify import LatexExtractionConfig, parse, verify
        rewards = []
        for content, sol in zip(completions, solution):
            gold_parsed = parse(sol, extraction_mode='first_match', extraction_config=[LatexExtractionConfig()])
            if len(gold_parsed) != 0:
                # We require the answer to be provided in correct latex (no malformed operators)
                answer_parsed = parse(
                    content,
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(
                                nits=False,
                                malformed_operators=False,
                                basic_latex=True,
                                equations=True,
                                boxed=True,
                                units=True,
                            ),
                            # Ensures that boxed is tried first
                            boxed_match_priority=0,
                            try_extract_without_anchor=False,
                        )
                    ],
                    extraction_mode='first_match',
                )
                # Reward 1 if the content is the same as the ground truth, 0 otherwise
                reward = float(verify(answer_parsed, gold_parsed))
            else:
                # If the gold solution is not parseable, we reward 1 to skip this example
                reward = 1.0
            rewards.append(reward)
        return rewards


class MathFormat(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r'^<think>.*?</think>\s*<answer>.*?</answer>(?![\s\S])'
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        return [1.0 if match else 0.0 for match in matches]


class CountdownORM(ORM):

    def __call__(self, completions, target, nums, **kwargs) -> List[float]:
        """
        Evaluates completions based on Mathematical correctness of the answer

        Args:
            completions (list[str]): Generated outputs
            target (list[str]): Expected answers
            nums (list[str]): Available numbers

        Returns:
            list[float]: Reward scores
        """
        rewards = []
        for completion, gt, numbers in zip(completions, target, nums):
            try:
                # Check if the format is correct
                match = re.search(r'<answer>(.*?)<\/answer>', completion)
                if match is None:
                    rewards.append(0.0)
                    continue
                # Extract the "answer" part from the completion
                equation = match.group(1).strip()
                if '=' in equation:
                    equation = equation.split('=')[0]
                # Extract all numbers from the equation
                used_numbers = [int(n) for n in re.findall(r'\d+', equation)]

                # Check if all numbers are used exactly once
                if sorted(used_numbers) != sorted(numbers):
                    rewards.append(0.0)
                    continue
                # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
                allowed_pattern = r'^[\d+\-*/().\s]+$'
                if not re.match(allowed_pattern, equation):
                    rewards.append(0.0)
                    continue

                # Evaluate the equation with restricted globals and locals
                result = eval(equation, {"__builti'ns__": None}, {})
                # Check if the equation is correct and matches the ground truth
                if abs(float(result) - float(gt)) < 1e-5:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            except Exception:
                # If evaluation fails, reward is 0
                rewards.append(0.0)
        return rewards


class MultiModalAccuracyORM(ORM):

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """
        Reward function that checks if the completion is correct.
        Args:
            completions (list[str]): Generated outputs
            solution (list[str]): Ground Truths.

        Returns:
            list[float]: Reward scores
        """
        rewards = []
        from math_verify import parse, verify
        for content, sol in zip(completions, solution):
            reward = 0.0
            # Try symbolic verification first
            try:
                answer = parse(content)
                if float(verify(answer, parse(sol))) > 0:
                    reward = 1.0
            except Exception:
                pass  # Continue to next verification method if this fails

            # If symbolic verification failed, try string matching
            if reward == 0.0:
                try:
                    # Extract answer from solution if it has think/answer tags
                    sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                    ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

                    # Extract answer from content if it has think/answer tags
                    content_match = re.search(r'<answer>(.*?)</answer>', content)
                    student_answer = content_match.group(1).strip() if content_match else content.strip()

                    # Compare the extracted answers
                    if student_answer == ground_truth:
                        reward = 1.0
                except Exception:
                    pass  # Keep reward as 0.0 if both methods fail
            print("Reward: ", reward)
            rewards.append(reward)
        return rewards


# ref implementation: https://github.com/huggingface/open-r1/blob/main/src/open_r1/rewards.py
class CodeReward(ORM):

    def __init__(self):
        import importlib.util
        assert importlib.util.find_spec('e2b') is not None, (
            "The e2b package is required but not installed. Please install it using 'pip install e2b-code-interpreter'."
        )
        from dotenv import load_dotenv
        load_dotenv()

    @staticmethod
    def extract_code(completion: str, language: str) -> str:
        pattern = re.compile(rf'```{language}\n(.*?)```', re.DOTALL)
        matches = pattern.findall(completion)
        extracted_answer = matches[-1] if len(matches) >= 1 else ''
        return extracted_answer

    def run_async_from_sync(self, scripts: List[str], languages: List[str]) -> List[float]:
        """Function wrapping the `run_async` function."""
        # Create a new event loop and set it
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Run the async function and get the result
            rewards = loop.run_until_complete(self.run_async(scripts, languages))
        finally:
            loop.close()

        return rewards

    async def run_async(self, scripts: List[str], languages: List[str]) -> List[float]:
        from e2b_code_interpreter import AsyncSandbox

        # Create the sandbox by hand, currently there's no context manager for this version
        try:
            sbx = await AsyncSandbox.create(timeout=30, request_timeout=3)
        except Exception as e:
            logger.warning(f'Error from E2B executor: {e}')
            return [0.0] * len(scripts)
        # Create a list of tasks for running scripts concurrently
        tasks = [self.run_script(sbx, script, language) for script, language in zip(scripts, languages)]

        # Wait for all tasks to complete and gather their results as they finish
        results = await asyncio.gather(*tasks)
        rewards = list(results)  # collect results

        # Kill the sandbox after all the tasks are complete
        await sbx.kill()

        return rewards

    async def run_script(self, sbx, script: str, language: str) -> float:
        try:
            execution = await sbx.run_code(script, language=language, timeout=30)
        except Exception as e:
            logger.warning(f'Error from E2B executor: {e}')
            return 0.0
        try:
            return float(execution.text)
        except (TypeError, ValueError):
            return 0.0

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that evaluates code snippets using the E2B code interpreter.

        Assumes the dataset contains a `verification_info` column with test cases.
        """
        evaluation_script_template = """
        import subprocess
        import json

        def evaluate_code(code, test_cases):
            passed = 0
            total = len(test_cases)
            exec_timeout = 5

            for case in test_cases:
                process = subprocess.run(
                    ["python3", "-c", code],
                    input=case["input"],
                    text=True,
                    capture_output=True,
                    timeout=exec_timeout
                )

                if process.returncode != 0:  # Error in execution
                    continue

                output = process.stdout.strip()
                if output.strip() == case["output"].strip():
                    passed += 1

            success_rate = (passed / total)
            return success_rate

        code_snippet = {code}
        test_cases = json.loads({test_cases})

        evaluate_code(code_snippet, test_cases)
        """
        verification_info = kwargs['verification_info']
        languages = [info['language'] for info in verification_info]
        code_snippets = [
            self.extract_code(completion, language) for completion, language in zip(completions, languages)
        ]
        scripts = [
            evaluation_script_template.format(
                code=json.dumps(code), test_cases=json.dumps(json.dumps(info['test_cases'])))
            for code, info in zip(code_snippets, verification_info)
        ]
        try:
            rewards = self.run_async_from_sync(scripts, languages)

        except Exception as e:
            logger.warning(f'Error from E2B executor: {e}')
            rewards = [0.0] * len(completions)

        return rewards


class CodeFormat(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        verification_info = kwargs['verification_info']
        rewards = []
        for content, info in zip(completions, verification_info):
            pattern = r'^<think>.*?</think>\s*<answer>.*?```{}.*?```.*?</answer>(?![\s\S])'.format(info['language'])
            match = re.match(pattern, content, re.DOTALL | re.MULTILINE)
            reward = 1.0 if match else 0.0
            rewards.append(reward)
        return rewards


class CodeRewardByJudge0(ORM):
    LANGUAGE_ID_MAP = {
        'assembly': 45,
        'bash': 46,
        'basic': 47,
        'c': 50,
        'c++': 54,
        'clojure': 86,
        'c#': 51,
        'cobol': 77,
        'common lisp': 55,
        'd': 56,
        'elixir': 57,
        'erlang': 58,
        'executable': 44,
        'f#': 87,
        'fortran': 59,
        'go': 60,
        'groovy': 88,
        'haskell': 61,
        'java': 62,
        'javascript': 63,
        'kotlin': 78,
        'lua': 64,
        'multi-file program': 89,
        'objective-c': 79,
        'ocaml': 65,
        'octave': 66,
        'pascal': 67,
        'perl': 85,
        'php': 68,
        'plain text': 43,
        'prolog': 69,
        'python': 71,
        'python2': 70,
        'python3': 71,
        'r': 80,
        'ruby': 72,
        'rust': 73,
        'scala': 81,
        'sql': 82,
        'swift': 83,
        'typescript': 74,
        'visual basic.net': 84
    }
    PYTHON_ID = 71

    def __init__(self):
        import os
        self.endpoint = os.getenv('JUDGE0_ENDPOINT')
        assert self.endpoint is not None, (
            'Judge0 endpoint is not set. Please set the JUDGE0_ENDPOINT environment variable.')
        x_auth_token = os.getenv('JUDGE0_X_AUTH_TOKEN')
        self.headers = {'Content-Type': 'application/json'}
        if x_auth_token is not None:
            self.headers['X-Auth-Token'] = x_auth_token

    @staticmethod
    def extract_code(completion: str, language: str) -> str:
        pattern = re.compile(rf'```{language}\n(.*?)```', re.DOTALL)
        matches = pattern.findall(completion)
        extracted_answer = matches[-1] if len(matches) >= 1 else ''
        return extracted_answer

    @classmethod
    def get_language_id(cls, language):
        if language is None:
            return cls.PYTHON_ID
        return cls.LANGUAGE_ID_MAP.get(language.lower().strip(), cls.PYTHON_ID)

    async def _evaluate_code(self, code, test_cases, language_id):
        import aiohttp
        try:
            passed = 0
            total = len(test_cases)

            for case in test_cases:
                if code is not None and code != '':
                    async with aiohttp.ClientSession() as session:
                        payload = {
                            'source_code': code,
                            'language_id': language_id,
                            'stdin': case['input'],
                            'expected_output': case['output']
                        }
                        logger.debug(f'Payload: {payload}')
                        async with session.post(
                                self.endpoint + '/submissions/?wait=true', json=payload,
                                headers=self.headers) as response:
                            response_json = await response.json()
                            logger.debug(f'Response: {response_json}')
                            if response_json['status']['description'] == 'Accepted':
                                passed += 1

            success_rate = (passed / total)
            return success_rate
        except Exception as e:
            logger.warning(f'Error from Judge0 executor: {e}')
            return 0.0

    def run_async_from_sync(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            rewards = loop.run_until_complete(self.run_async())
        finally:
            loop.close()
        return rewards

    async def run_async(self):
        tasks = [
            self._evaluate_code(code, info['test_cases'], CodeRewardByJudge0.get_language_id(info['language']))
            for code, info in zip(self.code_snippets, self.verification_info)
        ]
        results = await asyncio.gather(*tasks)
        rewards = list(results)
        return rewards

    def __call__(self, completions, **kwargs) -> List[float]:
        self.verification_info = kwargs['verification_info']

        languages = [info['language'] for info in self.verification_info]
        self.code_snippets = [
            self.extract_code(completion, language) for completion, language in zip(completions, languages)
        ]

        try:
            rewards = self.run_async_from_sync()
        except Exception as e:
            logger.warning(f'Error from Judge0 executor: {e}')
            rewards = [0.0] * len(completions)
        return rewards


import re


class VisualReasoningAccuracy(ORM):
    def __call__(self,
                 completions,
                 ground_truth: List[str],  # ✅ 更明确地命名参数为 ground_truth
                 # answer: List[str] = None,  # 可选 fallback 答案字段
                 **kwargs) -> List[float]:
        rewards = []
        for content, gt in zip(completions, ground_truth):
            pred = self.extract_answer(content)
            correct = self.extract_answer(gt)
            print(f"pred = {pred}, correct = {correct}")


            if pred and correct and pred.lower() == correct.lower():
                reward = 1.0
            else:
                reward = 0.0
            rewards.append(reward)
        return rewards

    @staticmethod
    def extract_answer(content: str) -> str:
        """
        从内容中提取答案，优先匹配 \\boxed{}，否则尝试提取字母或数字
        """
        if not isinstance(content, str):
            return ""

        boxed = re.search(r'\\boxed\{(.*?)\}', content)
        if boxed:
            return boxed.group(1).strip()

        match = re.search(r'\b[A-Da-d]\b|\b\d+\b', content)
        if match:
            return match.group(0).strip()

        return ""


import logging
import re
import json
import logging
from typing import List, Dict, Union, Optional, Any
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


def is_valid(result, task_type):
    """判断提取出的内容是否符合对应任务类型的要求"""
    if result is None:
        return False

    if task_type in ['cv_detection', 'cv_grounding']:
        # 检查是否为包含 bbox_2d 字段的字典列表
        return (
            isinstance(result, list) and len(result) > 0 and all(
                isinstance(box, dict) and 'bbox_2d' in box and
                isinstance(box['bbox_2d'], list) and len(box['bbox_2d']) == 4 and
                all(isinstance(coord, (int, float)) for coord in box['bbox_2d'])
                for box in result
            )
        )

    elif task_type == 'ocr':
        # OCR 要求是非空字符串
        return isinstance(result, str) and len(result.strip()) > 0

    elif task_type == 'counting':
        # Counting 要求是整数或者可转为整数的字符串
        if isinstance(result, int):
            return True
        elif isinstance(result, str):
            return result.strip().isdigit()
        return False

    elif task_type == 'math':
        # Math 要求是字符串且非空
        return isinstance(result, str) and len(result.strip()) > 0

    else:
        # 不支持的任务类型直接返回 False
        return False


import re
import json
from typing import Union, Optional, List, Dict
from difflib import SequenceMatcher
from sympy import sympify, Eq
# from sympy.core.exceptions import SympifyError
# from sympy.parsing.sympy_parser import SympifyError
from sympy import SympifyError
import ast

class VisualPerceptionAccuracy(ORM):
    def __init__(self):
        self.iou_threshold = 0.2
        self.ignore_case = True
        self.fuzzy_match_ratio = 0.5
        self.match_valid_only = False

    def __call__(self,
                 completions, solution, task,
                 **kwargs) -> List[float]:
        TASK_TYPES = ['cv_detection', 'cv_grounding', 'ocr', 'counting', 'math']

        def _get_task_type(content):
            for task_type in TASK_TYPES:
                result = self.extract_answer(content, task_type)
                if is_valid(result, task_type):
                    return task_type
            return None

        rewards = []

        for idx, (content, gt, task_type) in enumerate(zip(completions, solution, task)):
            if task_type in ["counting", "detection", "grounding"]:
                try:
                    # Step 1: 尝试从 ground_truth 中识别任务类型
                    task_type = _get_task_type(gt)

                    if task_type is None:
                        print(f"Sample {idx}: No valid task type detected.")
                        rewards.append(0.0)
                        continue

                    # print("task is: ", task_type)
                    # Step 4: 从 completion 中提取指定任务类型的内容
                    pred = self.extract_answer(content, task_type)
                    correct = self.extract_answer(gt, task_type)

                    # print(f"pred = {pred}, correct = {correct}")

                    # Step 6: 计算奖励分数
                    if is_valid(pred, task_type) and is_valid(correct, task_type):
                        reward = self.calculate_reward(pred, correct, task_type)
                    else:
                        print(f"Sample {idx}: Missing valid prediction or ground truth.")
                        reward = 0.0

                    rewards.append(reward)

                except Exception as e:
                    print(f"Error processing sample {idx}: {str(e)}")
                    rewards.append(0.0)
            else:
                rewards.append(None)
        return rewards

    @staticmethod
    def extract_answer(content: Union[str, dict, list], task_type: str) -> Union[List[Dict], str, int, None]:
        if isinstance(content, str):
            content_str = content
        else:
            try:
                content_str = json.dumps(content)
            except Exception as e:
                print(f"Failed to convert content to string: {str(e)}")
                return ""

        if task_type in ['cv_grounding', 'cv_detection']:
            return VisualPerceptionAccuracy._extract_bbox(content_str)
        elif task_type == 'ocr':
            return VisualPerceptionAccuracy._extract_ocr(content_str)
        elif task_type == 'counting':
            return VisualPerceptionAccuracy._extract_counting(content_str)
        elif task_type == 'math':
            return VisualPerceptionAccuracy._extract_math(content_str)
        else:
            print(f"Unknown task type: {task_type}")
            return ""

    @staticmethod
    def _extract_bbox(content: str) -> List[Dict]:
        """从字符串中提取边界框信息"""
        try:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

        match = re.search(r'<answer>\s*(.*?)\s*</answer>', content, re.DOTALL)
        if match:
            raw_content = match.group(1).strip()
            try:
                parsed = ast.literal_eval(raw_content)
                if isinstance(parsed, list):
                    return parsed
            except (SyntaxError, ValueError) as e:
                print(f"解析失败: {e}")
                return []
        return []

    @staticmethod
    def _extract_ocr(content: str) -> str:
        """
        按照指定优先级提取 OCR 答案：
        1. 提取 \boxed{...} 中的完整 LaTeX 公式；
        2. 提取 <answer>...</answer> 内容；
        3. 判断是否有数字（阿拉伯或英文），有则转为阿拉伯数字字符串；
        4. 否则返回提取内容的小写形式。
        """

        # Step 1: 尝试提取 \boxed{...} 内容（支持嵌套）
        boxed_match = regext.search(r"\\boxed$\s*({(?:[^{}]|(?R)|\s)*})\s*$", content, re.DOTALL)
        if boxed_match:
            return boxed_match.group(1).strip()

        # Step 2: 尝试提取 <answer> 标签内容
        answer_match = re.search(r"<answer.*?>([\s\S]*?)</answer>", content, re.DOTALL)
        if answer_match:
            answer_text = answer_match.group(1).strip()
        else:
            answer_text = content.strip()

        # Step 3: 判断是否有阿拉伯数字或英文数字
        digit_match = re.search(r"\d+", answer_text)
        words_match = re.findall(
            r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|"
            r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
            r"eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|"
            r"eighty|ninety|hundred|thousand)\b",
            answer_text, re.IGNORECASE
        )

        if digit_match or words_match:
            # Step 4: 存在数字，提取并转换为阿拉伯数字字符串
            digit_str = VisualPerceptionAccuracy.extract_number(answer_text)
            if digit_str:
                return digit_str

        # Step 5: 没有发现数字，返回小写形式
        return answer_text.lower()

    @staticmethod
    def extract_number(content: str) -> Optional[str]:
        """
        从 content 中提取第一个出现的数字，支持：
        - 阿拉伯数字（如 "89"）
        - 英文数字（如 "three", "five"）
        返回提取到的数字字符串，未找到则返回 None
        """
        # 尝试匹配阿拉伯数字
        digit_match = re.search(r"\d+", content)
        if digit_match:
            return digit_match.group(0)

        # 尝试匹配英文数字
        words = re.findall(
            r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|"
            r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
            r"eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|"
            r"eighty|ninety|hundred|thousand)\b",
            content, re.IGNORECASE
        )
        if not words:
            return None

        try:
            return str(w2n.word_to_num(' '.join(words)))
        except Exception:
            return None

    @staticmethod
    def _extract_counting(content: str) -> Optional[int]:
        """从字符串中提取计数结果（整数）"""
        # match = re.search(r'<answer>\s*(.*?)\s*</answer>', content, re.DOTALL)
        # if match:
        #     ans = match.group(1).strip()
        #     if ans.isdigit():
        #         return int(ans)
        #
        # patterns = [
        #     r"(?i)the\s+number\s+is\s+(\d+)",
        #     r"(?i)there\s+are\s+(\d+)\s",
        #     r"(?i)count:\s*(\d+)"
        # ]
        # for pattern in patterns:
        #     match = re.search(pattern, content)
        #     if match and match.group(1).isdigit():
        #         return int(match.group(1))
        #
        # digit_match = re.search(r'\d+', content)
        # if digit_match:
        #     return int(digit_match.group(0))
        answer_match = re.search(r"<answer.*?>([\s\S]*?)</answer>", content, re.DOTALL)
        if answer_match:
            answer_text = answer_match.group(1).strip()

            # 从文本中提取第一个数字
            number_match = re.search(r"\d+", answer_text)
            if number_match:
                return int(number_match.group(0))

        return None

    @staticmethod
    def _extract_math(content: str) -> str:
        """LaTeX 公式提取"""
        print("latex 提取中...")
        match = re.search(r'<answer>\s*(.*?)\s*</answer>', content, re.DOTALL)
        if match:
            return match.group(1).strip()

        boxed_match = re.search(r'\\boxed\{([^}]+)\}', content)
        if boxed_match:
            return boxed_match.group(1).strip()

        return content.strip()

    def calculate_reward(self,
                         preds: Union[List[Dict], str, int, None],
                         gts: Union[List[Dict], str, int, None],
                         task_type: str) -> float:
        """
        根据预测结果和真实值计算奖励。
        """
        if preds is None or gts is None:
            return 0.0

        if task_type in ['cv_grounding', 'cv_detection']:
            return self._calculate_bbox_reward(preds, gts)
        elif task_type == 'ocr':
            return self._calculate_ocr_reward(preds, gts)
        elif task_type == 'counting':
            return self._calculate_counting_reward(preds, gts)
        elif task_type == 'math':
            return self._calculate_math_reward(preds, gts)
        else:
            return 0.0

    def _calculate_bbox_reward(self, preds: List[Dict], gts: List[Dict]) -> float:
        # Step 1: 按照 label 分组
        pred_by_label = defaultdict(list)
        gt_by_label = defaultdict(list)

        for pred in preds:
            pred_by_label[pred['label']].append(pred)
        
        for gt in gts:
            gt_by_label[gt['label']].append(gt)
        
        total_reward = 0.0
        total_weight = 0.0

        # Step 2: 对每一类单独计算 IoU 奖励
        for label in gt_by_label:
            if label not in pred_by_label:
                print(f"类别 {label} 没有预测结果")
                continue
            
            label_preds = pred_by_label[label]
            label_gts = gt_by_label[label]

            total_iou, total_weight_for_label = self._match_and_calculate_weighted_iou(label_preds, label_gts)
            # num_gt = len(label_gts)

            class_reward = total_iou
            print(f"类别 [{label}] 的加权平均 IoU 奖励为: {class_reward:.4f}")

            total_reward += total_iou
            total_weight += total_weight_for_label

        # Step 3: 返回整体奖励
        overall_reward = total_reward / total_weight if total_weight > 0 else 0.0
        print(f"\n【最终整体加权奖励】{overall_reward:.4f}")
        return overall_reward

    def _match_and_calculate_weighted_iou(self, preds: List[Dict], gts: List[Dict]) -> Tuple[float, float]:
        matched_preds = set()
        total_iou = 0.0
        total_weight = 0.0

        for gt in gts:
            best_iou = 0.0
            best_pred_idx = -1
            gt_area = (gt['bbox_2d'][2] - gt['bbox_2d'][0]) * (gt['bbox_2d'][3] - gt['bbox_2d'][1])

            for i, pred in enumerate(preds):
                if i in matched_preds:
                    continue
                
                current_iou = self.calculate_iou(pred, gt)
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_pred_idx = i

            if best_pred_idx != -1:
                matched_preds.add(best_pred_idx)
                total_iou += best_iou * gt_area
                total_weight += gt_area

        return total_iou, total_weight

    @staticmethod
    def calculate_iou(boxA: Dict, boxB: Dict) -> float:
        xA = max(boxA['bbox_2d'][0], boxB['bbox_2d'][0])
        yA = max(boxA['bbox_2d'][1], boxB['bbox_2d'][1])
        xB = min(boxA['bbox_2d'][2], boxB['bbox_2d'][2])
        yB = min(boxA['bbox_2d'][3], boxB['bbox_2d'][3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        boxAArea = (boxA['bbox_2d'][2] - boxA['bbox_2d'][0] + 1) * (boxA['bbox_2d'][3] - boxA['bbox_2d'][1] + 1)
        boxBArea = (boxB['bbox_2d'][2] - boxB['bbox_2d'][0] + 1) * (boxB['bbox_2d'][3] - boxB['bbox_2d'][1] + 1)

        iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0.0
        return iou

    def _calculate_ocr_reward(self, pred: str, gt: str) -> float:
        # 统一清洗函数：去除 \boxed{} 和换行符，并做基本清理
        def clean_text(s: str) -> str:
            # 去除 \boxed{}
            boxed_match = re.search(r'\\boxed\s*{\s*(.*?)\s*}', s)
            if boxed_match:
                s = boxed_match.group(1).strip()
            else:
                s = s.strip()

            # 去除所有换行符和多余空白
            s = s.replace('\n', '').replace('\\n', '')  # 删除显式换行
            s = ' '.join(s.split())  # 合并多个空格为一个
            return s

        # 清洗 pred 和 gt
        pred = clean_text(pred)
        gt = clean_text(gt)
        print(f"now, pred = {pred}, gt = {gt}")
        if not isinstance(pred, str) or not isinstance(gt, str):
            return 0.0

        if self.ignore_case:
            pred = pred.lower()
            gt = gt.lower()

        if pred == gt:
            return 1.0

        # 新增：子串匹配逻辑
        if pred in gt or gt in pred:
            return 1.0

        ratio = SequenceMatcher(None, pred, gt).ratio()
        return 1.0 if ratio >= self.fuzzy_match_ratio else 0.0

    def _calculate_counting_reward(self, pred: int, gt: int) -> float:
        return 1.0 if pred == gt else 0.0

    def _calculate_math_reward(self, pred: str, gt: str) -> float:
        def clean_latex(latex_str):
            return latex_str.replace(" ", "").replace("\\ ", "").replace("\n", "").strip()

        pred_clean = clean_latex(pred)
        gt_clean = clean_latex(gt)

        try:
            from sympy import sympify, Eq, SympifyError  # 👈 在这里动态导入
            pred_expr = sympify(pred_clean)
            gt_expr = sympify(gt_clean)
            if Eq(pred_expr, gt_expr):
                return 1.0
        except SympifyError:
            pass
        except Exception as e:
            print(f"Math reward error during parsing: {e}")

        ratio = SequenceMatcher(None, pred_clean, gt_clean).ratio()
        return 1.0 if ratio >= self.fuzzy_match_ratio else 0.0


# 注册新的 ORM 到 orms 字典
orms['visual_perception_accuracy'] = VisualPerceptionAccuracy
orms['visual_reasoning_accuracy'] = VisualReasoningAccuracy
orms['external_math_acc'] = MathAccuracy
orms['external_math_format'] = MathFormat
orms['external_countdown'] = CountdownORM
orms['external_r1v_acc'] = MultiModalAccuracyORM
orms['external_code_reward'] = CodeReward
orms['external_code_format'] = CodeFormat
orms['external_code_reward_by_judge0'] = CodeRewardByJudge0


# For genrm you can refer to swift/llm/plugin/rm_plugin/GenRMPlugin
class CustomizedRMPlugin:
    """
    Customized Reward Model Plugin, same to DefaultRMPlugin

    It assumes that `self.model` is a classification model with a value head(output dimmension 1).
    The first logits value from the model's output is used as the reward score.
    """

    def __init__(self, model, template):
        self.model = model
        self.template: Template = template

    def __call__(self, inputs):
        batched_inputs = [self.template.encode(deepcopy(infer_request)) for infer_request in inputs]
        reward_inputs = to_device(self.template.data_collator(batched_inputs), self.model.device)
        reward_inputs.pop('labels')

        with torch.inference_mode():
            return self.model(**reward_inputs).logits[:, 0]


class QwenLongPlugin(DefaultRMPlugin):
    # https://arxiv.org/abs/2505.17667
    # NOTE: you should customize the verified reward function, you can refer to
    # https://github.com/Tongyi-Zhiwen/QwenLong-L1/tree/main/verl/verl/utils/reward_score
    # hf_dataset: https://huggingface.co/datasets/Tongyi-Zhiwen/DocQA-RL-1.6K/viewer/default/train
    # ms_dataset: https://modelscope.cn/datasets/iic/DocQA-RL-1.6K
    def __init__(self, model, template, accuracy_orm=None):
        super().__init__(model, template)
        # initilize PTEngine to infer
        self.engine = PtEngine.from_model_template(self.model, self.template, max_batch_size=0)  # 0: no limit
        self.request_config = RequestConfig(temperature=0)  # customise your request config here
        self.system = textwrap.dedent("""
            You are an expert in verifying if two answers are the same.

            Your input consists of a problem and two answers: Answer 1 and Answer 2.
            You need to check if they are equivalent.

            Your task is to determine if the two answers are equivalent, without attempting to solve the original problem.
            Compare the answers to verify they represent identical values or meanings,
            even when expressed in different forms or notations.

            Your output must follow this format:
            1) Provide an explanation for why the answers are equivalent or not.
            2) Then provide your final answer in the form of: [[YES]] or [[NO]]

            Problem: {problem_placeholder}
            Answer 1: {answer1_placeholder}
            Answer 2: {answer2_placeholder}
        """)  # noqa
        self.accuracy_orm = accuracy_orm

    def __call__(self, inputs):
        completions = [example['messages'][-1]['content'] for example in inputs]
        ground_truths = [example['reward_model']['ground_truth'] for example in inputs]
        rm_inputs = self.prepare_rm_inputs(inputs, completions, ground_truths)

        results = self.engine.infer(rm_inputs, self.request_config, use_tqdm=False)
        llm_rewards = self.compute_rewards(results)

        if self.accuracy_orm:
            verified_rewards = self.accuracy_orm(completions, ground_truths)
        else:
            verified_rewards = [0.0] * len(llm_rewards)

        rewards = [max(r1, r2) for r1, r2 in zip(llm_rewards, verified_rewards)]
        return torch.tensor(rewards, dtype=torch.float32)

    def prepare_rm_inputs(self, inputs: List[Dict], completions, ground_truths) -> List[Dict]:
        rm_inputs = []
        for infer_request, completion, ground_truth in zip(inputs, completions, ground_truths):
            # Deep copy to prevent modification of original input
            rm_infer_request = deepcopy(infer_request)
            problem = infer_request['messages'][0]['content']
            start_index = problem.index('</text>')
            end_index = problem.index('Format your response as follows:')
            question = problem[start_index:end_index].replace('</text>', '').strip()
            prompt = self.system.format(
                problem_placeholder=question, answer1_placeholder=completion, answer2_placeholder=ground_truth)

            # Construct new messages tailored for the reward model
            rm_messages = [{'role': 'user', 'content': prompt}]

            # Update the messages in the reward infer request
            rm_infer_request['messages'] = rm_messages
            rm_inputs.append(rm_infer_request)
        return rm_inputs

    @staticmethod
    def extract_reward(model_output: str) -> float:
        match = re.search(r'\[([A-Z]+)\]', model_output)
        if match:
            answer = match.group(1)
            if answer == 'YES':
                return 1.0
            elif answer == 'NO':
                return 0.0
            else:
                logger.warning("Unexpected answer, expected 'YES' or 'NO'.")
                return 0.0
        else:
            logger.warning("Unable to extract reward score from the model's output, setting reward to 0")
            return 0.0  # Or raise ValueError("Format incorrect")

    def compute_rewards(self, results: List[ChatCompletionResponse]) -> List[float]:
        """
        Compute average reward scores from the reward model's outputs.

        Args:
            results (List[ChatCompletionResponse]): A list of results from the reward model.

        Returns:
            List[float]: A list of average reward scores.
        """
        rewards = []
        for idx, output in enumerate(results):
            try:
                cur_rewards = []
                for choice in output.choices:
                    response = choice.message.content
                    reward = self.extract_reward(response)
                    cur_rewards.append(reward)
                cur_rewards = [r for r in cur_rewards if r is not None]
                if cur_rewards:
                    average_reward = sum(cur_rewards) / len(cur_rewards)
                else:
                    average_reward = 0.0
                    logger.warning('No valid rewards extracted. Assigning reward score of 0.0.')

                rewards.append(average_reward)
            except Exception as e:
                logger.error(f'Error computing reward: {e}')
                rewards.append(0.0)  # Assign default reward score on failure
        return rewards


rm_plugins['my_rmplugin'] = CustomizedRMPlugin
rm_plugins['qwenlong'] = QwenLongPlugin
