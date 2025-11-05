from openai import OpenAI
import asyncio
import re
from copy import deepcopy
from typing import List, Tuple

import json
import torch

from swift.llm import Template, to_device
from swift.plugin import ORM, orms, rm_plugins
from swift.utils import get_logger

import asyncio
from openai import AsyncOpenAI

logger = get_logger(__name__)

ATOMIC_ACTION_SET = [
    "HandShake", "Clap", "Wave", "Bow", "Dance", "Like", "Give a thumbs-up", 
    "Stretch", "HandOver", "Catch", "Screw", "Unscrew", "Peel", "Stack", 
    "Flip", "Heart sign", "Make a heart with hands", "Tear", "Pinch", "Search", 
    "Scoop", "Cut", "Chop", "Sweep", "Stir", "Throw", "Slide", "Drag", "Shake", 
    "Hammer", "Spread", "Pour", "Navigate", "Point", "Pick", "Place", 
    "Pick and Place", "Fold", "UnFold", "Press", "Open", "Close", "Pull", 
    "Push", "Insert", "Grasp and Release", "Grasp", "Release", "Lift", "Turn", 
    "Twist", "Wash", "Rinse", "Beat", "Hang", "Follow", "Rotate in place"]

def extract_score(text: str) -> float | None:
    """
    从字符串中提取最后一个 <score>...</score> 标签中的浮点数。
    若无匹配或转换失败，返回 None。
    """
    match = re.search(r'<score>\s*([0-9]+\.?[0-9]*)\s*</score>', text, flags=re.I)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None

class RMReward(ORM):
    
    def __init__(self):
        super().__init__()
        try:
            sync_client = OpenAI(
                            api_key='EMPTY',
                            base_url='http://10.208.68.115:31415/v1',
                        )
            self.verify_model_name = sync_client.models.list().data[0].id
            logger.info(f"Connected to model: {self.verify_model_name}")


            self.client = AsyncOpenAI(
                api_key='EMPTY',
                base_url='http://10.208.68.115:31415/v1',
            )
            # self.verify_model_name = self.client.models.list().data[0].id
            # logger.info(f"Connected to model: {self.verify_model_name}")
        except Exception as e:
            raise RuntimeError('Failed to connect to the model service. Please deploy the model '
                             "using 'swift deploy' or 'vllm serve'.") from e

    def _parse_score(self, text: str) -> float:
        """
        从模型返回文本中提取 0~1 的浮点数，支持多种格式：
        - 直接数字：0.85
        - 带单位：0.85分, 得分: 0.85
        - 中文描述后跟数字
        """
        # 移除常见无关字符
        text = text.strip().replace(' ', '').replace('：', ':').replace('：', ':')
        
        # 尝试匹配 0.x 或 1.0 格式的浮点数（0~1 范围）
        match = re.search(r'(?:得分|分数|score|grade|评分)?[:：]?\s*(\d*\.?\d+)', text, re.IGNORECASE)
        if match:
            try:
                score = float(match.group(1))
                if 0.0 <= score <= 1.0:
                    return score
            except ValueError:
                pass

        # 再次尝试全局搜索所有数字
        numbers = re.findall(r'\d*\.?\d+', text)
        for num_str in numbers:
            try:
                score = float(num_str)
                if 0.0 <= score <= 1.0:
                    return score
            except ValueError:
                continue

        # 如果都失败，返回 None 表示无法解析
        return None

    async def _get_reward(self, completion, sol, msg, task_type):
        if task_type != "planning_action_set":
            return None

        question_items = [item['content'] for item in msg if item['role'] == 'user']
        if not question_items:
            logger.warning("No user message found in conversation. Skipping.")
            return 0.0

        question = question_items[0]

        rm_prompt = f"""
        你是一位**机器人任务回复和规划领域的专家评审员**，你的任务是对模型生成的回复、任务规划和动作答案内容进行**严谨、客观、公正的评分**，评分范围从 0.00 到 1.00，保留两位小数，请简要思考后进行打分

        ---

        ### 评分参考准则
        请根据包括但不局限于以下四项核心维度，基于标准答案对模型生成的答案进行评价：

        ---

        #### 1. <response> 响应质量
        - **礼貌性**：是否自然回应用户请求，语气友好？
        - **准确性**：是否准确反映任务意图？
        - **简洁性**：是否简洁明了，不冗余？
        - **语言自然度**：是否符合自然习惯？

        ---

        #### 2. <plans> 规划质量
        - **完整性**：是否覆盖从起点到完成任务的**所有必要步骤**？
        - **逻辑性**：步骤顺序是否合理？是否存在“先操作再导航”等错误？
        - **可执行性**：每一步是否基于图像内容，不假设不可见物体？
        - **规范性**：
        - 是否严格使用 `[Navigate]` 和 `[Manipulate]` 这两种大写标签
        - **冗余性**：是否存在无意义或重复的步骤？

        ---

        #### 3. <actions> 动作序列质量
        - **动词准确性：** <actions> 中的动词是否 **完全** 来自预定义的原子动作集。任何不在此列表中的动词都会导致此项扣分
        - **结构规范性**：
        - 是否保持与 <plans> 的语义一致性？
        - **与<plans>的匹配度：** <actions> 中的序列是否与 <plans> 中的步骤语义等价，并保持一致的对象和位置名称。
        - **参数完整性**：是否包含必要 object 和 location？
        - **顺序一致性**：是否与 <plans> 的步骤一一对应？

        #### 4. 格式
        - 整体结构：是否严格遵循 <response>...</response><plans>...</plans><actions>...</actions> 的 XML 结构，且所有标签完整，没有多余或缺失的部分
        - 嵌套与顺序：标签的嵌套和排列顺序是否正确无误

        ---

        ### 错误等级判定（一旦触发，需要给较低的评分）
        - 使用了未定义的动作动词。
        - 使用了标准答案中不存在的物体或位置。
        - 遗漏关键步骤（如未抓取物品、未交付给用户）。
        - 动作顺序严重错误（如先放置再抓取）。
        - 输出格式不合法（如缺失 XML 标签、JSON 格式错误）。

        ---

        ### 评分细则（供参考）
        - **1.00**：与标准答案几乎完全一致，语言自然，逻辑严谨。
        - **0.90~0.99**：语义等价，略有表达差异，无逻辑错误。
        - **0.75~0.89**：结构合理，细节略有缺失或冗余，但不影响执行。
        - **0.50~0.74**：存在轻微逻辑或格式问题，仍可部分执行。
        - **0.25~0.49**：存在明显错误，如顺序颠倒、动作缺失。
        - **0.00~0.24**：严重错误，如使用非法动作、物体不存在、格式崩溃。

        ---

        ### 原问题
        {question}

        ---

        ### 标准答案
        {sol}

        ---

        ###原子动作集
        {ATOMIC_ACTION_SET}
        ---

        ### 模型生成答案
        {completion}

        ---

        ### 请将最终的评分结果放入<score></score>标签中，注意评分时请将极简思考过程放入<think></think>中，**不可以进行深入和冗长的思考**

        示例回复: <think>极简思考过程</think><score>0.75</score>
        """

        try:
            chat_response = await self.client.chat.completions.create(
                model=self.verify_model_name,
                messages=[
                    {
                        'role': 'system',
                        'content': 'You are a precise and objective scoring assistant. Output final score between 0 and 1.'
                    },
                    {
                        'role': 'user',
                        'content': rm_prompt
                    },
                ],
                max_tokens=4096,
                temperature=0.6,  # 降低随机性
                top_p=1.0
            )
            response_text = chat_response.choices[0].message.content.strip()
            logger.debug(f"Raw RM response: {response_text}")

            # 解析得分
            # parsed_score = self._parse_score(response_text)
            parsed_score = extract_score(response_text)
            if parsed_score is not None:
                return parsed_score
            else:
                logger.warning(f"Failed to parse score from response: '{response_text}'. Using default 0.0.")
                return 0.0

        except Exception as e:
            logger.error(f"Error during reward modeling: {str(e)}", exc_info=True)
            return 0.0  # 安全兜底

    def __call__(self, completions, solution, task, **kwargs) -> List[float]:
        messages = kwargs.get('messages', [])
        
        # # Collect tasks for parallel execution
        # tasks = []
        # for completion, sol, msg, task_type in zip(completions, solution, messages, task):
        #     tasks.append(self._get_reward(completion, sol, msg, task_type))

        # # Run asynchronous tasks in parallel
        # rewards = asyncio.run(asyncio.gather(*tasks))
        async def _run_all_rewards():
            tasks = []
            for completion, sol, msg, task_type in zip(completions, solution, messages, task):
                tasks.append(self._get_reward(completion, sol, msg, task_type))
            
            rewards = await asyncio.gather(*tasks)
            return rewards

        rewards = asyncio.run(_run_all_rewards())

        if None not in rewards:
            print("\n ============== \n")
            print(f"task: {task} \n")
            print(f"solution: {solution} \n")
            print(f"completions: {completions} \n")
            print(f"reward: {rewards}")
            print("\n ============== \n")

        return rewards

orms['external_planning_rm_ORM'] = RMReward