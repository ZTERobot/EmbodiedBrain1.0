# plugin_joint_if_spatial_training.py

import asyncio
import re
from copy import deepcopy
from typing import List, Tuple, Dict

import json
import torch

from swift.llm import Template, to_device
from swift.plugin import ORM, orms, rm_plugins
from swift.utils import get_logger
from tuluif import *


logger = get_logger()

class MathAnswerFormat(ORM):
    def __call__(self, completions, solution, task, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r'^.*?<answer>.*?</answer>(?![\s\S])'
        rewards = []
        for content, sol, task_name in zip(completions, solution, task):
            if task_name == "spatial_understanding":
                match = re.match(pattern, content, re.DOTALL | re.MULTILINE)
                reward = 1.0 if match else 0.0
                rewards.append(reward)
            else:
                rewards.append(None)
        
        return rewards

#    def __call__(self, completions, **kwargs) -> List[float]:
#        """Reward function that checks if the completion has a specific format."""
#        pattern = r'^.*?<answer>.*?</answer>(?![\s\S])'
#        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
#        return [1.0 if match else 0.0 for match in matches]
    

class InstructionFollowingORM(ORM):
    def __call__(self, completions, solution, task, **kwargs) -> List[float]:
        """
        Reward function that checks if the completion is correct.
        Args:
            completions (list[str]): Generated outputs
            solution (list[str]): Ground Truths.

        Returns:
            list[float]: Reward scores
        """

        rewards = []

        count = 0 
        for content, sol, task_name in zip(completions, solution, task):
            if task_name == "instruction_following":
                reward = 0.0

                # 将外部的 JSON 字符串解析为 Python 字典
                # ground_truth_dict = json.loads(solution.replace("'", '"'))
                # ground_truth_dict = json.loads(sol)
                ground_truth_dict = eval(sol)
                # 提取内部的 JSON 字符串
                inner_ground_truth = ground_truth_dict['ground_truth']

                reward = compute_score(content, inner_ground_truth)
            
                if count < 10:
                    count += 1
                    print("模型输出的answer为", content)
                    print("groundtruth为", inner_ground_truth)

            # # Try symbolic verification first
            # try:
            #     answer = parse(content)
            #     if float(verify(answer, parse(sol))) > 0:
            #         reward = 1.0
            # except Exception:
            #     pass  # Continue to next verification method if this fails

            # # If symbolic verification failed, try string matching
            # if reward == 0.0:
            #     try:
            #         # Extract answer from solution if it has think/answer tags
            #         sol_match = re.search(r'<answer>(.*?)</answer>', sol)
            #         ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

            #         # Extract answer from content if it has think/answer tags
            #         content_match = re.search(r'<answer>(.*?)</answer>', content)
            #         student_answer = content_match.group(1).strip() if content_match else content.strip()

            #         # Compare the extracted answers
            #         if student_answer == ground_truth:
            #             reward = 1.0
            #     except Exception:
            #         pass  # Keep reward as 0.0 if both methods fail
                rewards.append(reward)
            else:
                rewards.append(None)

        if None not in rewards:
            print("\n ============== \n")
            print(f"task: {task} \n")
            print(f"solution: {solution} \n")
            print(f"completions: {completions} \n")
            print(f"reward: {rewards}")
            print("\n ============== \n")

        return rewards
    

class SpatialReasoningORM(ORM):
    ENGLISH_PLACE_PREPOSITIONS = {
        "in", "under", "above", "below", 
        "beside", "next to", "between", "among", "behind", "left","right",
        "in front of", "inside", "outside", "near",
        "beside", "beyond", "across", "through", "onto", "into","beneath"
    }
    SYNONYMS_MAP = {
        "under": ["under", "below", "beneath"],
        "over": ["over", "above", "on top of"],
        # 可继续扩展
    }
    ANTONYMS_MAP = {
        "under": ["over", "above", "on top of"],
        "below": ["above", "over", "on top of"],
        "beneath": ["above", "over", "on top of"],
        "over": ["under", "below", "beneath"],
        "above": ["under", "below", "beneath"],
        "on top of": ["under", "below", "beneath"],
        "left": ["right"],
        "right": ["left"],
        # … 可根据需求扩展更多反义词对
    }

    def __call__(self, completions, solution, task, **kwargs) -> List[float]:
        # print("\n ============== \n")
        # print(f"all kwargs: {kwargs} \n")
        # print(f"all kwargs keys: {kwargs.keys()} \n")
        # print(f"len completions: {len(completions)} \n")
        # print(f"len solution: {len(solution)} \n")
        # print(f"len task: {len(task)} \n")
        # print(f"completions: {completions} \n")
        # print(f"solution: {solution} \n")
        # print(f"task: {task} \n")
        # print("\n ============== \n")
        rewards = []
        reverse_synonyms_map = self._build_reverse_synonyms_map(self.SYNONYMS_MAP)
        for content, sol, task_name in zip(completions, solution, task):
            if task_name == "spatial_understanding":
                reward = 0.0
                # 从标签中提取内容
                content_match = re.search(r'<answer>(.*?)<\/answer>', content)
                student_answer = content_match.group(1).strip() if content_match else ""
                # student_answer = content_match.group(1).strip() if content_match else content
                sol_match = re.search(r'<answer>(.*?)<\/answer>', sol)
                ground_truth = sol_match.group(1).strip() if sol_match else ""
                if not student_answer or not ground_truth:
                    # if not student_answer:
                    #    reward = -1
                    rewards.append(reward)
                    log_entry = {
                        "student_answer": student_answer,
                        "ground_truth": ground_truth,
                        "student_structure": "N/A",
                        "gt_structure": "N/A",
                        "reward": reward,
                        "reason": "Missing <answer> tag or empty content"
                    }
                    continue
                # 判断是单词/词组还是句子
                if self._is_word_or_phrase(student_answer) and self._is_word_or_phrase(ground_truth):
                    # 单词/词组直接比较（不区分大小写）
                    reward = 1.0 if student_answer.lower() == ground_truth.lower() else 0.0
                    log_entry = {
                        "student_answer": student_answer,
                        "ground_truth": ground_truth,
                        "student_structure": "",
                        "gt_structure": "",
                        "reward": reward,
                        "reason": "Comparison performed"
                    }
                else:
                    # 提取【对象 地点介词 对象】结构
                    # student_structure = self._extract_structure(student_answer)
                    # gt_structure = self._extract_structure(ground_truth)
                    # reward = 1.0 if student_structure == gt_structure else 0.0
                    student_answer_sent_count = self._count_clauses(student_answer)
                    ground_truth_sent_count = self._count_clauses(ground_truth)

                    gt_structure, is_gt_structure_preposition = self._extract_structure(ground_truth)
                    if is_gt_structure_preposition == 1:
                        passed_or_not, matched, _ = self._check_synonyms(
                                                            std_answer_key_word=gt_structure,
                                                            user_answer=student_answer,
                                                            reverse_map=reverse_synonyms_map,
                                                            require_all=True
                                                        )
                        if passed_or_not == True:
                            subject, object_ = self._extract_subject_object(gt_structure, ground_truth)
                            is_consisten = self._is_consistent_answer(matched[gt_structure], student_answer, subject, object_)
                            reward = 1.0 if is_consisten == True else 0.0
                        else:
                            reward = 0.0
                        
                        if reward < 1.0:
                            found_antonyms, matched_antonyms, _ = self._check_antonyms(
                                                                        std_answer_word=gt_structure,
                                                                        user_answer=student_answer,
                                                                        antonyms_map=self.ANTONYMS_MAP,
                                                                        require_all=False  # 只需检测到任意一个反义词
                                                                    )
                            if found_antonyms == True:
                                subject, object_ = self._extract_subject_object(gt_structure, ground_truth)
                                is_consisten = self._is_consistent_answer(matched_antonyms[gt_structure], student_answer, object_, subject)
                                if is_consisten == True:
                                    reward = 1.0
                        
                        student_structure, _ = self._extract_structure(student_answer)
                        # reward = 1.0 if student_structure == gt_structure else 0.0
                        # if reward == 1.0:
                        #     subject, object_ = self._extract_subject_object(gt_structure, ground_truth)
                        #     is_consisten = self._is_consistent_answer(gt_structure, student_answer, subject, object_)
                        #     if is_consisten == False:
                        #         reward = 0.0
                    else:
                        found = re.search(gt_structure, student_answer.lower())
                        reward = 1.0 if found else 0.0    
                        student_structure, _ = self._extract_structure(student_answer)  

                    if reward == 1.0:
                        if self._is_word_or_phrase(ground_truth):
                            penality_reward = len(student_answer.split()) * 0.2 / len(ground_truth.split())
                            reward = reward - penality_reward
                            reward = reward if reward > 0.0 else 0 
                        else:
                            penality_reward = (student_answer_sent_count - ground_truth_sent_count) * 0.2
                            reward = reward - penality_reward
                            reward = reward if reward > 0.0 else 0

                            if student_answer_sent_count == 1 and ground_truth_sent_count == 1:
                                lens_ratio = len(student_answer.split()) / len(ground_truth.split())
                                if lens_ratio > 1.3:
                                    penality_reward = len(student_answer.split()) * 0.2 / len(ground_truth.split())
                                    reward = reward - penality_reward
                                    reward = reward if reward > 0.0 else 0    
                    # else: # reward == 0.0
                        # please do not bullshit
                    #    if self._is_word_or_phrase(ground_truth):
                    #        penality_reward = len(student_answer.split()) * 0.2 / len(ground_truth.split())
                    #        reward = reward - penality_reward
                    #        reward = -1.0 if reward < -1.0 else reward
                    #    else:
                    #        penality_reward = (student_answer_sent_count - ground_truth_sent_count) * 0.3
                    #        reward = reward - penality_reward
                    #        reward = -1.0 if reward < -1.0 else reward
                    #        if student_answer_sent_count == 1 and ground_truth_sent_count == 1:
                    #            lens_ratio = len(student_answer.split()) / len(ground_truth.split())
                    #            if lens_ratio > 1.3:
                    #                penality_reward = len(student_answer.split()) * 0.3 / len(ground_truth.split())
                    #                reward = reward - penality_reward
                    #                reward = -1.0 if reward < -1.0 else reward 
                    #            elif lens_ratio < 0.7:
                    #                penality_reward = 1.0 - (lens_ratio * 0.85)
                    #                reward = reward - penality_reward 
                    #                reward = -1.0 if reward < -1.0 else reward

                    log_entry = {
                        "student_answer": student_answer,
                        "ground_truth": ground_truth,
                        "student_structure": student_structure,
                        "gt_structure": gt_structure,
                        "reward": reward,
                        "reason": "Comparison performed"
                    }
                rewards.append(reward)
            else:
                rewards.append(None)

        if None not in rewards:
            print("\n ============== \n")
            print(f"task: {task} \n")
            print(f"solution: {solution} \n")
            print(f"completions: {completions} \n")
            print(f"reward: {rewards}")
            print("\n ============== \n")

        return rewards

    def _is_word_or_phrase(self, text: str) -> bool:
        """判断文本是否为单词或词组（英文场景）"""
        if not text:
            return False
            
        # 简单规则：单词数少于4个且不含标点符号视为单词/词组
        words = text.split()
        has_punctuation = any(char in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~" for char in text)
        return len(words) < 4 and not has_punctuation
    
    def _extract_structure(self, sentence: str) -> str:
        """
        提取英文句子中的【对象 地点介词 对象】结构
        寻找地点介词，并向前和向后溯源找到最近的the/The，提取the后面的单词对象
        """
        if not sentence:
            return ""
            
        words = sentence.split()
        preposition_index = -1
        # 寻找第一个地点介词
        for i, word in enumerate(words):
            if word.lower() in self.ENGLISH_PLACE_PREPOSITIONS:
                preposition_index = i
                break
                
        if preposition_index == -1:
            return sentence.lower().strip(), 0 # here, 0 means no prepostions
            # 没找到地点介词，直接返回原内容（小写处理）
            # return sentence.lower().strip()
            
        preposition = words[preposition_index].lower()

        # 组合结构
        return f"{preposition}".strip(), 1 # here, 1 means has propositions
        # return f"{preposition}".strip()

    def _count_clauses(self, sentence: str) -> int:
        """
        返回一句话中，大致的子句数量（包括主句 + 从句）
        """
        # 载入英文模型
        import spacy
        from spacy.tokens import Doc

        nlp = spacy.load("en_core_web_sm")

        # 定义我们认为表示“子句”的依存关系标签集合
        CLAUSE_DEPS = {
            "ccomp",    # clausal complement
            "xcomp",    # open clausal complement
            "advcl",    # adverbial clause
            "relcl",    # relative clause
            "acl",      # clausal modifier of noun (非关系从句)
            "parataxis" # 并列主句
        }
    
        doc: Doc = nlp(sentence)
    
        # 至少有一个主句
        clause_count = 1
    
        for token in doc:
            # 子句依存标签
            if token.dep_ in CLAUSE_DEPS:
                clause_count += 1
        
            # 并列动词（动词之间的 conj）也可能是并列句
            # e.g. "She sings and dances." -> two子句
            if token.dep_ == "conj" and token.pos_ == "VERB":
                clause_count += 1
    
        return clause_count

    def _extract_subject_object(self, gt_structure: str, text: str) -> Tuple[str, str]:
        """
        从任意包含 'left' 的句子里，自动提取主词和宾词：
        - 在 'left' 之前：去掉停用词后最后一个单词为 subject
        - 在 'left' 之后：去掉停用词后第一个单词为 object
        """
        # 一组简单停用词，去除常见介词、连词、冠词等
        STOPWORDS = {
            'the','a','an','of','on','to','in','at','is','are',
            'and','side','by','for','with','that','this'
        }

        tokens = re.findall(r"\w+", text.lower())
        if gt_structure not in tokens:
            raise ValueError("输入文本中未发现关键字")
        idx = tokens.index(gt_structure)

        # 提取 subject
        before = [tok for tok in tokens[:idx] if tok not in STOPWORDS]
        if not before:
            raise ValueError("无法在 keyword 之前提取到 subject")
        subject = before[-1]

        # 提取 object
        after = [tok for tok in tokens[idx+1:] if tok not in STOPWORDS]
        if not after:
            raise ValueError("无法在 keyword 之后提取到 object")
        object_ = after[0]

        return subject, object_
	

    def _is_consistent_answer(self, gt_structure: str, ans: str, subject: str, object_: str) -> bool:
        """
        检查 ans 中是否同时包含 subject、left、object，
        且顺序为 subject → left → object。
        """
        tokens = re.findall(r"\w+", ans.lower())
        # 必须同时出现
        if not all(x in tokens for x in (subject, gt_structure, object_)):
            return False

        subj_i = tokens.index(subject)
        left_i = tokens.index(gt_structure)
        obj_i = tokens.index(object_)
        return subj_i < left_i < obj_i

    def _tokenize(self, text: str) -> List[str]:
        """
        split sentence into words,  remove all the Periods and commas.
        """
        return re.findall(r"\b\w+\b", text.lower())

    
    def _build_reverse_synonyms_map(self, synonyms_map):
        """
        Build a reverse dictionary from SYNONYMS_MAP, so that every synonyms can be mapped to its synonyms list.
        """
        reverse_map = {}
        for base_word, syn_list in synonyms_map.items():
            for syn in syn_list:
                reverse_map[syn.lower()] = [w.lower() for w in syn_list]
    
        return reverse_map        

    def _check_synonyms(self, std_answer_key_word: str,
                        user_answer: str,
                        reverse_map: Dict[str, List[str]],
                        require_all: bool = True) -> Tuple[bool, Dict[str, str], List[str]]:
        """
        检查 user_answer 中是否包含 std_answer_key_word 的词或其同义词。

        参数:
        std_answer_key_word: 标准答案文本
        user_answer: 用户回答文本
        reverse_map: 反向同义词映射表
        require_all: 是否要求所有标准词都匹配（True），
                    或只要匹配到任意一个即通过（False）

        返回:
        matched_all: 是否通过检测
        matched:     {标准词: 用户回答中匹配到的同义词}
        missing:     未匹配到的标准词列表
        """
        user_tokens = set(self._tokenize(user_answer))

        matched = {}
        missing = []
        # 获取该词在 reverse_map 中的同义词集合，若不存在，则仅自身
        candidates = reverse_map.get(std_answer_key_word, [std_answer_key_word])
        for syn in candidates:
            if syn in user_tokens:
                matched[std_answer_key_word] = syn
                break
        else:
            missing.append(std_answer_key_word)

        if require_all:
            return len(missing) == 0, matched, missing
        return (len(matched) > 0), matched, missing


    def _check_antonyms(self, std_answer_word: str, 
                        user_answer: str, 
                        antonyms_map: Dict[str, List[str]], 
                        require_all: bool = False) -> Tuple[bool, Dict[str, str], List[str]]:
        """
        检查 user_answer 中是否包含 std_answer_word 的词或短语的反义词。

        参数:
        std_answer_word: 标准答案文本
        user_answer: 用户回答文本
        antonyms_map: 反义词映射字典
        require_all: 是否要求针对所有标准词都检测到反义词（True）
                    或只要检测到任意一个即可（False，默认）

        返回:
        found_any: 是否检测到反义词（当 require_all=False 时表示检测到至少一个；
                    require_all=True 时表示所有词都有对应检测）
        matched:   dict，{标准词: 匹配到的反义词}
        missing:   未匹配到反义词的标准词列表
        """
        # 分词及预处理
        user_tokens = set(self._tokenize(user_answer))
        user_text_lower = user_answer.lower()

        matched = {}
        missing = []
        
        candidates = antonyms_map.get(std_answer_word.lower(), [])
        found = False
        for ant in candidates:
            # 短语匹配
            if " " in ant:
                if ant.lower() in user_text_lower:
                    matched[std_answer_word] = ant
                    found = True
                    break
            else:
                if ant.lower() in user_tokens:
                    matched[std_answer_word] = ant
                    found = True
                    break
        if not found:
            missing.append(std_answer_word)

        if require_all:
            return len(missing) == 0, matched, missing
        return (len(matched) > 0), matched, missing


class PlanningActionSetORM(ORM):
    # 原子动作集合，基于用户提供的prompt中的动作列表
    ATOMIC_ACTION_SET = {
        "HandShake", "Clap", "Wave", "Bow", "Dance", "Like", "Give a thumbs-up", 
        "Stretch", "HandOver", "Catch", "Screw", "Unscrew", "Peel", "Stack", 
        "Flip", "Heart sign", "Make a heart with hands", "Tear", "Pinch", "Search", 
        "Scoop", "Cut", "Chop", "Sweep", "Stir", "Throw", "Slide", "Drag", "Shake", 
        "Hammer", "Spread", "Pour", "Navigate", "Point", "Pick", "Place", 
        "Pick and Place", "Fold", "UnFold", "Press", "Open", "Close", "Pull", 
        "Push", "Insert", "Grasp and Release", "Grasp", "Release", "Lift", "Turn", 
        "Twist", "Wash", "Rinse", "Beat", "Hang", "Follow", "Rotate in place"
    }

    def __call__(self, completions, solution, task, **kwargs) -> List[float]:
        """
        奖励函数用于验证planning_action_set任务的输出格式
        Args:
            completions (list[str]): 生成的输出
            solution (list[str]): 标准答案
            task (list[str]): 任务名称列表
        Returns:
            list[float]: 奖励分数列表
        """
        rewards = []
        
        for content, sol, task_name in zip(completions, solution, task):
            if task_name == "planning_action_set":
                reward = 0.0
                
                # response部分：占20% (0.2分) - 只检查XML标签正确闭合
                response_score = self._check_response_tag(content) * 0.2
                
                # plans部分：占30% (0.3分) - 检查XML标签 + step_type验证 + 格式验证
                plans_score, plans_count = self._check_plans_section(content)
                plans_score *= 0.3
                
                # actions部分：占50% (0.5分) - 检查XML标签 + 动词验证
                actions_score, actions_count = self._check_actions_section(content)
                actions_score *= 0.5
                
                # 检查plans和actions数量是否一致，如果不一致则都打五折
                if plans_count != actions_count and plans_count > 0 and actions_count > 0:
                    plans_score *= 0.5
                    actions_score *= 0.5
                
                # 综合计算奖励
                reward = response_score + plans_score + actions_score
                
                rewards.append(reward)
            else:
                rewards.append(None)
        
        if None not in rewards:
            print("\n ============== Planning Action Set ORM ============== \n")
            print(f"task: {task} \n")
            print(f"solution: {solution} \n")
            print(f"completions: {completions} \n")
            print(f"reward: {rewards}")
            print("\n ============================================== \n")
            
        return rewards
    
    def _check_response_tag(self, content: str) -> float:
        """
        检查response标签是否存在且正确闭合
        返回1.0或0.0
        """
        start_match = re.search(r'<response>', content, re.IGNORECASE)
        end_match = re.search(r'</response>', content, re.IGNORECASE)
        
        if start_match and end_match and start_match.start() < end_match.start():
            return 1.0
        return 0.0
    
    def _check_plans_section(self, content: str) -> tuple:
        """
        检查plans部分：XML标签正确性 + step_type验证 + 格式验证
        如果XML标签错误直接返回(0.0, 0)，否则按step_type正确比例给分
        要求：
        1. step_type必须是Navigate或Manipulate（首字母大写）
        2. 必须有序号标记（1. 2. 3. 等）
        返回：(score, count)
        """
        # 首先检查XML标签
        start_match = re.search(r'<plans>', content, re.IGNORECASE)
        end_match = re.search(r'</plans>', content, re.IGNORECASE)
        
        if not (start_match and end_match and start_match.start() < end_match.start()):
            return 0.0, 0
            
        # 提取plans内容
        plans_content = content[start_match.end():end_match.start()]
        
        # 查找所有带序号的行，格式如：1.[step_type] description
        numbered_lines = re.findall(r'\d+\.\s*\[(\w+)\]', plans_content)
        
        if not numbered_lines:
            return 0.0, 0
            
        # 验证每个step_type是否为Navigate或Manipulate（严格大小写）
        valid_step_types = {"Navigate", "Manipulate"}
        correct_steps = sum(1 for step_type in numbered_lines 
                           if step_type in valid_step_types)
        
        score = correct_steps / len(numbered_lines)
        return score, len(numbered_lines)
    
    def _check_actions_section(self, content: str) -> tuple:
        """
        检查actions部分：XML标签正确性 + 动词验证
        如果XML标签错误直接返回(0.0, 0)，否则按动词正确比例给分
        返回：(score, count)
        """
        # 首先检查XML标签
        start_match = re.search(r'<actions>', content, re.IGNORECASE)
        end_match = re.search(r'</actions>', content, re.IGNORECASE)
        
        if not (start_match and end_match and start_match.start() < end_match.start()):
            return 0.0, 0
        
        # 提取actions内容
        actions_content = content[start_match.end():end_match.start()].strip()
        
        try:
            # 尝试解析JSON格式的actions
            actions_content = re.sub(r'\s+', ' ', actions_content)
            actions_list = eval(actions_content)
            
            if not isinstance(actions_list, list):
                return 0.0, 0
                
            total_actions = len(actions_list)
            if total_actions == 0:
                return 0.0, 0
                
            correct_score = 0.0
            
            for action in actions_list:
                if isinstance(action, list) and len(action) > 0:
                    verb = action[0]
                    if isinstance(verb, str) and verb in self.ATOMIC_ACTION_SET:
                        # 检查动作list长度，如果小于2（没有动作对象），只得一半分数
                        if len(action) < 2:
                            correct_score += 0.5
                        else:
                            correct_score += 1.0
                        
            score = correct_score / total_actions
            return score, total_actions
            
        except Exception as e:
            # 如果解析失败，返回0分
            return 0.0, 0


orms['external_planning_rule_ORM'] = PlanningActionSetORM


