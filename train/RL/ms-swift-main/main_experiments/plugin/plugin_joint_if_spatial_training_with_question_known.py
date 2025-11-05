# plugin_joint_if_spatial_training_with_question_known.py

import asyncio
import re
from copy import deepcopy
from typing import List, Tuple, Dict, Optional

import json
import torch

from swift.llm import Template, to_device
from swift.plugin import ORM, orms, rm_plugins
from swift.utils import get_logger
from tuluif import *


logger = get_logger()

class MathAnswerFormat(ORM):
    def __call__(self, completions, solution, task, question, **kwargs) -> List[float]:
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
    def __call__(self, completions, solution, task, question, **kwargs) -> List[float]:
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
            
                # if count < 10:
                #    count += 1
                #    print("模型输出的answer为", content)
                #    print("groundtruth为", inner_ground_truth)

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
            print(f"quesiton: {question} \n")
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
        "under": ["under", "below", "beneath", "down"],
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
    KEY_WORDS_FOR_WHICH_QUESTIONS = {
        "shortest", "farthest", "most", "nearest", "closest", "greatest", 
    }

    def __call__(self, completions, solution, task, question, **kwargs) -> List[float]:

        rewards = []

        for content, sol, task_name, quest in zip(completions, solution, task, question):
            if task_name == "spatial_understanding":
                reward = 0.0
                # 从标签中提取内容
                content_match = re.search(r'<answer>(.*?)<\/answer>', content)
                student_answer_in_tags = content_match.group(1).strip() if content_match else ""
                student_answer = content_match.group(1).strip() if content_match else content
                sol_match = re.search(r'<answer>(.*?)<\/answer>', sol)
                ground_truth = sol_match.group(1).strip() if sol_match else ""
                if not student_answer or not ground_truth:
                    # if not student_answer:
                    #    reward = -1
                    rewards.append(reward)
                    continue
                
                found_which, options_for_answers = self._extract_which_opitons(quest)
                if found_which: # which
                    reward = self._check_answer_for_single_selection_question(ground_truth, student_answer, quest, options_for_answers)
                else: # what, how
                    reverse_synonyms_map = self._build_reverse_synonyms_map(self.SYNONYMS_MAP)
                    reward = self._check_answer_for_description_question(ground_truth, student_answer, quest, reverse_synonyms_map)

                if student_answer_in_tags != student_answer:
                    reward -= 0.4
                    if reward < 0.0:
                        reward = 0.0

                rewards.append(reward)
            else:
                rewards.append(None)

        if None not in rewards:
            print("\n ============== \n")
            print(f"task: {task} \n")
            print(f"question: {question} \n")
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


    def _check_answer_for_description_question(self, ground_truth: str, student_answer: str, question: str, reverse_map):
        reward = 0.0
        reverse_synonyms_map = self._build_reverse_synonyms_map(self.SYNONYMS_MAP)
        if self._is_word_or_phrase(ground_truth):
            reward = 0.0
            print("Warning: Invalid Spational Reasoning Corpus, what and how")
        else:
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
            
            else:
                found = re.search(gt_structure, student_answer.lower())
                reward = 1.0 if found else 0.0 
                student_structure, _ = self._extract_structure(student_answer)

            if reward == 1.0:
                penality_reward = (student_answer_sent_count - ground_truth_sent_count) * 0.2
                reward = reward - penality_reward
                reward = reward if reward > 0.0 else 0

                if student_answer_sent_count == 1 and ground_truth_sent_count == 1:
                    lens_ratio = len(student_answer.split()) / len(ground_truth.split())
                    if lens_ratio > 1.6:
                        # penality_reward = len(student_answer.split()) * 0.2 / len(ground_truth.split())
                        penality_reward = (lens_ratio - 1.0) * 0.2
                        reward = reward - penality_reward
                        reward = reward if reward > 0.0 else 0
    

        return reward


    def _check_answer_for_single_selection_question(self, 
                                                    ground_truth: str, 
                                                    student_answer: str, 
                                                    question: str, 
                                                    options_for_answers: Optional[List[str]]):
        reward = 0.0

        if self._is_word_or_phrase(ground_truth):
            if self._is_word_or_phrase(student_answer):
                # 单词/词组直接比较（不区分大小写）
                reward = 1.0 if student_answer.lower() == ground_truth.lower() else 0.0    
            else: # student answer are sentences.
                if ground_truth in student_answer.lower():
                    key_word = self._extract_key_word_in_single_selection_question(question)
                    if key_word is None:
                        reward = 0.0
                        print("Warning: Can not find key word in single selection question.")
                    else:
                        if options_for_answers is None:
                            reward = 0.0
                            print("Warning: Can not extract options in single selection question.")
                        else:
                            other_options = [x for x in options_for_answers if x != ground_truth]
                            other_options_in_answer = any([x.lower() in student_answer.lower() for x in other_options])
                            pattern = re.compile(key_word, flags=re.IGNORECASE)
                            key_word_in_answer = pattern.search(student_answer)
                            if key_word_in_answer and other_options_in_answer != True:
                                reward = 1.0
                                student_answer_sent_count = self._count_clauses(student_answer)
                                if student_answer_sent_count > 2:
                                    penality_reward = (student_answer_sent_count - 2) * 0.2
                                    reward = reward - penality_reward
                                    reward = reward if reward > 0.0 else 0
                            else: 
                                reward = 0.0
        else:
            reward = 0.0
            print("Warning: Invalid Spational Reasoning Corpus, which")

        return reward


    def _extract_key_word_in_single_selection_question(self, question: str):
        for key_word in self.KEY_WORDS_FOR_WHICH_QUESTIONS:
            if key_word in question.lower().split():
                return key_word

        return None

    def _extract_which_opitons(self, text: str) -> Tuple[bool, Optional[List[str]]]:
        items = None
        which_pattern = re.compile(r"\bwhich\b", flags=re.IGNORECASE)
        found_which = which_pattern.search(text)
        if found_which:
            option_pattern = re.compile(r'options:\s*(.+)', flags=re.IGNORECASE)
            match_option = option_pattern.search(text)
            if match_option:
                # 拿到 Options: 后面那段内容
                tail = match_option.group(1)
                # 按分号拆分，去除两端空白，并过滤空字符串
                items = [item.strip() for item in tail.split(';') if item.strip()]
            
        return found_which, items


orms['external_sr_ORM'] = SpatialReasoningORM
orms['external_if_ORM'] = InstructionFollowingORM
orms['external_wothink_format'] = MathAnswerFormat