import argparse
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList, TopKLogitsWarper, TopPLogitsWarper, \
    TemperatureLogitsWarper
import sys
sys.path.append("..")
sys.path.append("../..")
from src.reward_model import RewardModel
from src.SAC import DistriDiscreteSAC
import torch
import torch.nn as nn
import ray
import logging
import copy

log = logging.getLogger("ray")

PROMPT_TEMPLATE = {
    "llama": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    "mistral_instruct_v1": "<s> [INST] {} [/INST]",
    "mistral_instruct_v2": "<s> [INST] {} [/INST]",
    "mistral_instruct_v3": "<s>[INST] {} [/INST]"
}


class MARAGenerator():
    def __init__(
            self,
            args
    ):
        self.args = args
        # 文本生成模型初始化
        self.policy_model_path = args.policy_model_path
        self.policy_model_device = torch.device(args.policy_model_device)
        self.policy_model = AutoModelForCausalLM.from_pretrained(self.policy_model_path, torch_dtype=torch.float16).to(
            self.policy_model_device).eval().requires_grad_(False)
        self.policy_tokenizer = AutoTokenizer.from_pretrained(self.policy_model_path)
        self.policy_tokenizer.pad_token_id = self.policy_tokenizer.eos_token_id
        self.policy_template = PROMPT_TEMPLATE[args.policy_model_type]

        self.topk = args.topk
        self.topp = args.topp
        self.length_penalty = args.length_penalty
        self.max_new_token = args.max_new_token
        self.temperature = args.temperature

        if args.eval_mode == "proxy":
            log.info("Execute proxy generate.....")
            # instantiate logits processors and wraper
            self.logits_wraper = LogitsProcessorList(
                [TopKLogitsWarper(args.topk), TopPLogitsWarper(top_p=self.topp),
                 TemperatureLogitsWarper(self.temperature)])

            if self.args.state_transition == "v0":
                self.rank_topk_ouputs = self.rank_topk_ouputs_v0
                log.info("using rank_topk_ouputs_v0")
            else:
                self.rank_topk_ouputs = self.rank_topk_ouputs_v1
                log.info("using rank_topk_ouputs_v1")

            '''init agent'''
            self.agent = DistriDiscreteSAC(args, args.state_dim, args.action_dim, is_learner=False, worker_id=-1,
                                           device=args.policy_model_device)
            try:
                self.agent.load(self.args.agent_model_path, episode=self.args.load_episode)
            except:
                log.error("No models detected!")
                raise
            self.instruction = ""  # 输入指令
            self.generate_ids = []  # 已生成的token
            self.new_token_cnt = 0  # 已生成token数
            self.curr_input_ids = None  # 生成new_token的输入input_id，
            self.curr_new_token_ids_list = None  # 采样的new_token
            self.sum_logprobs = None
            self.curr_new_outputs_list = None  # 以self.curr_input_ids+curr_new_token_id为输入的模型输出
            self.next_new_outputs_list = None  # 对应self.curr_new_outputs_list下一个token
            if args.serial_action:
                self.is_new_token = True
                self.chosen_indices = None
                self.cand_chosen_idx = 0
                self.last_outputs = None
                self.attention_mask = None
                self.position_id = None
            # proxy_detail
            self.gen_info = {"gen_token_cnt": 0,  # 每个样例生成的结果长度，len(gen_token_cnt_list)=sample_cnt
                             "proxy_token_cnt": 0,  # 需要经过agent进行决策的token长度，len(proxy_token_cnt_listt)=sample_cnt
                             "cand_token_dict": {},  # 每个决策位的候选token数，1<=cand_token_cnt<=topK，0表示不需要决策
                             "accept_index_dict": {}  # 每个决策位的选择第几个token， accept_idx
                             }


        elif args.eval_mode == "BON":
            log.info("Execute BON generate.....")
            self.reward_model_device = args.reward_model_device
            self.reward_model = RewardModel(reward_model_type=self.args.reward_model_type,
                                            reward_model_path=self.args.reward_model_path,
                                            reward_device=self.reward_model_device)
            self.N = args.N

        elif args.eval_mode == "aligner":
            self.aligner_model_device = args.aligner_model_device
            self.aligner_model = AutoModelForCausalLM.from_pretrained(args.aligner_model_path,
                                                                      torch_dtype=torch.float16).to(
                self.aligner_model_device)

            self.aligner_tokenizer = AutoTokenizer.from_pretrained(args.aligner_model_path, use_fast=False)
            self.aligner_template = 'BEGINNING OF CONVERSATION: USER: Edit the following Question-Answer pair to make it more helpful and harmless: {question} | {answer} ASSISTANT:'

        else:
            if args.do_sample:
                log.info("Execute base generate with sample strategy...")
            else:
                log.info("Execute base generate with greedy strategy...")

    def get_input_text(self, chat_pair, add_prompt_template=False):
        if isinstance(chat_pair, str):  # chat_pair=instruction
            instruction = chat_pair
            if add_prompt_template:
                messages = [{"role": "user", "content": instruction}]
                input_text = self.policy_tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                input_text = instruction
        else:  # chat_pair = [{"role": "user", "content": instruction}]
            input_text = self.policy_tokenizer.apply_chat_template(
                chat_pair,
                tokenize=False,
                add_generation_prompt=True
            )
        return input_text

    def get_raw_output(self, chat_pair, add_prompt_template=True):
        if self.args.do_sample:
            generation_config = {"do_sample": True, "max_new_tokens": self.max_new_token, "top_k": self.topk,
                                 "top_p": self.topp, "temperature": self.temperature}
        else:
            generation_config = {"do_sample": False, "max_new_tokens": self.max_new_token}
        input_text = self.get_input_text(chat_pair, add_prompt_template=add_prompt_template)
        inputs = self.policy_tokenizer(input_text, return_tensors="pt").to(self.policy_model_device)

        output_ids = self.policy_model.generate(**inputs, **generation_config)[0]
        response = self.policy_tokenizer.decode(output_ids[len(inputs.input_ids[0]):], skip_special_tokens=True)
        return response, {}

    def get_bon_output(self, chat_pair, add_prompt_template=True):
        generation_config = {"do_sample": True, "max_new_tokens": self.max_new_token, "top_k": self.topk,
                             "top_p": self.topp, "temperature": self.temperature, "num_return_sequences": self.N}
        input_text = self.get_input_text(chat_pair, add_prompt_template=add_prompt_template)
        log.info("***model input***")
        log.info(input_text)
        inputs = self.policy_tokenizer(input_text, return_tensors="pt").to(self.policy_model_device)

        output_ids = self.policy_model.generate(**inputs, **generation_config)

        generate_ids = [item[len(inputs.input_ids[0]):] for item in output_ids]

        candidate_result_list = self.policy_tokenizer.batch_decode(generate_ids, skip_special_tokens=True)
        log.info("***candidate_result_list***")
        log.info(candidate_result_list)

        # call reward model to choose the best
        if isinstance(chat_pair, str):
            reward_score_list = self.reward_model.get_score([chat_pair for _ in range(len(candidate_result_list))],
                                                            candidate_result_list)
        else:
            chat_pair_list = []
            for i in range(len(candidate_result_list)):
                candidate_chat_pair = chat_pair[:]
                candidate_chat_pair.append({"role": "assistant", "content": candidate_result_list[i]})
                chat_pair_list.append(candidate_chat_pair[:])
            reward_score_list = self.reward_model.get_score([], [], chat_pair_list)
        log.info("***reward_score_list***")
        log.info(reward_score_list)
        if self.args.keep_all_result:
            # 如果保存所有结果，则返回所有模型生成结果和对应打分
            output_with_score_list = []
            for i in range(len(candidate_result_list)):
                output_with_score_list.append(
                    {"output": candidate_result_list[i], "reward_score": reward_score_list[i]})
            return output_with_score_list, {}
        else:
            max_score = max(reward_score_list)
            for i in range(len(reward_score_list)):
                if reward_score_list[i] == max_score:
                    return candidate_result_list[i], {}

    def get_aligner_output(self, instruction):
        raw_output, _ = self.get_raw_output(instruction)
        aligner_input_text = self.aligner_template.format(question=instruction, answer=raw_output)

        inputs = self.aligner_tokenizer(aligner_input_text, return_tensors="pt").to(self.aligner_model_device)
        generation_config = {"do_sample": False, "max_new_tokens": self.max_new_token}
        output_ids = self.aligner_model.generate(**inputs, **generation_config)[0]
        aligner_response = self.aligner_tokenizer.decode(output_ids[len(inputs.input_ids[0]):],
                                                         skip_special_tokens=True)
        return aligner_response, {"raw_output": raw_output}

    def get_proxy_output_serial(self, chat_pair, add_prompt_template=True):
        input_text = self.get_input_text(chat_pair, add_prompt_template=add_prompt_template)
        log.info("***model input***")
        log.info(input_text)
        model_inputs = self.policy_tokenizer(input_text, return_tensors="pt").to(self.policy_model_device)
        self.new_token_cnt = 0
        self.generate_ids = []
        self.gen_info = {"gen_token_cnt": 0,  # 每个样例生成的结果长度，len(gen_token_cnt_list)=sample_cnt
                         "proxy_token_cnt": 0,  # 需要经过agent进行决策的token长度，len(proxy_token_cnt_listt)=sample_cnt
                         "cand_token_dict": {},  # 每个决策位的候选token数，1<=cand_token_cnt<=topK，0表示不需要决策
                         "accept_index_dict": {}  # 每个决策位的选择第几个token， accept_idx
                         }

        input_ids = model_inputs.input_ids
        self.attention_mask = model_inputs.attention_mask
        self.curr_input_ids = input_ids
        self.last_outputs = self.policy_model(input_ids=input_ids,
                                              attention_mask=self.attention_mask,
                                              output_hidden_states=True)

        # 一个token一个token地进行状态转移
        self.is_new_token = True
        self.cand_chosen_idx = 0
        self.position_id = None
        end_of_generate = False
        while not end_of_generate:
            end_of_generate, self.chosen_indices = self.rank_topk_ouputs_serial(self.curr_input_ids, self.last_outputs)
            self.is_new_token = False
            if end_of_generate:
                break
            accept_idx = self.args.default_action_idx
            curr_new_outputs_list = []
            for i in range(len(self.chosen_indices)):
                _, curr_new_outputs = self.rank_topk_ouputs_serial(self.curr_input_ids, self.last_outputs)
                curr_new_outputs_list.append(curr_new_outputs)
                curr_state = curr_new_outputs['hidden_states'][-1][0, -1].detach().tolist()

                action_list, probs = self.agent.select_action([curr_state], train=False)
                log.info("***action_1_probs***")
                log.info(probs.tolist()[0])
                log.info("***curr_action***")
                log.info(action_list)
                if action_list[0] == self.args.accept_action:
                    accept_idx = i
                    break
            self.is_new_token = True
            log.info(
                "new_token_cnt:{}/{}, accept_idx: {}".format(self.new_token_cnt, self.max_new_token, accept_idx))
            accept_token_id = self.chosen_indices[accept_idx]
            self.generate_ids.append(accept_token_id)
            self.new_token_cnt += 1
            self.gen_info["gen_token_cnt"] += 1
            self.gen_info["proxy_token_cnt"] += 1
            if len(self.chosen_indices) not in self.gen_info["cand_token_dict"]:
                self.gen_info["cand_token_dict"][len(self.chosen_indices)] = 1
            else:
                self.gen_info["cand_token_dict"][len(self.chosen_indices)] += 1
            if accept_idx not in self.gen_info["accept_index_dict"]:
                self.gen_info["accept_index_dict"][accept_idx] = 1
            else:
                self.gen_info["accept_index_dict"][accept_idx] += 1

            self.curr_input_ids = torch.cat(
                (self.curr_input_ids, torch.LongTensor([[accept_token_id]]).to(self.policy_model_device)), dim=-1)
            self.last_outputs = curr_new_outputs_list[accept_idx]

            if accept_token_id == self.policy_tokenizer.eos_token_id or self.new_token_cnt >= self.max_new_token:
                end_of_generate = True
        completion = self.policy_tokenizer.decode(self.generate_ids, skip_special_tokens=True)
        return completion, copy.deepcopy(self.gen_info)

    def get_proxy_output(self, chat_pair, add_prompt_template=True):
        input_text = self.get_input_text(chat_pair, add_prompt_template=add_prompt_template)
        log.info("***model input***")
        log.info(input_text)
        model_inputs = self.policy_tokenizer(input_text, return_tensors="pt").to(self.policy_model_device)
        self.new_token_cnt = 0
        self.generate_ids = []
        self.gen_info = {"gen_token_cnt": 0,  # 每个样例生成的结果长度，len(gen_token_cnt_list)=sample_cnt
                         "proxy_token_cnt": 0,  # 需要经过agent进行决策的token长度，len(proxy_token_cnt_listt)=sample_cnt
                         "cand_token_dict": {},  # 每个决策位的候选token数，1<=cand_token_cnt<=topK，0表示不需要决策
                         "accept_index_dict": {}  # 每个决策位的选择第几个token， accept_idx
                         }

        input_ids = model_inputs.input_ids
        attention_mask = model_inputs.attention_mask
        self.curr_input_ids = input_ids
        last_outputs = self.policy_model(input_ids=input_ids,
                                         attention_mask=attention_mask,
                                         output_hidden_states=True)

        end_of_generate = False
        while not end_of_generate:
            end_of_generate, curr_new_token_ids_list, curr_new_token_logprobs_list, curr_new_outputs_list = self.rank_topk_ouputs(
                self.curr_input_ids, last_outputs)
            if end_of_generate:
                break

            state_list = [item['hidden_states'][-1][0, -1].detach().tolist() for item in curr_new_outputs_list]
            accept_idx = self.args.default_action_idx
            if self.args.proxy_strategy == "max_prob":
                accept_idx, probs = self.agent.select_action_with_max_prob(state_list)
                action_list = probs
            else:
                action_list, probs = self.agent.select_action(state_list, train=False)
                log.info("***action_1_probs***")
                log.info(probs.tolist())
                log.info("***action_list***")
                log.info(action_list)
                if self.args.proxy_strategy == "random":
                    cand_idx_list = []
                    for i in range(len(action_list)):
                        if action_list[i] == 1:
                            cand_idx_list.append(i)
                    if len(cand_idx_list) != 0:
                        accept_idx = random.choice(cand_idx_list)
                else:  # top1
                    # 逆向选择的，如果都不选，强制接受top1
                    if self.args.reverse_choice:
                        accept_idx = -1
                    for i in range(len(action_list)):
                        if action_list[i] == self.args.accept_action:
                            accept_idx = i
                            break
            log.info(
                "new_token_cnt:{}/{}, action_list:{}, accept_idx: {}".format(self.new_token_cnt, self.max_new_token,
                                                                             action_list, accept_idx))
            accept_token_id = self.curr_new_token_ids_list[accept_idx][0].tolist()[-1]
            self.generate_ids.append(accept_token_id)
            self.new_token_cnt += 1
            self.gen_info["gen_token_cnt"] += 1
            self.gen_info["proxy_token_cnt"] += 1
            if len(action_list) not in self.gen_info["cand_token_dict"]:
                self.gen_info["cand_token_dict"][len(action_list)] = 1
            else:
                self.gen_info["cand_token_dict"][len(action_list)] += 1
            if accept_idx not in self.gen_info["accept_index_dict"]:
                self.gen_info["accept_index_dict"][accept_idx] = 1
            else:
                self.gen_info["accept_index_dict"][accept_idx] += 1

            self.curr_input_ids = torch.cat(
                (self.curr_input_ids, self.curr_new_token_ids_list[accept_idx].to(self.policy_model_device)), dim=-1)
            last_outputs = curr_new_outputs_list[accept_idx]

            if accept_token_id == self.policy_tokenizer.eos_token_id or self.new_token_cnt >= self.max_new_token:
                end_of_generate = True
        completion = self.policy_tokenizer.decode(self.generate_ids, skip_special_tokens=True)
        return completion, copy.deepcopy(self.gen_info)

    def one_step_transfer(self, pre_input_ids, past_key_values, new_token_id):
        attention_mask = torch.ones_like(pre_input_ids)
        attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_id = position_ids[:, -1:].to(self.policy_model_device)
        new_token_id = new_token_id.to(self.policy_model_device)

        new_outputs = self.policy_model(input_ids=new_token_id,
                                        attention_mask=attention_mask.to(self.policy_model_device),
                                        position_ids=position_id,
                                        past_key_values=past_key_values,
                                        output_hidden_states=True)
        new_input_ids = torch.cat((pre_input_ids, new_token_id), dim=-1)
        return new_input_ids, new_outputs

    def rank_topk_ouputs_v0(self, pre_input_ids, last_outputs):
        end_of_generate = False
        self.curr_new_token_ids_list = []  # 采样的new_token
        self.curr_new_token_logprobs_list = []  # 候选 token 的log_softmax 分数
        self.curr_new_outputs_list = []  # 以self.curr_input_ids+curr_new_token_id为输入的模型输出 state
        self.next_new_outputs_list = []  # next_state

        next_token_logits = last_outputs.logits[:, -1, :].clone()  # (batch_size, vocab_size)
        next_token_scores = nn.functional.log_softmax(next_token_logits, dim=-1)  # (batch_size, vocab_size)

        ori_sorted_scores, ori_sorted_indices = torch.sort(next_token_scores, descending=True)

        next_token_scores = self.logits_wraper(pre_input_ids, next_token_scores)
        sorted_scores, sorted_indices = torch.sort(next_token_scores, descending=True)
        chosen_indices = torch.masked_select(sorted_indices, sorted_scores != -float("Inf")).tolist()
        # 当候选token数只有1个且不为终止eos_token_id时直接一步转移
        while len(chosen_indices) == 1:
            self.new_token_cnt += 1
            self.gen_info["gen_token_cnt"] += 1
            self.generate_ids.append(chosen_indices[0])
            if chosen_indices[0] == self.policy_tokenizer.eos_token_id or self.new_token_cnt >= self.max_new_token:
                end_of_generate = True
                self.curr_new_token_ids_list.append(torch.LongTensor([[chosen_indices[0]]]))
                self.curr_new_token_logprobs_list.append(sorted_scores[0])
                self.curr_new_outputs_list.append(last_outputs)
                return end_of_generate, self.curr_new_token_ids_list, self.curr_new_token_logprobs_list, self.curr_new_outputs_list
            else:
                pre_input_ids, last_outputs = self.one_step_transfer(pre_input_ids,
                                                                     past_key_values=last_outputs.past_key_values,
                                                                     new_token_id=torch.LongTensor(
                                                                         [[chosen_indices[0]]]))
                self.curr_input_ids = pre_input_ids

                next_token_logits = last_outputs.logits[:, -1, :].clone()  # (batch_size, vocab_size)
                next_token_scores = nn.functional.log_softmax(next_token_logits, dim=-1)  # (batch_size, vocab_size)
                ori_sorted_scores, ori_sorted_indices = torch.sort(next_token_scores, descending=True)
                next_token_scores = self.logits_wraper(pre_input_ids, next_token_scores)
                sorted_scores, sorted_indices = torch.sort(next_token_scores, descending=True)
                chosen_indices = torch.masked_select(sorted_indices, sorted_scores != -float("Inf")).tolist()

        sorted_scores = sorted_scores.tolist()[0]
        attention_mask = torch.ones_like(pre_input_ids)
        attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_id = position_ids[:, -1:].to(self.policy_model_device)
        # log.info("len(chosen_indices):{}".format(len(chosen_indices)))
        for i in range(len(chosen_indices)):
            new_token_id = torch.LongTensor([[chosen_indices[i]]])
            self.curr_new_token_ids_list.append(new_token_id)
            self.curr_new_token_logprobs_list.append(sorted_scores[i])
            curr_next_output = self.policy_model(input_ids=new_token_id.to(self.policy_model_device),
                                                 attention_mask=attention_mask.to(self.policy_model_device),
                                                 position_ids=position_id.to(self.policy_model_device),
                                                 past_key_values=last_outputs.past_key_values,
                                                 output_hidden_states=True)
            self.curr_new_outputs_list.append(curr_next_output)
        if self.args.reverse_choice:
            self.curr_new_token_ids_list = self.curr_new_token_ids_list[::-1]
            self.curr_new_token_logprobs_list = self.curr_new_token_logprobs_list[::-1]
            self.curr_new_outputs_list = self.curr_new_outputs_list[::-1]
            self.next_new_outputs_list = self.curr_new_outputs_list[1:]
            self.next_new_outputs_list.append(self.curr_new_outputs_list[-1])
        else:
            # 加上一个移位token
            new_token_id = ori_sorted_indices[:, len(chosen_indices)].view(1, -1)
            self.next_new_outputs_list = self.curr_new_outputs_list[1:]
            curr_next_output = self.policy_model(input_ids=new_token_id.to(self.policy_model_device),
                                                 attention_mask=attention_mask.to(self.policy_model_device),
                                                 position_ids=position_id.to(self.policy_model_device),
                                                 past_key_values=last_outputs.past_key_values,
                                                 output_hidden_states=True)
            self.next_new_outputs_list.append(curr_next_output)
        return end_of_generate, self.curr_new_token_ids_list, self.curr_new_token_logprobs_list, self.curr_new_outputs_list

    def rank_topk_ouputs_v1(self, pre_input_ids, last_outputs):
        end_of_generate = False
        self.curr_new_token_ids_list = []  # 采样的new_token
        self.curr_new_token_logprobs_list = []  # 候选 token 的log_softmax 分数
        self.curr_new_outputs_list = []  # 以self.curr_input_ids+curr_new_token_id为输入的模型输出 state
        self.next_new_outputs_list = []  # next_state

        next_token_logits = last_outputs.logits[:, -1, :].clone()  # (batch_size, vocab_size)
        next_token_scores = nn.functional.log_softmax(next_token_logits, dim=-1)  # (batch_size, vocab_size)

        next_token_scores = self.logits_wraper(pre_input_ids, next_token_scores)
        sorted_scores, sorted_indices = torch.sort(next_token_scores, descending=True)
        chosen_indices = torch.masked_select(sorted_indices, sorted_scores != -float("Inf")).tolist()
        # 当候选token数只有1个且不为终止eos_token_id时直接一步转移
        while len(chosen_indices) == 1:
            self.new_token_cnt += 1
            self.gen_info["gen_token_cnt"] += 1
            self.generate_ids.append(chosen_indices[0])
            if chosen_indices[0] == self.policy_tokenizer.eos_token_id or self.new_token_cnt >= self.max_new_token:
                end_of_generate = True
                self.curr_new_token_ids_list.append(torch.LongTensor([[chosen_indices[0]]]))
                self.curr_new_token_logprobs_list.append(sorted_scores[0])
                self.curr_new_outputs_list.append(last_outputs)
                return end_of_generate, self.curr_new_token_ids_list, self.curr_new_token_logprobs_list, self.curr_new_outputs_list
            else:
                pre_input_ids, last_outputs = self.one_step_transfer(pre_input_ids,
                                                                     past_key_values=last_outputs.past_key_values,
                                                                     new_token_id=torch.LongTensor(
                                                                         [[chosen_indices[0]]]))
                self.curr_input_ids = pre_input_ids

                next_token_logits = last_outputs.logits[:, -1, :].clone()  # (batch_size, vocab_size)
                next_token_scores = nn.functional.log_softmax(next_token_logits, dim=-1)  # (batch_size, vocab_size)
                next_token_scores = self.logits_wraper(pre_input_ids, next_token_scores)
                sorted_scores, sorted_indices = torch.sort(next_token_scores, descending=True)
                chosen_indices = torch.masked_select(sorted_indices, sorted_scores != -float("Inf")).tolist()

        sorted_scores = sorted_scores.tolist()[0]
        attention_mask = torch.ones_like(pre_input_ids)
        attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_id = position_ids[:, -1:].to(self.policy_model_device)
        # log.info("len(chosen_indices):{}".format(len(chosen_indices)))
        for i in range(1, len(chosen_indices)):
            new_token_id = torch.LongTensor([[chosen_indices[i]]])
            self.curr_new_token_ids_list.append(new_token_id)
            self.curr_new_token_logprobs_list.append(sorted_scores[i])
            curr_next_output = self.policy_model(input_ids=new_token_id.to(self.policy_model_device),
                                                 attention_mask=attention_mask.to(self.policy_model_device),
                                                 position_ids=position_id.to(self.policy_model_device),
                                                 past_key_values=last_outputs.past_key_values,
                                                 output_hidden_states=True)
            self.curr_new_outputs_list.append(curr_next_output)
        # top1 token放在最后一个
        new_token_id = torch.LongTensor([[chosen_indices[0]]])
        self.curr_new_token_ids_list.append(new_token_id)
        self.curr_new_token_logprobs_list.append(sorted_scores[0])
        curr_next_output = self.policy_model(input_ids=new_token_id.to(self.policy_model_device),
                                             attention_mask=attention_mask.to(self.policy_model_device),
                                             position_ids=position_id.to(self.policy_model_device),
                                             past_key_values=last_outputs.past_key_values,
                                             output_hidden_states=True)
        self.curr_new_outputs_list.append(curr_next_output)

        # top1 的next_state是top2
        self.next_new_outputs_list = self.curr_new_outputs_list[1:]
        self.next_new_outputs_list.append(self.curr_new_outputs_list[0])
        return end_of_generate, self.curr_new_token_ids_list, self.curr_new_token_logprobs_list, self.curr_new_outputs_list

    def rank_topk_ouputs_serial(self, pre_input_ids, last_outputs):
        if self.is_new_token:
            end_of_generate = False
            next_token_logits = last_outputs.logits[:, -1, :].clone()  # (batch_size, vocab_size)
            next_token_scores = nn.functional.log_softmax(next_token_logits, dim=-1)  # (batch_size, vocab_size)
            next_token_scores = self.logits_wraper(pre_input_ids, next_token_scores)
            sorted_scores, sorted_indices = torch.sort(next_token_scores, descending=True)
            chosen_indices = torch.masked_select(sorted_indices, sorted_scores != -float("Inf")).tolist()
            # 当候选token数只有1个且不为终止eos_token_id时直接一步转移
            while len(chosen_indices) == 1:
                self.new_token_cnt += 1
                self.gen_info["gen_token_cnt"] += 1
                self.generate_ids.append(chosen_indices[0])
                if chosen_indices[0] == self.policy_tokenizer.eos_token_id or self.new_token_cnt >= self.max_new_token:
                    end_of_generate = True
                    return end_of_generate, None
                else:
                    pre_input_ids, last_outputs = self.one_step_transfer(pre_input_ids,
                                                                         past_key_values=last_outputs.past_key_values,
                                                                         new_token_id=torch.LongTensor(
                                                                             [[chosen_indices[0]]]))
                    self.curr_input_ids = pre_input_ids

                    next_token_logits = last_outputs.logits[:, -1, :].clone()  # (batch_size, vocab_size)
                    next_token_scores = nn.functional.log_softmax(next_token_logits, dim=-1)  # (batch_size, vocab_size)
                    next_token_scores = self.logits_wraper(pre_input_ids, next_token_scores)
                    sorted_scores, sorted_indices = torch.sort(next_token_scores, descending=True)
                    chosen_indices = torch.masked_select(sorted_indices, sorted_scores != -float("Inf")).tolist()

            attention_mask = torch.ones_like(pre_input_ids)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_id = position_ids[:, -1:].to(self.policy_model_device)
            # log.info("len(chosen_indices):{}".format(len(chosen_indices)))
            self.chosen_indices = chosen_indices
            self.cand_chosen_idx = 0
            self.attention_mask = attention_mask
            self.position_id = position_id
            self.last_outputs = last_outputs
            return end_of_generate, chosen_indices
        else:
            new_token_id = torch.LongTensor([[self.chosen_indices[self.cand_chosen_idx]]])
            curr_next_output = self.policy_model(input_ids=new_token_id.to(self.policy_model_device),
                                                 attention_mask=self.attention_mask.to(self.policy_model_device),
                                                 position_ids=self.position_id.to(self.policy_model_device),
                                                 past_key_values=self.last_outputs.past_key_values,
                                                 output_hidden_states=True)
            self.cand_chosen_idx += 1
            return False, curr_next_output

    def sample_action(self, state_list):
        action_list = [random.choice([0, 1]) for _ in range(len(state_list))]
        action_prob_list = [random.random() for _ in range(len(state_list))]
        return action_list, action_prob_list


if __name__ == "__main__":
    import time
    parser = argparse.ArgumentParser()
    # drl agent
    parser.add_argument("--agent_model_path",
                        default="../train_result/multi_reward/mistral_v3_2_1/run2/",
                        type=str,
                        help='Path to the drl policy model checkpoint')
    parser.add_argument("--load_episode", default=18000, type=int,
                        help="load the model trained for `load_episode` episode")
    parser.add_argument('--state_dim', type=int, default=4096)
    parser.add_argument('--action_dim', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=1024)

    # policy_model
    parser.add_argument("--policy_model_type", type=str, default="mistral_instruct_v3",
                        help="use to get instruction prompt")
    parser.add_argument("--policy_model_path", default="path2model/Mistral-7B-Instruct-v0.3",
                        type=str, help='Path to the llm model')
    parser.add_argument("--policy_model_device", type=str, default="cuda:0", help="specify the gpu")

    # reward_model
    parser.add_argument("--reward_model_type", type=str, default='beaver_reward')
    parser.add_argument("--reward_model_path", default="path2model/beaver-7b-v1.0-reward", type=str,
                        help='Path to the reward model checkpoint')
    parser.add_argument("--reward_model_device", type=str, default="cuda:1", help="specify the gpu")

    # text_generate control
    parser.add_argument("--proxy_strategy", type=str, default="top1", choices=["top1", "random", "max_prob"],
                        help="use when eval_mode is proxy")
    parser.add_argument("--do_sample", action='store_true', help="use when eval_mode is raw")
    parser.add_argument('--max_new_token', type=int, default=2048)
    parser.add_argument("--topk", default=40, type=int)
    parser.add_argument("--topp", default=0.95, type=float)
    parser.add_argument("--temperature", default=0.8, type=float)
    parser.add_argument("--min_tokens_to_keep", default=1, type=int)
    parser.add_argument("--length_penalty", default=1.0, type=float)
    parser.add_argument("--N", default=15, type=int, help="candidate sample cnt of BON")
    parser.add_argument("--keep_all_result", action='store_true', help="use when eval_mode is BON")
    parser.add_argument('--reverse_choice', action='store_true')
    parser.add_argument('--accept_action', type=int, default=1)
    parser.add_argument('--state_transition', type=str, default="v0")
    parser.add_argument('--default_action_idx', type=int, default=0,
                        help="accept token of default_action_idx if all action is 0")
    parser.add_argument("--serial_action", action='store_true', help="use when eval_mode is proxy")
    #
    parser.add_argument("--eval_mode", default="proxy", type=str, choices=["raw", "proxy", "BON", "dpo", "rlhf", "aligner"],
                        required=True)
    parser.add_argument("--eval_action", default="generate", type=str, help="eval_action in [generate, score]")
    parser.add_argument("--eval_exp_name", default="", type=str, help="extra info to distinguish eval results")

    # eval setting for eval_mode is aligner aligner/aligner-7b-v1.0
    parser.add_argument("--aligner_model_type", type=str, default="aligner_7b_v1",
                        help="use to get instruction prompt")
    parser.add_argument("--aligner_model_path", default="path2model/aligner-7b-v1.0",
                        type=str, help='Path to the llm model')
    parser.add_argument("--aligner_model_device", type=str, default="cuda:0", help="specify the gpu")
    args = parser.parse_args()
    proxy_generator = MARAGenerator(args)
    instruction = "Please introduce yourself."
    raw_result = proxy_generator.get_raw_output(instruction)
    begin_time = time.time()
    print("raw_result")
    print(raw_result)
    print("use time: {}s".format(time.time() - begin_time))
    begin_time = time.time()
    proxy_result = proxy_generator.get_proxy_output_serial(instruction)
    print("proxy_result")
    print(proxy_result)
    print("use time: {}s".format(time.time() - begin_time))
