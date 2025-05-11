import copy
import random
import torch
from torch import nn

from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList, TopKLogitsWarper, TopPLogitsWarper, \
    TemperatureLogitsWarper
import ray
import logging

log = logging.getLogger("ray")

PROMPT_TEMPLATE = {
    "llama": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    "mistral_instruct_v1": "<s> [INST] {} [/INST]",
    "mistral_instruct_v2": "<s> [INST] {} [/INST]",
    "mistral_instruct_v3": "<s>[INST] {} [/INST]"
}


class TextGenEnv():
    def __init__(
            self,
            args,
            data_fetcher,
            policy_model_device,
            worker_id
    ):
        self.args = args
        # 文本生成模型初始化
        self.policy_model_path = args.policy_model_path
        self.policy_model_device = torch.device(policy_model_device)
        self.policy_model = AutoModelForCausalLM.from_pretrained(self.policy_model_path, torch_dtype=torch.float16).to(
            self.policy_model_device).eval().requires_grad_(False)
        self.policy_tokenizer = AutoTokenizer.from_pretrained(self.policy_model_path)
        self.policy_tokenizer.pad_token_id = self.policy_tokenizer.eos_token_id
        self.policy_template = PROMPT_TEMPLATE[args.policy_model_type]

        if self.args.state_transition == "v0":
            self.rank_topk_ouputs = self.rank_topk_ouputs_v0
        else:
            self.rank_topk_ouputs = self.rank_topk_ouputs_v1
        # 数据集初始化
        self.data_fetcher = data_fetcher
        self.data_count = len(self.data_fetcher)
        self.start_data_index = int(max(worker_id, 0) * self.data_count // args.num_agents)
        self.end_data_index = min(int(self.start_data_index + self.data_count // args.num_agents),
                                  self.data_count)
        log.info("start_data_index:{}".format(self.start_data_index))
        log.info("end_data_index:{}".format(self.end_data_index))
        self.data_index = self.start_data_index
        self.repeat_index = 1
        self.repeat_cnt = args.repeat_cnt
        if args.continue_train:
            self.data_index = min(self.end_data_index, int(self.start_data_index + 2 * args.reload_episode))
        log.info("data_index:{}".format(self.data_index))
        log.info("input_data_case:{}".format(self.data_fetcher[self.data_index]))
        self.max_episode = args.max_episode
        self.topk = args.topk
        self.topp = args.topp
        self.length_penalty = args.length_penalty
        self.max_new_token = args.max_new_token
        self.temperature = args.temperature

        # instantiate logits processors and wraper
        self.logits_wraper = LogitsProcessorList(
            [TopKLogitsWarper(args.topk), TopPLogitsWarper(top_p=self.topp), TemperatureLogitsWarper(self.temperature)])

        self.instruction = ""  # 指令
        self.generate_ids = []  # 已生成的token
        self.new_token_cnt = 0  # 已生成token数
        self.sum_logprobs = None  # 当前累积logit值
        self.last_reward = 0.0  # 未采纳新token前已经获得的sum_logit值
        self.curr_input_ids = None  # 生成new_token的输入input_id，
        self.curr_new_token_ids_list = None  # 采样的new_token
        self.curr_new_outputs_list = None  # 以self.curr_input_ids+curr_new_token_id为输入的模型输出 state
        self.next_new_outputs_list = None  # 对应self.curr_new_outputs_list下一个token next_state

    def get_raw_output(self, instruction):
        generation_config = {"do_sample": False, "max_new_tokens": self.max_new_token}
        input_text = self.policy_template.format(instruction)

        inputs = self.policy_tokenizer(input_text, return_tensors="pt").to(self.policy_model_device)

        output_ids = self.policy_model.generate(**inputs, **generation_config)[0]
        response = self.policy_tokenizer.decode(output_ids[len(inputs.input_ids[0]):], skip_special_tokens=True)
        return response

    def get_proxy_output(self, instruction, agent):
        input_text = self.policy_template.format(instruction)
        log.info("***model input***")
        log.info(input_text)
        model_inputs = self.policy_tokenizer(input_text, return_tensors="pt").to(self.policy_model_device)
        self.new_token_cnt = 0
        self.generate_ids = []

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
            accept_idx = 0
            action_list, action_prob_list = agent.select_action(state_list, train=False)
            for i in range(len(action_list)):
                if action_list[i] == 1:
                    accept_idx = i
                    break
            log.info(
                "new_token_cnt:{}/{}, action_list:{}, accept_idx: {}".format(self.new_token_cnt, self.max_new_token,
                                                                             action_list, accept_idx))
            accept_token_id = self.curr_new_token_ids_list[accept_idx][0].tolist()[-1]
            self.generate_ids.append(accept_token_id)
            self.new_token_cnt += 1

            self.curr_input_ids = torch.cat(
                (self.curr_input_ids, self.curr_new_token_ids_list[accept_idx].to(self.policy_model_device)), dim=-1)
            last_outputs = curr_new_outputs_list[accept_idx]

            if accept_token_id == self.policy_tokenizer.eos_token_id or self.new_token_cnt >= self.max_new_token:
                end_of_generate = True
        completion = self.policy_tokenizer.decode(self.generate_ids, skip_special_tokens=True)
        return completion

    def reset(self):
        """
            从数据集里取一个新的input
        """
        while True:
            if self.data_index == self.end_data_index:
                self.data_index = self.start_data_index
            self.instruction = self.data_fetcher[self.data_index]["prompt"]
            input_text = self.policy_template.format(self.instruction)
            if self.repeat_index >= self.repeat_cnt:
                self.data_index += 1
                self.repeat_index = 1
            else:
                self.repeat_index += 1
            self.new_token_cnt = 0
            self.generate_ids = []
            self.sum_logprobs = 0.0
            model_inputs = self.policy_tokenizer(input_text, return_tensors="pt").to(self.policy_model_device)

            input_ids = model_inputs.input_ids
            attention_mask = model_inputs.attention_mask
            self.curr_input_ids = input_ids
            self.prompt_len = len(input_ids[0])

            outputs = self.policy_model(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        output_hidden_states=True)
            # 只有在候选token数大于1时才调用模型进行决策
            end_of_generate, curr_new_token_ids_list, curr_new_token_logprobs_list, curr_new_outputs_list = self.rank_topk_ouputs(
                input_ids, outputs)
            if not end_of_generate:
                break

        state_list = [item['hidden_states'][-1][0, -1].detach().tolist() for item in curr_new_outputs_list]
        reward_list = self.reward_fun(curr_new_token_logprobs_list)
        return state_list, reward_list, self.prompt_len

    def reward_fun(self, curr_new_token_logprobs_list):
        if self.args.reward_type == "sum_logit":
            sum_logprobs_list = [self.sum_logprobs + curr_logprobs for curr_logprobs in curr_new_token_logprobs_list]
            # 选择对应token后的sum_logprobs值
            reward_list = [sum_logprobs / ((self.new_token_cnt + 1) ** self.length_penalty) for sum_logprobs in
                           sum_logprobs_list]
        elif self.args.reward_type == "logit":
            reward_list = [curr_logprobs * (self.new_token_cnt + 1) / ((self.new_token_cnt + 1) ** self.length_penalty)
                           for curr_logprobs in curr_new_token_logprobs_list]
        else:
            reward_list = curr_new_token_logprobs_list
        return reward_list

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
            sorted_scores = sorted_scores.tolist()[0]
            self.new_token_cnt += 1
            self.sum_logprobs += sorted_scores[0]
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
            self.next_new_outputs_list.append(self.curr_new_outputs_list[0])
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
        # top1 token放在最后一个，增加探索性
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
            sorted_scores = sorted_scores.tolist()[0]
            self.new_token_cnt += 1
            self.sum_logprobs += sorted_scores[0]
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

    def sample_action(self, state_list):
        action_list = [random.choice([0, 1]) for _ in range(len(state_list))]
        return action_list

    def step(self, action_list, reward_list, valid_action_idx=None):
        '''
        Args:
            action_list: 强化模型对所有topk_outputs['hidden_states']输出0/1，
            reward_list: 每个候选token的即时奖励

        Returns:
            self.current_outputs_list: next_state
             transition: multi_transition
        '''
        if valid_action_idx is None:
            valid_action_idx = [i for i in range(len(action_list))]
        deny_reward_list = reward_list[:]
        transition = []
        # 按顺序采纳第一个action=1的token, 如果所有action都为0，强制接受最后一个
        action = 0
        accept_idx = -1
        for i in range(len(action_list) - 1):
            valid_i = valid_action_idx[i]
            if action_list[i] == 1:
                state = self.curr_new_outputs_list[valid_i]['hidden_states'][-1][0, -1].detach().tolist()
                action = 1
                accept_idx = valid_i
                break
            else:
                state = self.curr_new_outputs_list[valid_i]['hidden_states'][-1][0, -1].detach().tolist()
                next_state = self.next_new_outputs_list[valid_i]['hidden_states'][-1][0, -1].detach().tolist()
                reward = deny_reward_list[i]
                done = False
                transition.append([state, action, reward, next_state, done])
        if action == 0:
            # 说明所有action都为0，强制接受最后一个
            state = self.curr_new_outputs_list[-1]['hidden_states'][-1][0, -1].detach().tolist()
            action = 1

        # accept_transition
        self.new_token_cnt += 1
        self.sum_logprobs += self.curr_new_token_logprobs_list[accept_idx]
        accept_token_id = self.curr_new_token_ids_list[accept_idx][0].tolist()[-1]

        reward = reward_list[accept_idx]
        self.curr_input_ids = torch.cat(
            (self.curr_input_ids, self.curr_new_token_ids_list[accept_idx].to(self.policy_model_device)), dim=-1)
        self.generate_ids.append(accept_token_id)

        if accept_token_id == self.policy_tokenizer.eos_token_id or self.new_token_cnt >= self.max_new_token:
            end_of_generate = True
            next_state = state[:]
            state_list = []
            new_reward_list = []
        else:
            end_of_generate, curr_new_token_ids_list, curr_new_token_logprobs_list, curr_new_outputs_list = self.rank_topk_ouputs(
                self.curr_input_ids, self.curr_new_outputs_list[accept_idx])

            next_state = curr_new_outputs_list[0]['hidden_states'][-1][0, -1].detach().tolist()

            state_list = [item['hidden_states'][-1][0, -1].detach().tolist() for item in curr_new_outputs_list]
            new_reward_list = self.reward_fun(curr_new_token_logprobs_list)  # action=1

        if end_of_generate:
            completion = self.policy_tokenizer.batch_decode([self.generate_ids], skip_special_tokens=True)[0]
            done = True
            llm_io = {"instruction": self.instruction, "generate_result": completion}
            if self.args.sft_as_reward_baseline:
                llm_io["sft_generate_result"] = self.get_raw_output(self.instruction)
        else:
            completion = ''
            done = False
            llm_io = {"instruction": self.instruction, "generate_result": completion}
        transition.append([state, action, reward, next_state, done])
        return state_list, new_reward_list, transition, done, llm_io
