import random
import numpy as np
from collections import deque

import torch
import wandb

from reward_model import RewardModel
from llm_env import TextGenEnv
from SAC import DistriDiscreteSAC
import ray
import logging
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

log = logging.getLogger("ray")


# ====================================  WORKER CONSTRUCTION  ===============================

class ParameterServer:
    def __init__(self, weights):
        self.weights = weights

    def push(self, weights):
        self.weights = weights

    def pull(self):
        return self.weights


# ====================================  ReplayBuffer CONSTRUCTION  ===============================
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])

    def add_batch(self, batch_transition):
        self.buffer.extend(batch_transition)

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    def size(self):
        return len(self.buffer)


class Learner():
    def __init__(self, all_args):
        super().__init__()
        self.args = all_args
        self.learner_device = all_args.learner_device
        self.replay_buffer = ReplayBuffer(capacity=all_args.buffer_capacity)
        self.episode = 0
        '''init reward model'''
        self.reward_model_list = []
        self.reward_model_type = all_args.reward_model_type
        for i in range(len(all_args.reward_model_type)):
            self.reward_model_list.append(RewardModel(reward_model_type=all_args.reward_model_type[i],
                                                      reward_model_path=all_args.reward_model_path[i],
                                                      reward_device=all_args.reward_model_device[i]))

        '''init agent'''
        self.agent = DistriDiscreteSAC(self.args, self.args.state_dim,
                                       self.args.action_dim, is_learner=True, worker_id=-1,
                                       device=self.learner_device)

        if self.args.continue_train:
            log.info("reload_path:{}".format(self.args.reload_path))
            try:
                self.agent.load(self.args.reload_path, episode=self.args.reload_episode)
            except Exception:
                log.error("No models detected!")
                raise
            self.episode = self.args.reload_episode + 1

        '''init wandb'''
        wandb.init(config=all_args.__dict__,
                   entity=all_args.wandb_entity_name,
                   project=all_args.wandb_project_name,
                   name=f"{all_args.algo_name}_{all_args.dataset}_{all_args.run_index}",
                   dir=all_args.save_path,
                   group=f"{all_args.algo_name}_{all_args.dataset}_{all_args.run_index}")

    def update(self, ps, curr_buffer, llm_io, logging_dict):
        multi_reward = 0
        multi_reward_list = []
        for i in range(len(self.reward_model_list)):
            curr_reward_score = self.reward_model_list[i].get_score([llm_io["instruction"]], [llm_io["generate_result"]])[0]
            logging_dict["reward_loss/{}_reward".format(self.reward_model_type[i])] = curr_reward_score
            multi_reward += self.args.reward_multiplier[i] * curr_reward_score
            multi_reward_list.append(curr_reward_score)
        log.info("generate_result: {}\nmulti_reward_list: {}".format(llm_io["generate_result"], multi_reward_list))

        if self.args.sft_as_reward_baseline:
            assert "sft_generate_result" in llm_io, "can't get sft_generate_result from llm_io"
            sft_multi_reward = 0
            sft_multi_reward_list = []
            for i in range(len(self.reward_model_list)):
                curr_sft_reward_score = self.reward_model_list[i].get_score([llm_io["instruction"]], [llm_io["sft_generate_result"]])[0]
                logging_dict["reward_loss/{}_sft_reward".format(self.reward_model_type[i])] = curr_sft_reward_score
                sft_multi_reward += self.args.reward_multiplier[i] * curr_sft_reward_score
                sft_multi_reward_list.append(curr_sft_reward_score)
            log.info("sft_generate_result: {}\nsft_multi_reward_list: {}".format(llm_io["sft_generate_result"], sft_multi_reward_list))
            final_reward = multi_reward - sft_multi_reward
        else:
            final_reward = multi_reward - self.args.reward_baseline
        final_reward = final_reward * self.args.baseline_multiplier

        log.info("final_reward: {}".format(final_reward))
        curr_buffer[-1][-3] += final_reward

        if self.args.postback_reward == 1:
            discounted_reward = final_reward
            for i in range(len(curr_buffer) - 2, 0, -1):
                if curr_buffer[i][1] == 1:
                    discounted_reward *= self.args.gamma
                curr_buffer[i][-3] += discounted_reward
        self.replay_buffer.add_batch(batch_transition=curr_buffer)
        logging_dict["train/replay_buffer_size"] = self.replay_buffer.size()
        logging_dict["reward_loss/multi_reward"] = multi_reward
        logging_dict["reward_loss/final_reward"] = final_reward

        wandb.log(logging_dict)
        if self.replay_buffer.size() > self.args.batch_size * 64:
            self.agent.update(self.replay_buffer, wandb)
            wandb.log({"train/update_episode": self.episode})
            self.episode += 1
            if self.episode % self.args.save_interval == 0:
                self.agent.save(self.args.save_path, self.episode)
            ps.push(self.agent.get_weights())

    def get_weights(self):
        return self.agent.get_weights()


@ray.remote(num_gpus=1.0)
class Worker(object):
    def __init__(self, train_data, worker_id: int, all_args):
        super().__init__()
        self.worker_id = worker_id
        self.args = all_args
        self.model_device = torch.device("cuda:0")
        self.data_fetcher = train_data
        '''init environment'''
        self.env = TextGenEnv(args=self.args, data_fetcher=self.data_fetcher, policy_model_device=self.model_device,
                              worker_id=worker_id)
        '''init agent'''
        self.agent = DistriDiscreteSAC(self.args, self.args.state_dim, self.args.action_dim, is_learner=False,
                                       worker_id=worker_id, device=self.model_device)

    def run(self, ps):
        '''Worker receive actor from Learner'''
        weights = ps.pull()
        self.agent.set_weights(weights, self.model_device)
        '''Worker explore_with_env save the training data to replay_buffer'''
        # accumulated transition is treated as the new replay buffer for the learner
        accumulated_transition, llm_io, logging_dict = self.agent.explore_with_env(self.env)
        # print(self.env.tokenizer.decode(self.env.current_ids[0], skip_special_tokens=False))
        # wandb.log(logging_dict)

        return accumulated_transition, llm_io, logging_dict, self.worker_id
