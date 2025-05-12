import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import ray
import logging

log = logging.getLogger("ray")


# ====================================  NET CONSTRUCTION ===============================

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.uniform_(m.bias, a=-0.003, b=0.003)


def move_state_dict(state_dict, device):
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k] = v.to(device)
    return new_state_dict


class DiscreteActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=1024):
        super(DiscreteActor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.ln = nn.LayerNorm(state_dim)
        self.linear1 = nn.Linear(self.state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, int(hidden_dim / 4))
        # self.apply(weights_init_)
        self.output_linear = nn.Linear(int(hidden_dim / 4), self.action_dim)

    def forward(self, state):
        state = self.ln(state)
        x = torch.relu(self.linear1(state))
        x = torch.relu(self.linear2(x))
        output = torch.softmax(self.output_linear(x), dim=1)
        return output

    def sample(self, state):
        prob = self.forward(state)
        distribution = Categorical(torch.Tensor(prob))
        sample_action = distribution.sample().unsqueeze(-1).detach()
        z = (prob == 0.0).float() * 1e-8
        logprob = torch.log(prob + z)
        greedy_action = torch.argmax(prob, dim=-1).unsqueeze(-1)  # 1d tensor
        return sample_action, prob, logprob, greedy_action


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=1024):
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.ln = nn.LayerNorm(state_dim)

        # q1
        self.linear11 = nn.Linear(self.state_dim, hidden_dim)
        self.linear12 = nn.Linear(hidden_dim, int(hidden_dim / 4))
        self.linear13 = nn.Linear(int(hidden_dim / 4), self.action_dim)

        # q2
        self.linear21 = nn.Linear(self.state_dim, hidden_dim)
        self.linear22 = nn.Linear(hidden_dim, int(hidden_dim / 4))
        self.linear23 = nn.Linear(int(hidden_dim / 4), self.action_dim)
        # self.apply(weights_init_)

    def forward(self, state):
        state = self.ln(state)
        x1 = torch.relu(self.linear11(state))
        x1 = torch.relu(self.linear12(x1))
        q1 = self.linear13(x1)

        x2 = torch.relu(self.linear21(state))
        x2 = torch.relu(self.linear22(x2))
        q2 = self.linear23(x2)
        return q1, q2


# ====================================  ALGO CONSTRUCTION ===============================
class DistriDiscreteSAC:
    def __init__(self, args, state_dim=4096, action_dim=2, is_learner=False, worker_id=-1, device="cpu"):
        super(DistriDiscreteSAC, self).__init__()
        self.args = args
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.learner_device = torch.device(device)
        self.critic_net = Critic(state_dim, action_dim, args.hidden_dim).to(self.learner_device)
        self.actor_net = DiscreteActor(state_dim, action_dim, args.hidden_dim).to(self.learner_device)

        self.worker_id = worker_id
        self.episode = 0
        self.is_learner = is_learner
        if is_learner:
            self.batch_size = args.batch_size
            self.update_time = args.update_time
            self.gamma = args.gamma
            self.alpha = args.alpha
            self.lr_actor = args.lr_actor
            self.lr_critic = args.lr_critic
            self.max_grad_norm = args.max_grad_norm
            self.target_critic_net = Critic(state_dim, action_dim, args.hidden_dim).to(self.learner_device)
            self.target_critic_net.load_state_dict(self.critic_net.state_dict())
            self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=self.lr_critic)
            self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=self.lr_actor)

            self.lr_alpha = args.lr_alpha
            if args.target_entropy_type == 'simple':
                self.target_entropy = -args.target_entropy_factor * self.action_dim
            elif args.target_entropy_type == 'log':
                self.target_entropy = args.target_entropy_factor * -np.log(1 / self.action_dim)
            self.log_alpha = torch.tensor(np.log(self.alpha), dtype=torch.float, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr_alpha)

            self.training_step = 0
            self.batch_size = args.batch_size
            self.replace_tau = args.replace_tau
            self.accumulated_step = args.accumulated_step

    def cal_entropy(self, action_prob, action_log_prob, is_discrete=True):
        if is_discrete:
            # 离散版本，参考https://github.com/boyu-ai/Hands-on-RL/blob/main/%E7%AC%AC14%E7%AB%A0-SAC%E7%AE%97%E6%B3%95.ipynb
            entropy = -torch.sum(action_prob * action_log_prob, dim=1, keepdim=True)
            return entropy
        else:
            # 连续版本
            entropy = -action_log_prob
        return entropy

    def select_action(self, state, train=True):
        state = torch.tensor(state, dtype=torch.float).to(self.learner_device)
        sample_action, prob, logprob, greedy_action = self.actor_net.sample(state)
        if train:
            action = sample_action
        else:
            action = greedy_action
        choice_action = torch.ones_like(action)
        choice_prob = torch.gather(prob, 1, choice_action)
        action = action.squeeze(-1).tolist()
        return action, choice_prob

    def select_action_with_max_prob(self, state):
        state = torch.tensor(state, dtype=torch.float).to(self.learner_device)
        sample_action, prob, logprob, greedy_action = self.actor_net.sample(state)
        max_prob = torch.max(prob[:, 1]).item()
        if max_prob > 0.5:
            action = torch.argmax(prob[:, 1]).item()
        else:
            action = 0
        return action, prob

    def _kl_penalty(self, logprobs, ref_logprobs):
        kl_penalty = []
        for i in range(len(logprobs)):
            if self.args.kl_penalty == "kl":
                kl_penalty.append(logprobs[i] - ref_logprobs[i])

            elif self.args.kl_penalty == "abs":
                kl_penalty.append(abs(logprobs[i] - ref_logprobs[i]))

            elif self.args.kl_penalty == "mse":
                kl_penalty.append(0.5 * (logprobs[i] - ref_logprobs[i]) ** 2)

            elif self.args.kl_penalty == "full":
                kl_penalty.append(math.exp(logprobs[i]) * (logprobs[i] - ref_logprobs[i]))
            else:
                raise NotImplementedError
        return kl_penalty

    def compute_rewards(self, rl_log_probs, llm_log_probs):
        kl_divergence_estimate = self._kl_penalty(rl_log_probs, llm_log_probs)
        rewards = [-self.args.kl_ctl * item for item in kl_divergence_estimate]
        return rewards

    def explore_with_env(self, env):
        """
        Collect trajectories through the actor-environment interaction for a **single** environment instance.

        env: RL training environment. env.reset() env.step(). It should be a vector env.
        return: `[(state, action, reward, next_state, terminated) * self.accumulated_step] ` for off-policy
                logging_dict  for log show
        """
        accumulated_transition = []
        state_list, llm_logprob_list, input_len = env.reset()

        # 实际决策的步数
        action_step = 0
        # 决策结果为接受的次数
        accept_cnt = 0
        # 决策结果为拒绝的次数
        deny_cnt = 0
        # 被接受的token得分
        accept_token_avg_reward = []
        # 被拒绝的token得分
        deny_token_avg_reward = []
        while True:
            valid_state_list = []
            valid_llm_logprob_list = []
            valid_action_idx = []
            for i in range(len(state_list)):
                if np.isnan(np.array(state_list[i])).any():
                    continue
                valid_state_list.append(state_list[i])
                valid_llm_logprob_list.append(llm_logprob_list[i])
                valid_action_idx.append(i)
            state_list = valid_state_list
            llm_logprob_list = valid_llm_logprob_list

            action_list, choice_prob = self.select_action(state_list)
            if self.args.cond_prob:
                rl_final_log_prob_list = [choice_prob[0]]
                refuse_prob = 1 - choice_prob[0]
                for i in range(1, len(choice_prob)):
                    rl_final_log_prob_list.append(refuse_prob * choice_prob[i])
                    refuse_prob = refuse_prob * (1 - choice_prob[i])
                rl_logprob_list = torch.log_softmax(torch.Tensor(rl_final_log_prob_list), dim=0).tolist()
            else:
                rl_logprob_list = torch.log_softmax(choice_prob, dim=0).squeeze(-1).tolist()
            reward_list = self.compute_rewards(rl_logprob_list, llm_logprob_list)
            action_len = len(state_list)
            for i in range(action_len):
                if action_list[i] == 1:
                    accept_token_avg_reward.append(reward_list[i])
                else:
                    deny_token_avg_reward.append(reward_list[i])
            deny_cnt += (len(action_list) - sum(action_list))
            accept_cnt += sum(action_list)
            action_step += 1
            next_state_list, llm_logprob_list, transition, done, llm_io = env.step(action_list, reward_list,
                                                                                   valid_action_idx)

            accumulated_transition.extend(transition)

            if done:
                self.episode += 1
                if len(accept_token_avg_reward) > 0:
                    accept_token_avg_reward = sum(accept_token_avg_reward) / len(accept_token_avg_reward)
                else:
                    accept_token_avg_reward = 0.0
                if len(deny_token_avg_reward) > 0:
                    deny_token_avg_reward = sum(deny_token_avg_reward) / len(deny_token_avg_reward)
                else:
                    deny_token_avg_reward = 0.0
                logging_dict = {'episode_{}'.format(self.worker_id): self.episode,
                                'data_idx_{}'.format(self.worker_id): env.data_index,
                                "ep_action_step_{}".format(self.worker_id): action_step,
                                "ep_deny_cnt_{}".format(self.worker_id): deny_cnt,
                                "ep_accept_cnt_{}".format(self.worker_id): accept_cnt,
                                "ep_deny_avg_reward_{}".format(self.worker_id): deny_token_avg_reward,
                                "ep_accept_avg_reward_{}".format(self.worker_id): accept_token_avg_reward,
                                "total_len_{}".format(self.worker_id): input_len + action_step,
                                "generate_token_cnt_{}".format(self.worker_id): action_step,
                                }
                log.info(logging_dict)
                return accumulated_transition, llm_io, logging_dict
            else:
                state_list = next_state_list[:]

    def update(self, buffer, wandb):
        for i in range(self.update_time):
            state, action, reward, next_state, terminal = buffer.sample(self.batch_size)
            state = torch.tensor(state, dtype=torch.float).to(self.learner_device)
            action = torch.tensor(action, dtype=torch.int64).view(-1, 1).to(self.learner_device)
            reward = torch.tensor(reward, dtype=torch.float).view(-1, 1).to(self.learner_device)
            next_state = torch.tensor(next_state, dtype=torch.float).to(self.learner_device)
            terminal = torch.tensor(terminal, dtype=torch.float).view(-1, 1).to(self.learner_device)
            with torch.no_grad():
                next_state_action, next_action_prob, next_action_log_prob, _ = self.actor_net.sample(
                    next_state)
                q1_next_target, q2_next_target = self.target_critic_net(next_state)
                # v1
                min_qvalue = torch.sum(next_action_prob * torch.min(q1_next_target, q2_next_target), dim=1,
                                       keepdim=True)
                min_q_next_target = min_qvalue + self.log_alpha.exp() * self.cal_entropy(next_action_prob,
                                                                                         next_action_log_prob)
                next_q_value = reward + (1 - terminal) * self.gamma * min_q_next_target

            q1, q2 = self.critic_net(state)
            q1 = q1.gather(1, action.long())
            q2 = q2.gather(1, action.long())  # [batch, 1] , pick the actin-value for the given batched actions

            q1_loss = 0.5 * F.mse_loss(q1, next_q_value.detach())
            q2_loss = 0.5 * F.mse_loss(q2, next_q_value.detach())
            q_loss = q1_loss + q2_loss
            # update critic network
            self.critic_optimizer.zero_grad()
            q_loss.backward()
            nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()

            # update actor network
            q1_pi, q2_pi = self.critic_net(state)
            pi, pi_prob, pi_log_prob, _ = self.actor_net.sample(state)
            # v1
            min_qvalue = torch.sum(pi_prob * torch.min(q1_pi, q2_pi), dim=1, keepdim=True)  # 直接根据概率计算期望
            pi_loss = -torch.mean(self.log_alpha.exp() * self.cal_entropy(pi_prob, pi_log_prob) + min_qvalue)

            # pi_loss = (pi_prob*(self.alpha.detach() * pi_log_prob - min_q_pi)).sum(1).mean()
            self.actor_optimizer.zero_grad()
            pi_loss.backward()
            nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            # update alpha
            alpha_loss = torch.mean(
                (self.cal_entropy(pi_prob, pi_log_prob) - self.target_entropy).detach() * self.log_alpha.exp())

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().detach()

            self.training_step += 1

            logging_dict = {
                "train/training_step": self.training_step,
                'reward_loss/critic loss': q_loss.item(),
                "reward_loss/actor loss": pi_loss.item(),
                "reward_loss/alpha loss": alpha_loss.item(),
                "train/alpha": self.alpha.clone().item(),
                "train/predicted_q_value1": q1.mean().item(),
                "train/predicted_q_value2": q2.mean().item(),
            }
            wandb.log(logging_dict)
        for target_param, source_param in zip(self.target_critic_net.parameters(), self.critic_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.replace_tau) + source_param.data * self.replace_tau)

    def save(self, save_path, episode):
        base_path = os.path.join(save_path, 'trained_model')
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        model_actor_path = os.path.join(base_path, "actor_" + str(episode) + ".pth")
        torch.save(self.actor_net.state_dict(), model_actor_path)
        model_critic_path = os.path.join(base_path, "critic_" + str(episode) + ".pth")
        torch.save(self.critic_net.state_dict(), model_critic_path)
        param_alpha_path = os.path.join(base_path, "alpha_" + str(episode) + ".pth")
        torch.save(self.log_alpha, param_alpha_path)
        print("success saving model in {}".format(str(base_path)))

    def load(self, load_path, episode):
        print(f'\nBegin to load model: ')
        run_path = os.path.join(load_path, 'trained_model')
        model_actor_path = os.path.join(run_path, "actor_" + str(episode) + ".pth")
        model_critic_path = os.path.join(run_path, "critic_" + str(episode) + ".pth")
        param_alpha_path = os.path.join(run_path, "alpha_" + str(episode) + ".pth")
        print(f'Actor path: {model_actor_path}')
        print(f'Critic path: {model_critic_path}')
        print(f'param_alpha_path: {param_alpha_path}')

        if os.path.exists(model_actor_path):
            actor = torch.load(model_actor_path, map_location=self.learner_device)
            self.actor_net.load_state_dict(actor)
            print("Model loaded!")
        else:
            sys.exit(f'Model not founded!')

        if self.is_learner:
            self.target_critic_net.load_state_dict(self.critic_net.state_dict())
            self.log_alpha = torch.load(param_alpha_path, map_location=self.learner_device).requires_grad_()
            self.alpha = self.log_alpha.exp().detach()
            print("log_alpha loaded!")

    def load_actor(self, actor_model_path):
        print('\nBegin to load actor model from {}'.format(actor_model_path))
        if os.path.exists(actor_model_path):
            actor = torch.load(actor_model_path, map_location=self.learner_device)
            self.actor_net.load_state_dict(actor)
            print("Model loaded!")
        else:
            sys.exit(f'Model not founded!')

    def get_weights(self):
        return [move_state_dict(self.actor_net.state_dict(), "cpu")]

    def set_weights(self, weights, device):
        self.actor_net.load_state_dict(move_state_dict(weights[0], device))
