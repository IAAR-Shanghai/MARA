import argparse
import os
import sys
import time
import json
from pathlib import Path
import wandb

from utils import make_save_path, seed_everything
from distri_executor import Learner, Worker, ParameterServer
import ray
import logging

log = logging.getLogger("ray")

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
# os.environ['WANDB_MODE'] = 'offline'
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['RAY_DEDUP_LOGS'] = '0'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

base_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(base_dir)

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', default="multi_reward", type=str)
parser.add_argument('--algo_name', default="mistralv3_saferlhf", type=str)
parser.add_argument('--train_epoch', default=1, type=int)
parser.add_argument('--max_episode', default=21000, type=int)

# drl agent
parser.add_argument("--continue_train", action='store_true')
parser.add_argument("--reload_path", default="", type=str, help="path for reload the trained model")
parser.add_argument("--reload_run", default=1, type=int, help="load the model of run(load_run)")
parser.add_argument("--reload_episode", default=20, type=int,
                    help="reload the model trained for `load_episode` episode")
parser.add_argument('--state_dim', type=int, default=4096)
parser.add_argument('--action_dim', type=int, default=2)
parser.add_argument('--hidden_dim', type=int, default=1024)
parser.add_argument('--buffer_capacity', default=int(1e6), type=int)
parser.add_argument('--batch_size', default=1024, type=int)
parser.add_argument('--lr_actor', default=0.003, type=float)
parser.add_argument('--lr_critic', default=0.003, type=float)
parser.add_argument('--lr_alpha', default=0.003, type=float)
parser.add_argument('--alpha', default=0.01, type=float)
parser.add_argument('--max_grad_norm', default=0.5, type=float)
parser.add_argument('--replace_tau', default=0.005, type=float)
parser.add_argument('--gamma', default=0.99, type=float)
parser.add_argument("--update_time", default=10, type=int, help='The number of repeated updates on a generated batch.')
parser.add_argument('--random_seed', default=2024, type=int)
parser.add_argument("--save_interval", default=500, type=int)

# sac entropy
parser.add_argument('--target_entropy_factor', type=float, default=0.6)
parser.add_argument('--target_entropy_type', type=str, default='log', choices=['simple', 'log'])
parser.add_argument("--accumulated_step", default=1, type=int,
                    help="collect accumulated_step step while exploring, then update networks")

# upstream llm
parser.add_argument("--policy_model_type", type=str, default="mistral_instruct_v3",
                    choices=["llama", "mistral_instruct_v1", "mistral_instruct_v2", "mistral_instruct_v3"])
parser.add_argument("--policy_model_path", default="Path2Model/Mistral-7B-Instruct-v0.3", type=str,
                    help='upstream model path')
parser.add_argument("--policy_model_device", type=str, default="cuda:0", help="device to load policy model")

# reward_model
parser.add_argument("--reward_model_type", nargs='+', type=str, default=["beaver_reward", "beaver_cost"])
parser.add_argument("--reward_model_path", nargs='+', type=str,
                    default=["Path2Model/beaver-7b-v1.0-reward",
                             "Path2Model/beaver-7b-v1.0-cost"],
                    help='reward model path')
parser.add_argument("--reward_model_device", nargs='+', type=str, default=["cuda:7", "cuda:7"],
                    help="device to load reward model")
# final reward
parser.add_argument("--reward_multiplier", nargs='+', type=float, default=[1.0, -1.0],
                    help="reward_multiplier of reward model score")
parser.add_argument("--sft_as_reward_baseline", default=False, action='store_true',
                    help="use sft result score as reward_baseline")
parser.add_argument("--baseline_multiplier", default=1.0, type=float,
                    help="multiplier of init_reward_score-reward_baseline")
parser.add_argument("--reward_baseline", type=float, default=-15.0,
                    help="final reward baseline,  final_reward=baseline_multiplier*(init_reward_score-reward_baseline), set reward_score within a reasonable range. ")
parser.add_argument("--postback_reward", type=int, default=0,
                    help="postback the final reward to all token or not")
# instant reward
parser.add_argument("--reward_type", type=str, default="kl_div",
                    help="How instant rewards are calculated， now can choice from (sum_logit, logit, kl_div)")
parser.add_argument("--cond_prob", default=False, action='store_true',
                    help="use contingent probability to cal kl or not")
parser.add_argument("--kl_ctl", type=float, default=0.1, help="reward = - kl_ctl * kl_div")
parser.add_argument("--kl_penalty", type=str, default="full", help="can choice from (kl, abs, mse, full)")

# multi gpu
parser.add_argument("--num_gpus", type=int, default=8, help="the number of use gpu")
parser.add_argument("--num_agents", type=int, default=4, help="the number of agent that interacts with the llm_env and collects sample.")
parser.add_argument("--share_device", default=False, action='store_true')
parser.add_argument("--learner_device", default="cuda:0", type=str)

# dataset
parser.add_argument('--repeat_cnt', type=int, default=1, help="repeat generate the same sample for explore")
parser.add_argument('--dataset', type=str, default='SafeRLHF')
parser.add_argument("--dataset_path", default="../data/SafeRLHF/train.json", type=str)

# text_generate control
parser.add_argument('--state_transition', type=str, default="v0")
parser.add_argument('--reverse_choice', action='store_true')
parser.add_argument('--max_new_token', type=int, default=512)
parser.add_argument("--topk", default=40, type=int)
parser.add_argument("--topp", default=0.95, type=float)
parser.add_argument("--temperature", default=0.8, type=float)
parser.add_argument("--min_tokens_to_keep", default=1, type=int)
parser.add_argument("--length_penalty", default=1.0, type=float)

# wandb init in proxy_rlhf.scripts.distri_executor_v2 Learner() __init__
parser.add_argument('--wandb_entity_name', type=str, default="xx")
parser.add_argument('--wandb_project_name', type=str, default="xx")



# ====================================  EXECUTE FUNCTION  ===============================

def distri_train(args, train_data):
    begin_time = time.time()

    log.info("env_name:{}".format(args.env_name))
    seed_everything(args.random_seed)

    if args.continue_train:
        # continue train
        save_path = args.reload_path
        run_index = "continue_run{}".format(args.reload_run)
    else:
        # define save path
        save_path, run_index = make_save_path(args.env_name, args.algo_name)
        log.info("run_dir:{}".format(save_path))
    # save train params
    params_file = os.path.join(save_path, 'training_params.json')
    with open(params_file, 'w') as f:
        json.dump(vars(args), f)

    args.save_path = save_path
    args.run_index = run_index
    learner = Learner(all_args=args)
    ray.init(num_cpus=8, num_gpus=args.num_gpus)

    try:
        workers = []
        for i in range(args.num_agents):
            workers.append(Worker.remote(train_data, worker_id=i, all_args=args))

        ps = ParameterServer(learner.get_weights())

        episode_ids = []
        for i, worker in enumerate(workers):
            episode_ids.append(worker.run.remote(ps))
            time.sleep(1)

        for _ in range(1, args.max_episode):
            ready, not_ready = ray.wait(episode_ids)
            current_buffer, llm_io, logging_dict, worker_id = ray.get(ready)[0]

            episode_ids = not_ready
            episode_ids.append(workers[worker_id].run.remote(ps))

            # gc.collect()

            learner.update(ps, current_buffer, llm_io, logging_dict)
    except KeyboardInterrupt:
        log.info('\nTraining has been Shutdown \n')
    finally:

        ray.shutdown()
        end_time = time.time()
        log.info(f"Training costs {(end_time - begin_time):.2f} s")
        wandb.finish()


if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.dataset_path, "r") as f:
        train_data = json.load(f)
    for data in train_data:
        if 'prompt' not in data:
            data['prompt'] = data["instruction"]
    distri_train(args, train_data)
