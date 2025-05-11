import argparse
import copy
import hashlib
import os
import re
import sys
import json
from typing import Union
from multiprocessing.dummy import Pool as ThreadPool
import time

import requests
import tqdm
from pprint import pprint

sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")
from src.mara_generator import MARAGenerator
from src.reward_model import RewardModel
from src.general_logger import general_logger

log = general_logger(name="algorithm_logger")

parser = argparse.ArgumentParser()

# drl agent
parser.add_argument("--agent_model_path",
                    default="../train_result/multi_reward/mistral_v3_2_1/run2",
                    type=str,
                    help='Path to the drl policy model checkpoint')
parser.add_argument("--load_run", default=2, type=str, help="load the model trained for `load_run`")
parser.add_argument("--load_episode", default=18000, type=int, help="load the model trained for `load_episode` episode")
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
parser.add_argument("--reward_model_device", type=str, default="cuda:0", help="specify the gpu")

# text_generate control
parser.add_argument("--do_sample", action='store_true', help="use when eval_mode is raw")
parser.add_argument("--proxy_strategy", type=str, default="top1", choices=["top1", "random", "max_prob"],
                    help="use when eval_mode is proxy")
parser.add_argument('--max_new_token', type=int, default=2048)
parser.add_argument("--topk", default=40, type=int)
parser.add_argument("--topp", default=0.95, type=float)
parser.add_argument("--temperature", default=0.8, type=float)
parser.add_argument("--min_tokens_to_keep", default=1, type=int)
parser.add_argument("--length_penalty", default=1.0, type=float)
parser.add_argument("--N", default=10, type=int, help="candidate sample cnt of BON")
parser.add_argument("--keep_all_result", action='store_true', help="use when eval_mode is BON")
parser.add_argument('--reverse_choice', action='store_true')
parser.add_argument('--state_transition', default="v0", choices=["v0", "v1"])
parser.add_argument('--accept_action', type=int, default=1, choices=[0, 1])
parser.add_argument('--default_action_idx', type=int, default=0,
                    help="accept token of default_action_idx if all action is 0")
parser.add_argument("--serial_action", action='store_true', help="use when eval_mode is proxy")

# base eval setting
parser.add_argument("--eval_mode", type=str, default="proxy", choices=["raw", "proxy", "BON", "dpo", "rlhf", "aligner"])
parser.add_argument("--eval_action", default="generate", type=str,
                    choices=["generate", "reward_score"],
                    help="eval_action")
parser.add_argument("--eval_exp_name", default="", type=str, help="extra info to distinguish eval results")

# eval setting for eval_dataset
parser.add_argument("--eval_dataset_path", default="../data/SafeRLHF/test.json", type=str)
parser.add_argument("--eval_result_path", default="./SafeRLHF/eval_result", type=str)
parser.add_argument('--eval_from_start', action='store_true')
parser.add_argument('--eval_sample_cnt', type=int, default=-1, help="the number of eval data")

# eval setting for eval_action = reward_score
parser.add_argument("--score_dataset_path", default="./SafeRLHF/eval_result", type=str)

# eval setting for eval_mode is aligner aligner/aligner-7b-v1.0
parser.add_argument("--aligner_model_type", type=str, default="aligner_7b_v1",
                    help="use to get instruction prompt")
parser.add_argument("--aligner_model_path", default="path2model/aligner-7b-v1.0",
                    type=str, help='Path to the llm model')
parser.add_argument("--aligner_model_device", type=str, default="cuda:0", help="specify the gpu")


def get_model_output(args, eval_data):
    gen_info = {}
    begin_time = time.time()
    if not os.path.exists(args.eval_result_path):
        os.makedirs(args.eval_result_path)
    if args.eval_exp_name != "":
        eval_exp_name = "_" + args.eval_exp_name
    else:
        eval_exp_name = ""
    if args.eval_mode in ["raw", "dpo", "rlhf"]:
        generator_name = "{}_{}{}".format(args.policy_model_type.lower().replace("-", "_"),
                                          args.eval_mode, eval_exp_name)
    elif args.eval_mode == "aligner":
        generator_name = "{}_{}_{}{}".format(args.policy_model_type.lower().replace("-", "_"),
                                             args.eval_mode, args.aligner_model_type, eval_exp_name)
    elif args.eval_mode == "BON":
        generator_name = "{}_{}_{}_{}{}".format(args.policy_model_type.lower().replace("-", "_"),
                                                args.eval_mode, args.reward_model_type, args.N, eval_exp_name)
    else:
        generator_name = "{}_{}_run_{}_episode_{}_{}_topk{}_topp{}_temperature{}{}".format(
            args.policy_model_type.lower().replace("-", "_"),
            args.eval_mode, args.load_run, args.load_episode, args.proxy_strategy, args.topk, args.topp,
            args.temperature,
            eval_exp_name)
        gen_info = {"gen_token_cnt_list": [],  # 每个样例生成的结果长度，len(gen_token_cnt_list)=sample_cnt
                    "proxy_token_cnt_list": [],  # 需要经过agent进行决策的token长度，len(proxy_token_cnt_listt)=sample_cnt
                    "cand_token_dict": {},  # 每个决策位的候选token数，1<=cand_token_cnt<=topK，0表示不需要决策
                    "accept_index_dict": {},  # 每个决策位的选择第几个token， accept_idx
                    "eval_cnt": len(eval_data),
                    "topk": args.topk,
                    "topp": args.topp,
                    "temperature": args.temperature
                    }
    target_path = os.path.join(args.eval_result_path, "{}_output.json".format(generator_name))
    log.info("target_path:{}".format(target_path))
    if os.path.exists(target_path) and not args.eval_from_start:
        with open(target_path, 'r', encoding='utf-8') as file:
            target_datas = json.load(file)
        log.info("there are {} completed_data".format(len(target_datas)))
        completed_cnt = len(target_datas)
        if completed_cnt >= len(eval_data):
            log.info("get_model_output finish!")
        uncompleted_data = eval_data[len(target_datas):]
    else:
        target_datas = []
        uncompleted_data = eval_data
    log.info("there are {} uncompleted_data".format(len(uncompleted_data)))
    llm_generator = MARAGenerator(args)
    if args.eval_mode in ["raw", "dpo", "rlhf"]:
        generator = llm_generator.get_raw_output
    elif args.eval_mode == "aligner":
        generator = llm_generator.get_aligner_output
    elif args.eval_mode == "BON":
        generator = llm_generator.get_bon_output
    else:
        generator = llm_generator.get_proxy_output_serial
    for data in tqdm.tqdm(uncompleted_data):
        output, sub_gen_info = generator(data["instruction"])
        # sub_gen_info = {"gen_token_cnt": int,  # 每个样例生成的结果长度，len(gen_token_cnt_list)=sample_cnt
        #             "proxy_token_cnt": int,  # 需要经过agent进行决策的token长度，len(proxy_token_cnt_listt)=sample_cnt
        #             "cand_token_dict": {},  # 每个决策位的候选token数，1<=cand_token_cnt<=topK，0表示不需要决策
        #             "accept_index_dict": {}  # 每个决策位的选择第几个token， accept_idx
        #             }
        log.info("*****input****")
        log.info(data["instruction"])
        log.info("*****output****")
        log.info(output)
        data["output"] = output
        data["generator"] = generator_name
        if args.eval_mode == "aligner":
            data["raw_output"] = sub_gen_info["raw_output"]
        target_datas.append(copy.deepcopy(data))
        if args.eval_mode == "proxy":
            gen_info["gen_token_cnt_list"].append(sub_gen_info["gen_token_cnt"])
            gen_info["proxy_token_cnt_list"].append(sub_gen_info["proxy_token_cnt"])
            for k, v in sub_gen_info["cand_token_dict"].items():
                if k not in gen_info["cand_token_dict"]:
                    gen_info["cand_token_dict"][k] = v
                else:
                    gen_info["cand_token_dict"][k] += v
            for k, v in sub_gen_info["accept_index_dict"].items():
                if k not in gen_info["accept_index_dict"]:
                    gen_info["accept_index_dict"][k] = v
                else:
                    gen_info["accept_index_dict"][k] += v
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(target_datas, f, indent=4, ensure_ascii=False)
        log.info("finish to save {} sample to {}".format(len(target_datas), target_path))
    return generator_name, time.time() - begin_time, gen_info


def get_reward_score(args):
    source_path = args.score_dataset_path
    if os.path.exists(source_path):
        with open(source_path, 'r', encoding='utf-8') as file:
            target_datas = json.load(file)
        if args.eval_sample_cnt != -1:
            target_datas = target_datas[:args.eval_sample_cnt]
        log.info("there are {} data to be scored. ".format(len(target_datas)))
    else:
        log.info("target_path not exist!")
        return
    target_path = source_path.replace("output.json", "{}_score.json".format(args.reward_model_type))
    log.info("target_path: {}".format(target_path))
    reward_model = RewardModel(reward_model_type=args.reward_model_type,
                               reward_model_path=args.reward_model_path,
                               reward_device=args.reward_model_device)
    for data in tqdm.tqdm(target_datas):
        log.info(data["output"])
        if isinstance(data["output"], str):
            score = reward_model.get_score([data["instruction"]], [data["output"]])[0]
            log.info(score)
            data["reward_score"] = score
        else:  # list(dict) {"output": candidate_result_list[i], "reward_score": reward_score_list[i]}
            for item in data["output"]:
                score = reward_model.get_score([data["instruction"]], [item["output"]])[0]
                log.info(score)
                item["reward_score"] = score
        data["reward_model"] = args.reward_model_type
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(target_datas, f, indent=4, ensure_ascii=False)
        log.info("finish to save {} sample to {}".format(len(target_datas), target_path))


def record_time(eval_data_name, generator_name, use_time):
    # eval_data_name, generator_name,
    log.info("eval_data_name:{}, generator_name:{}, use_time:{}".format(eval_data_name, generator_name, use_time))
    target_path = "record_time.json"
    if os.path.exists(target_path):
        with open(target_path, 'r', encoding='utf-8') as file:
            time_dict = json.load(file)
    else:
        time_dict = {}
    if eval_data_name not in time_dict:
        time_dict[eval_data_name] = {}
    time_dict[eval_data_name][generator_name] = use_time
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(time_dict, f, indent=4, ensure_ascii=False)
    log.info("success to update new time_dict to {}.".format(target_path))


def record_gen_info(eval_data_name, generator_name, gen_info):
    log.info("eval_data_name:{}, generator_name:{}, use_time:{}".format(eval_data_name, generator_name, gen_info))
    target_path = "gen_info.json"
    if os.path.exists(target_path):
        with open(target_path, 'r', encoding='utf-8') as file:
            gen_info_dict = json.load(file)
    else:
        gen_info_dict = {}
    if eval_data_name not in gen_info_dict:
        gen_info_dict[eval_data_name] = {}
    gen_info_dict[eval_data_name][generator_name] = gen_info
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(gen_info_dict, f, indent=4, ensure_ascii=False)
    log.info("success to update new time_dict to {}.".format(target_path))


if __name__ == '__main__':
    args = parser.parse_args()
    if args.eval_action == "generate":
        if not os.path.exists(args.eval_result_path):
            os.makedirs(args.eval_result_path)
        eval_data_name = args.eval_dataset_path.split("/")[-2]
        print(eval_data_name)
        with open(args.eval_dataset_path, "r") as f:
            eval_data = json.load(f)
        log.info("success get {} eval data".format(len(eval_data)))
        if args.eval_sample_cnt != -1:
            eval_data = eval_data[:args.eval_sample_cnt]
        log.info("there are {} data needed to get model output.".format(len(eval_data)))
        log.info(eval_data[0])
        generator_name, use_time, gen_info = get_model_output(args, eval_data)
        record_time(eval_data_name, generator_name, use_time)
        if args.eval_mode == "proxy":
            record_gen_info(eval_data_name, generator_name, gen_info)
    if args.eval_action == "reward_score":
        get_reward_score(args)
