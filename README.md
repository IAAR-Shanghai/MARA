<h1 align="center">
    Token-level Accept or Reject: A micro alignment approach for Large Language Models
</h1>

This is the official code of the paper **Token-level Accept or Reject: A micro alignment approach for Large Language Models**
## Overview
TL;DR: **MARA** 
(Micro token-level Accept-Reject Aligning)  simplifies the alignment process by decomposing sentence-level preference learning into token-level binary classification, where a compact three-layer fully-connected network  determines whether candidate tokens are “Accepted” or “Rejected” as part of the response.

## Dependencies
The required dependencies and their versions can be found in the [`requirements.txt`](requirements.txt). The main packages are `pytorch`, `transformers` and `ray`.

To install all the required packages along with their dependencies, run
```sh
pip install -r requirements.txt
```

### Quickstart

#### Training

Use the following command to run MARA training of the Mistral-7B-Instruct-v0.3 model with preference rate between reward model (beaver-7b-v1.0-reward) and cost model (beaver-7b-v1.0-cost) as 2:1 .

```bash
wandb login --relogin xxxx
cd src
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
    --env_name multi_reward \
    --algo_name mistral_v3 \
    --dataset safeRLHF \
    --dataset_path ../data/SafeRLHF/train.json \
    --train_epoch 1 \
    --max_episode 21000 \
    --state_dim 4096 \
    --action_dim 2 \
    --hidden_dim 1024 \
    --buffer_capacity 1000000 \
    --batch_size 1024 \
    --lr_actor 0.0003 \
    --lr_critic 0.0003 \
    --alpha 0.01 \
    --lr_alpha 0.0003 \
    --target_entropy_factor 0.6 \
    --target_entropy_type log \
    --replace_tau 0.005 \
    --update_time 10 \
    --save_interval 500 \
    --policy_model_type  mistral_instruct_v3 \
    --policy_model_path path2model/Mistral-7B-Instruct-v0.3  \
    --reward_model_type  beaver_reward beaver_cost \
    --reward_model_path path2model/beaver-7b-v1.0-reward path2model/beaver-7b-v1.0-cost  \
    --reward_multiplier 2.0 -1.0\
    --reward_model_device cuda:7 \
    --reward_type kl_div \
    --cond_prob \
    --reward_baseline 0.0 \
    --postback_reward 1 \
    --kl_ctl 0.1 \
    --kl_penalty kl \
    --num_gpus 8 \
    --num_agents 7 \
    --learner_device cuda:7 \
    --max_new_token 512 \
    --topk 40 \
    --topp 0.95 \
    --temperature 0.8 \
    --length_penalty 1 \
    --wandb_entity_name xxx \
    --wandb_project_name xxx

```

#### Evaluation

We offer the trained RL model of the Mistral-7B-Instruct-v0.3 model with preference rate 2:1. Use the following command to get the alignment result and the reward score and cost score of the model output.

```bash
cd evaluation
CUDA_VISIBLE_DEVICES=0 python eval_executor.py \
--eval_dataset_path ./SafeRLHF/eval_data/test.json \
--eval_result_path ./SafeRLHF/eval_result \
--eval_action generate \
--eval_mode proxy \
--serial_action \
--state_dim 4096 \
--eval_from_start \
--eval_sample_cnt 200 \
--agent_model_path ../train_result/multi_reward/mistral_v3_2_1/run2 \
--load_episode 18000 \
--load_run 2 \
--policy_model_type  mistral_instruct_v3 \
--policy_model_path path2model/Mistral-7B-Instruct-v0.3  \x
--policy_model_device cuda:0 \
--state_transition v0 \
--default_action_idx 0 \
--proxy_strategy top1 \
--topk 40 \
--topp 0.95 \
--temperature 0.8 \
--max_new_token 2048


CUDA_VISIBLE_DEVICES=0 python eval_executor.py \
  --score_dataset_path ./SafeRLHF/eval_result/mistral_instruct_v3_proxy_run_2_episode_18000_top1_topk40_topp0.95_temperature0.8_output.json \
  --eval_action reward_score \
  --reward_model_type  beaver_reward \
  --reward_model_path path2model/beaver-7b-v1.0-reward  \
  --reward_model_device cuda:0

CUDA_VISIBLE_DEVICES=0 python eval_executor.py \
  --score_dataset_path ./SafeRLHF/eval_result/mistral_instruct_v3_proxy_run_2_episode_18000_top1_topk40_topp0.95_temperature0.8_output.json \
  --eval_action reward_score \
  --reward_model_type  beaver_cost \
  --reward_model_path path2model/beaver-7b-v1.0-cost  \
  --reward_model_device cuda:0

```

All of the pre-trained alignment models, detail evaluation result and detailed experimental logs will be released when our work  is officially published. Our open-source commitment ensures that all results presented in this paper can be independently verified and reproduced with minimal effort, promoting transparency and facilitating future research in the field.