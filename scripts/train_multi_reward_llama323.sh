cd src
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
    --env_name multi_reward \
    --algo_name llama323_safeRLHF \
    --dataset safeRLHF \
    --dataset_path ../data/SafeRLHF/train.json \
    --train_epoch 1 \
    --max_episode 21000 \
    --state_dim 3072 \
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
    --policy_model_type  llama \
    --policy_model_path path2model/Llama-3.2-3B-Instruct  \
    --reward_model_type  beaver_reward beaver_cost \
    --reward_model_path path2model/beaver-7b-v1.0-reward path2model/beaver-7b-v1.0-cost  \
    --reward_multiplier 1.0 -1.0 \
    --reward_model_device cuda:7 cuda:7 \
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
    --temperature 0.8