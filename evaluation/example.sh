CUDA_VISIBLE_DEVICES=0 python eval_executor.py \
--eval_dataset_path ../data/SafeRLHF/test.json \
--eval_result_path ./SafeRLHF/eval_result \
--eval_action generate \
--eval_mode proxy \
--serial_action \
--state_dim 4096 \
--eval_from_start \
--eval_sample_cnt 200 \
--agent_model_path ../train_result/trained_model/mistral_v3_2_1_actor.pth \
--policy_model_type  mistral_instruct_v3 \
--policy_model_path path2model/Mistral-7B-Instruct-v0.3  \
--policy_model_device cuda:0 \
--state_transition v0 \
--default_action_idx 0 \
--proxy_strategy top1 \
--topk 40 \
--topp 0.95 \
--temperature 0.8 \
--max_new_token 2048

CUDA_VISIBLE_DEVICES=0 python eval_executor.py \
  --score_dataset_path ./SafeRLHF/eval_result/mistral_v3_2_1_actor_top1_topk40_topp0.95_temperature0.8_output.json \
  --eval_action reward_score \
  --reward_model_type  beaver_reward \
  --reward_model_path path2model/beaver-7b-v1.0-reward  \
  --reward_model_device cuda:0

CUDA_VISIBLE_DEVICES=0 python eval_executor.py \
  --score_dataset_path ./SafeRLHF/eval_result/mistral_v3_2_1_actor_top1_topk40_topp0.95_temperature0.8_output.json \
  --eval_action reward_score \
  --reward_model_type  beaver_cost \
  --reward_model_path path2model/beaver-7b-v1.0-cost  \
  --reward_model_device cuda:0