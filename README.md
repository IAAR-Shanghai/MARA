<h1 align="center">
  <img src="./assets/icons.png" alt="MARA Icon" width="45"  height="45"/> Token-level Accept or Reject: A micro alignment approach for Large Language Models
</h1>


<div style="display: flex; justify-content: center; gap: 10px;">
<p align="center">
  <a href="https://github.com/IAAR-Shanghai/MARA">
    <img src="https://img.shields.io/badge/GitHub-Repository-blue?logo=github" alt="GitHub"/>
  </a>
  <a href="https://huggingface.co/IAAR-Shanghai/MARA_AGENTS">
    <img src="https://img.shields.io/badge/🤗%20Hugging%20Face-MARA_AGENTS-yellow" alt="Hugging Face"/>
  </a>
    <a href="https://arxiv.org/abs/2505.19743">
        <img src="https://img.shields.io/badge/arXiv-Paper-8B0000?style=flat-square&logo=arxiv&logoColor=white">
    </a>
</p>
</div>

This is the official code of the paper [**Token-level Accept or Reject: A micro alignment approach for Large Language
Models** [IJCAI 2025]](http://arxiv.org/abs/2505.19743) .



## Overview
TL;DR: **MARA** (Micro token-level Accept-Reject Alignment) simplifies the alignment process by breaking down sentence-level preference learning into fine-grained token-level binary classification. The MARA agent—a lightweight multi-layer perceptron (MLP)—operates as an alignment model that evaluates and classifies each candidate token as either *Accepted* or *Rejected* during LLM text generation.

<figure>
  <img src="./assets/mara_architecture.png" alt="mara_architecture" style="display: block; margin: 0 auto;" />
  <figcaption style="text-align: center;">Architecture of MARA: The alignment model performs token selection through accept-reject decisions.</figcaption>
</figure>

## Dependencies

The required dependencies and their versions can be found in the [`requirements.txt`](requirements.txt). The main
packages are `pytorch`, `transformers` and `ray`.

To install all the required packages along with their dependencies, run

```sh
pip install -r requirements.txt
```

## Quickstart

### Training

Use the following command to run MARA training of
the [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) model with preference rate
between reward model ([beaver-7b-v1.0-reward](https://huggingface.co/PKU-Alignment/beaver-7b-v1.0-reward)) and cost
model ([beaver-7b-v1.0-cost](https://huggingface.co/PKU-Alignment/beaver-7b-v1.0-cost)) as 2:1. See [`scripts`](scripts)
for more training commands.

```bash
bash scripts/train_multi_reward_mistral_v3.sh
```

### Evaluation

We offer the trained RL actor model based on Mistral-7B-Instruct-v0.3 with preference rate 2:1. Use the following
command to get the alignment result and the reward score and cost score of the model output. More trained MARA agents
can be found at [MARA_AGENTS](https://huggingface.co/IAAR-Shanghai/MARA_AGENTS).

```bash
cd evaluation
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

```

## Results for Experiment
<table class="center">
    <tr>
        <td width=100% style="border: none">
        <img src="assets/table1.png" style="width:50%; max-width:100%;">
        <div style="text-align: left; margin-top: 8px;">Performance improvements of MARA across PKUSafeRLHF, BeaverTails, and HarmfulQA datasets. Each entry shows the percentage improvement in preference rate achieved by applying MARA compared to using the original LLM alone.</div>
        </td>
    </tr>
</table>
<table class="center">
    <tr>
        <td width=100% style="border: none">
        <img src="assets/table3.png" style="width:50%; max-width:100%;">
        <div style="text-align: left; margin-top: 8px;">Compatibility analysis of MARA, an alignment model trained with a LLM to be aggregate with other inference LLM. The value of each cell represents the percentage improvement in preference rate of our algorithm over the upstream model, i.e., inference model.</div>
        </td>
    </tr>
</table>

<table class="center">
    <tr>
        <td width=100% style="border: none">
            <img src="assets/table2.png" style="width:100%">
            <div style="text-align: left; margin-top: 8px;">Performance comparison of MARA against RLHF, DPO, and Aligner measured by percentage improvements of preference rate.</div>
        </td>
    </tr>
</table>

More details and analyses about experimental results can be found in our [paper](https://arxiv.org/abs/2505.19743).

## 📖 BibTeX
If the code or the paper has been useful in your research, please add a citation to our work:
```
@article{zhang2025tokenlevelacceptrejectmicro,
      title={Token-level Accept or Reject: A Micro Alignment Approach for Large Language Models}, 
      author={Yang Zhang and Yu Yu and Bo Tang and Yu Zhu and Chuxiong Sun and Wenqiang Wei and Jie Hu and Zipeng Xie and Zhiyu Li and Feiyu Xiong and Edward Chung},
      journal={arXiv preprint arXiv:2505.19743},
      year={2025}
}
```