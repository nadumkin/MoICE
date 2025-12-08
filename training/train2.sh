#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train-balance.py \
    --model_name_or_path ../models/llama2-7b-chat \
    --data_path ../data/train_data.json \
    --eval_data_path ../data/eval_data.json \
    --output_dir ../checkpoints/moice-tinyllama \
    --overwrite_output_dir True \
    --save_safetensors False \
    --num_train_epochs 1 \
    --max_steps 50 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --logging_steps 1 \
    --weight_decay 0.0 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type constant \
    --topk 7 \
    --expert_nums 7 \
    --router_aux_loss_coef 0.3 \
    --base_set [10000,17500,18000,19000,20000,22500,25000] \
    --pretrain_loss True \
    --model_max_length 32768 \
    --source_model_max_length 4096 \
    --gradient_checkpointing True \
    --only_train_gate True \
    --lazy_preprocess True \
    --fp16 True \
    --report_to none

