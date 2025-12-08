#rope moe train OpenHermes balance
train_script="train-balance.py \
    --model_name_or_path ../models/llama2-7b-chat \
    --data_path  ../data/train_data.json \
    --eval_data_path  ../data/eval_data.json \
    --bf16 False \
    --output_dir ../checkpoints/moice-llama2 \
    --overwrite_output_dir True \
    --save_safetensors False \
    --router_aux_loss_coef 0.3 \
    --num_train_epochs 10 \
    --max_steps 7 \
    --pretrain_loss True \
    --topk 7 \
    --expert_nums 7 \
    --deepspeed ds_z3_bf16.json \
    --base_set [10000,17500,18000,19000,20000,22500,25000] \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy no \
    --only_train_gate True \
    --save_strategy epoch \
    --save_total_limit 10 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.2 \
    --logging_steps 1 \
    --lr_scheduler_type constant \
    --source_model_max_length 4096 \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to tensorboard "

deepspeed --num_gpus=4 ${train_script}
