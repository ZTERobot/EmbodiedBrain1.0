PYTHONWARNINGS="ignore"
nnodes=2
nproc_per_node=8

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export FPS_MAX_FRAMES=128
export MAX_PIXELS=262144
export VIDEO_MAX_PIXELS=16384

NNODES=$nnodes \
NODE_RANK=0 \
MASTER_ADDR=<MASTER_IP> \
MASTER_PORT=29500 \
NPROC_PER_NODE=$nproc_per_node \
swift sft \
    --model_type qwen2_5_vl \
    --model <PATH_TO_BASE_MODEL> \
    --train_type full \
    --max_length 8192 \
    --freeze_vit True \
    --freeze_aligner True \
    --freeze_llm False \
    --truncation_strategy left \
    --output_dir <PATH_TO_OUTPUT_DIR> \
    --torch_dtype bfloat16 \
    --dataset \
        <PATH_TO_DATASET_DIR>/dataset_1.jsonl#7500 \
        <PATH_TO_DATASET_DIR>/dataset_2.jsonl#7500 \
        <PATH_TO_DATASET_DIR>/dataset_3.jsonl#5000 \
    --deepspeed zero2 \
    --per_device_train_batch_size 1 \
    --num_train_epochs 3 \
    --learning_rate 1.0e-6 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type 'cosine' \
    --warmup_ratio 0.1 \
    --save_steps 7935 \
    --save_total_limit 6 \
    --logging_steps 5 \
    --attn_impl flash_attn \
    --report_to tensorboard \
    --logging_dir <PATH_TO_LOG_DIR> \
    --gradient_checkpointing true \
    --split_dataset_ratio 0.0