CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
MAX_PIXELS=401408 \
swift rlhf \
    --rlhf_type grpo \
    --model_type qwen2_5_vl \
    --model <PATH_TO_BASE_MODEL> \
    --external_plugins \
        <PATH_TO_PLUGIN_DIR>/plugin_rm_v2_async.py \
        <PATH_TO_PLUGIN_DIR>/plugin_joint_if_spatial_training_with_question_known.py \
        <PATH_TO_PLUGIN_DIR>/plugin_vpt.py \
        <PATH_TO_PLUGIN_DIR>/plugin_joint_if_spatial_planningformat_training.py \
    --reward_funcs external_sr_ORM external_wothink_format external_planning_rm_ORM external_planning_rule_ORM visual_perception_accuracy \
    --reward_weights 0.95 0.05 0.8 0.2 1.0 \
    --torch_dtype bfloat16 \
    --dataset \
        <PATH_TO_DATASET_DIR>/embspatial_all_right_partial_right_all_wrong_but_consistent_with_question_known.jsonl \
        <PATH_TO_DATASET_DIR>/selfrevision_formatted_rl_26k_with_system_prompt.jsonl \
        <PATH_TO_DATASET_DIR>/Alfred_data_mid_v4_with_system_prompt.jsonl \
        <PATH_TO_DATASET_DIR>/swift_VPT_grpo_except_ocr_with_tasktype.jsonl \
    --max_completion_length 2048 \
    --num_train_epochs 1 \
    --train_type full \
    --per_device_train_batch_size 4 \
    --learning_rate 2.5e-6 \
    --gradient_accumulation_steps 1 \
    --save_steps 100 \
    --save_only_model true \
    --eval_strategy no \
    --save_total_limit 500 \
    --logging_steps 1 \
    --max_length 4096 \
    --output_dir <PATH_TO_OUTPUT_DIR> \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 16 \
    --temperature 1.2 \
    --deepspeed zero1 \
    --log_completions true \
    --vllm_max_model_len 1024 \
    --num_iterations 1 \
    --async_generate false \
    --beta 0.001 \
    --loss_type bnpo \
    --epsilon_high 0.28 \
    --dynamic_sample true \
    --overlong_filter true \
    --max_resample_times 3 \
    --top_p 0.98 \
    --top_k 160