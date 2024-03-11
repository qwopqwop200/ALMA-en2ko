OUTPUT_DIR=${1:-"./alma-10.7b-ft-lora"}
pairs=${2:-"ko-en,en-ko"}
LORA_RANK=${3:-"64"}

export HF_DATASETS_CACHE=".cache/huggingface_cache/datasets"
export TRANSFORMERS_CACHE=".cache/models/"

# random port between 30000 and 50000
port=$(( RANDOM % (50000 - 30000 + 1 ) + 30000 ))

accelerate launch --main_process_port ${port} --config_file configs/deepspeed_train_config_bf16.yaml \
    run_llmmt.py \
    --model_name_or_path yanolja/EEVE-Korean-10.8B-v1.0 \
    --mmt_data_path  ./human_written_data/ \
    --use_peft \
    --lora_rank ${LORA_RANK} \
    --do_train \
    --do_eval \
    --do_predict \
    --language_pairs ${pairs} \
    --load_best_model_at_end \
    --low_cpu_mem_usage \
    --bf16 \
    --learning_rate 2e-3 \
    --weight_decay 0.01 \
	--gradient_checkpointing true \
    --gradient_accumulation_steps 11 \
    --lr_scheduler_type inverse_sqrt \
    --warmup_ratio 0.01 \
    --ignore_pad_token_for_loss \
    --ignore_prompt_token_for_loss \
    --per_device_train_batch_size 3 \
    --per_device_eval_batch_size 3 \
    --evaluation_strategy steps \
    --eval_steps 379 \
    --save_strategy steps \
    --save_steps 379 \
    --save_total_limit 1 \
    --logging_strategy steps \
    --logging_steps 0.05 \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 3 \
    --predict_with_generate \
    --prediction_loss_only \
    --max_new_tokens 2048 \
    --max_source_length 2048 \
    --seed 42 \
    --overwrite_output_dir \
    --num_beams 5 \
    --ddp_timeout 999999 \
    --report_to none \
    --overwrite_cache
    
## Evaluation (BLEU, COMET)
#bash ./evals/eval_generation.sh ${OUTPUT_DIR} ${pairs}