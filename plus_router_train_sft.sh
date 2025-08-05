cd /141nfs/username/paper_register
source /141nfs/username/anaconda3/bin/activate paper

log_dir=logs_router_train
mkdir -p ${log_dir}

model_name="Qwen3-0.6B"
per_device_train_batch_size=40
gradient_accumulation_steps=1

model_path=/141nfs/username/hf_models/${model_name}

tag=SFT-AUG
output_dir=models/plus_${model_name}_${tag}
log_path=${log_dir}/plus_${model_name}_${tag}.log
train_data_path=data_train/plus_datas_train.jsonl_aug.jsonl

config_file=utils/accelerate_configs/deepspeed_zero3.yaml
nohup accelerate launch --config_file ${config_file} --main_process_port 25154 \
    plus_router_train_sft.py \
        --model_path $model_path \
        --output_dir $output_dir \
        --train_data_path $train_data_path \
        --per_device_train_batch_size $per_device_train_batch_size \
        --gradient_accumulation_steps $gradient_accumulation_steps \
    > ${log_path} &