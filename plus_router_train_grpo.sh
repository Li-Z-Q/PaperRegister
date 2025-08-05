CUDA_VISIBLE_DEVICES=0,1

cd /141nfs/username/paper_register
source /141nfs/username/anaconda3/bin/activate paper

log_dir=logs_router_train
mkdir -p ${log_dir}


temperature=2.0

reward_method=Tree

model_size=0.6B
model_path=models/plus_Qwen3-${model_size}_SFT-AUG
train_tag=SFT-AUG-GRPO-${reward_method}_${temperature}

per_device_train_batch_size=200
gradient_accumulation_steps=2

output_dir=models/plus_Qwen3-${model_size}_${train_tag}
log_path=logs_router_train/plus_Qwen3-${model_size}_${train_tag}.log 
train_data_path=data_train/plus_datas_train.jsonl_aug.jsonl_grpo.jsonl

config_file=utils/accelerate_configs/deepspeed_zero3.yaml

echo "model_path ${model_path}"
echo "output_dir ${output_dir}"
echo "temperature ${temperature}"
echo "reward_method ${reward_method}"
echo "train_data_path ${train_data_path}"

nohup accelerate launch --config_file ${config_file} --main_process_port 25151 \
    plus_router_train_grpo.py \
        --model_path $model_path \
        --output_dir $output_dir \
        --temperature $temperature \
        --reward_method $reward_method \
        --train_data_path $train_data_path \
        --per_device_train_batch_size $per_device_train_batch_size \
        --gradient_accumulation_steps $gradient_accumulation_steps \
    > ${log_path} &
