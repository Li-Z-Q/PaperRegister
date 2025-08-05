cd /141nfs/username/paper_register
source /141nfs/username/anaconda3/bin/activate paper


return_seq_num=5




# train_tag=SFT-AUG-GRPO-Tree_2.0


model_name="Qwen3-0.6B"
worker_num=50


data_split=test

model_path=models/plus_${model_name}_${train_tag}
prediction_save_dir=result/plus_${data_split}_${model_name}_${train_tag}_${return_seq_num}
log_dir_path=logs_router_inference/plus_${data_split}_${model_name}_${train_tag}_${return_seq_num}

mkdir -p ${log_dir_path}
echo "starting ${model_name} ${train_tag} ${data_split}"

export CUDA_VISIBLE_DEVICES=0
for ((id=0; id<worker_num/2; id++))
do
    nohup python -u plus_router_inference.py \
        --model_path ${model_path} \
        --data_split ${data_split} \
        --prediction_save_dir ${prediction_save_dir} \
        --worker_num ${worker_num} \
        --worker_id ${id} \
        --return_seq_num ${return_seq_num} \
        > ${log_dir_path}/${id}.log &
done

export CUDA_VISIBLE_DEVICES=1
for ((id=worker_num/2; id<worker_num; id++))
do
    nohup python -u plus_router_inference.py \
        --model_path ${model_path} \
        --data_split ${data_split} \
        --prediction_save_dir ${prediction_save_dir} \
        --worker_num ${worker_num} \
        --worker_id ${id} \
        --return_seq_num ${return_seq_num} \
        > ${log_dir_path}/${id}.log &
done

