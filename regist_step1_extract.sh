source /141nfs/username/anaconda3/bin/activate paper

# 定义所有worker的URL列表
urls=(
    ""
    ""
)
worker_num=${#urls[@]}  # 自动获取worker数量

# 循环启动worker进程
for ((i=0; i<worker_num; i++)); do
    nohup python -u regist_step1_extract.py \
        --url "${urls[i]}" \
        --worker_id $i \
        --worker_num $worker_num \
        > "regist_step1_extract_$i.log" &
done