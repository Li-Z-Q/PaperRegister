import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
import tqdm
import torch
import argparse
from utils.prompt import prompt
from plus_router_get_tree import tree_str
from transformers import AutoModelForCausalLM, AutoTokenizer


parser = argparse.ArgumentParser()
parser.add_argument("--worker_id", type=int, default=0)
parser.add_argument("--worker_num", type=int, default=1)
parser.add_argument("--model_path", type=str, default="/141nfs/username/hf_models/Qwen3-1.7B-SFT/checkpoint-500")
parser.add_argument("--prediction_save_dir", type=str, default="result/tmp")
parser.add_argument("--data_split", type=str, default="dev", choices=["dev", "test"])
parser.add_argument("--return_seq_num", type=int, help="number of return sequences")
args = parser.parse_args()
for k, v in vars(args).items():
    print(f"{k}: {v}")


tokenizer = AutoTokenizer.from_pretrained(f"{args.model_path}")
model = AutoModelForCausalLM.from_pretrained(f"{args.model_path}", torch_dtype=torch.float16, device_map="cuda:0")
model.eval()

START_STR = " **"
VOCAB_SIZE = tokenizer.vocab_size
EOS_TOKEN_ID = tokenizer.eos_token_id
CONSTRAIN_DICT = json.load(open("utils/plus_tree.json"))


def constrain_func(_, input_ids):
    
    if input_ids[-1] in [EOS_TOKEN_ID, tokenizer.pad_token_id]:
        return [tokenizer.pad_token_id]
    
    start_tokens_id = tokenizer(START_STR, add_special_tokens=False)["input_ids"]
    
    start_pos = -1
    for i in reversed(range(len(input_ids) - len(start_tokens_id) + 1)):
        if [int(j) for j in list(input_ids[i: i + len(start_tokens_id)])] == start_tokens_id:
            start_pos = i + len(start_tokens_id) - 1
            break
    
    if start_pos == -1:
        return list(range(VOCAB_SIZE)) # if input do not contain "Title:", use total vocab as candidates

    con_dict = CONSTRAIN_DICT
    for i in range(start_pos, len(input_ids)):
        if str(int(input_ids[i])) not in con_dict:
            return [EOS_TOKEN_ID]
        con_dict = con_dict[str(int(input_ids[i]))]
        if isinstance(con_dict, str): # here str should be int, but anyway, it will never be used, because only when input_ids[-1] is ** or <|im_end|> can cause here, but if it is ** or <|im_end|>, it will directly stop in generate, due to they are stop_word
            return [EOS_TOKEN_ID]

    next_token_ids = []
    if isinstance(con_dict, dict):
        next_token_ids = [int(i) for i in con_dict.keys()]
    else:
        next_token_ids = [EOS_TOKEN_ID]
    return next_token_ids


@torch.no_grad()
def generate_constrained(input_text):
    model_inputs = tokenizer([input_text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs, max_new_tokens=128, 
        prefix_allowed_tokens_fn=constrain_func, 
        num_return_sequences=args.return_seq_num, 
        # do_sample=True, top_p=0.99999, temperature=25.0, 
        num_beams = args.return_seq_num,
        eos_token_id=[EOS_TOKEN_ID, tokenizer(START_STR, add_special_tokens=False)["input_ids"][0]]
    )
    
    contents = []
    for i in range(generated_ids.shape[0]):
        output_ids = generated_ids[i][len(model_inputs.input_ids[0]):].tolist() 

        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")

        content = content.strip().strip("**").strip()
        contents.append(content)
        
    return contents


@torch.no_grad()
def generate_thinking(input_text):
    model_inputs = tokenizer([input_text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs, max_new_tokens=1024,
        eos_token_id=[EOS_TOKEN_ID, 151668] # 151668 is </think>
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    
    return thinking_content


if __name__ == "__main__":

    if args.data_split == "dev":
        datas = json.load(open("data_train/datas_dev.json", "r"))
    elif args.data_split == "test":
        datas = json.load(open("data_test/datas.json", "r"))
    else:
        raise ValueError("args.data_split must be dev or test")
    
    datas = datas[args.worker_id::args.worker_num]
    candidate_intents = [intent.strip().replace('<|im_end|>', "") for intent in tree_str.split("\n")]
    
    prediction_save_dir = args.prediction_save_dir
    if not os.path.exists(prediction_save_dir):
        os.makedirs(prediction_save_dir, exist_ok=True)
    prediction_save_path = f"{prediction_save_dir}/inference_{args.worker_id}.jsonl"
    
    prediction_fw = open(prediction_save_path, "a")
    exist_ids = [data['qid'] for data in [json.loads(line) for line in open(prediction_save_path, "r")]]
    
    for data in tqdm.tqdm(datas):
        if data['qid'] in exist_ids:
            print(f"{data['qid']} has been processed")
            continue
        
        query = prompt.format(query=data["query"])

        if 0:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": query}], tokenize=False, add_generation_prompt=True, enable_thinking=True 
            )
            thinking = generate_thinking(text)
            if thinking.endswith('</think>'):
                text = f"{text}{thinking}\n\nIntent:{START_STR}"
            else:
                print(f"=============== {data['qid']} thinking is bad, it is {thinking}")
                text = f"{text}<think>\n\n</think>\n\nIntent:{START_STR}"
        else:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": query}], tokenize=False, add_generation_prompt=True, enable_thinking=False 
            ) + f"Intent:{START_STR}"
        
        outputs = generate_constrained(text)
        assert all([output in candidate_intents for output in outputs]), f"outputs {outputs} not in candidate_intents {candidate_intents}"
        data['predictions'] = outputs
        
        prediction_fw.write(json.dumps(data, ensure_ascii=False) + "\n")
        prediction_fw.flush()
    
    prediction_fw.close()
    
    print(f"{args.data_split} {args.model_path} done")