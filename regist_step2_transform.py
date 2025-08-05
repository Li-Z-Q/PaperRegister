import os
import json
import tqdm
import copy
import time
import argparse
from utils.vllm_api import VLLMAPI


prompt = """\
You are an information integration expert, and now I need your help to complete an information integration task. 
I will provide you with a two-level tree structure, including a root node and two child nodes. Ideally, the content of the root node should be a summary and generalization of the two child nodes. Your task is to generate the content of the root node based on the content of the two child nodes. Note the following:  

1. The input I provide you is a dictionary in JSON format, including the keys `root_name` and `children`, where the value of `children` is a list of child nodes.  
2. Each child node contains three fields: `node_name`, `node_desc`, and `node_value`. You should primarily use the content of `node_value` for summarization and generalization.  
3. The `root_value` field should provide an abstraction and summary of the two child nodes’ contents. In other words, the root_value must not repeat keywords from the child nodes; instead, it should abstract based on those keywords. More strictly, the root_value length must not exceed that of either child node.
4. Your output should be a dictionary in JSON format, meaning the input dictionary will have content for the `root_value` field. The final output should start with '```json' and end with '```'.  

<input tree>
{tree}
"""


def get_root_value(root_name, children):
    for try_times in range(5):
        try:
            completion, content = vllm_api.get_response(
                input_text=prompt.format(tree=json.dumps(
                    {
                        "root_name": root_name,
                        "root_value": "",
                        "children": children
                    }, ensure_ascii=False, indent=4)
                ), 
                return_complete_completion=True, enable_thinking=True
            )
            content = content.split('```json')[-1]
            content = json.loads(content.strip().strip("```").strip())
            assert "root_value" in content, f"content should have key 'root_value', now is {content}"
            break
        except Exception as e:
            content = "error"
            try_times += 1
            print(f"\n\ntry times: {try_times}, {e}")

    return content


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker_id", type=int, default=0)
    parser.add_argument("--worker_num", type=int, default=1)
    parser.add_argument("--url", type=str, default="")
    args = parser.parse_args()
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    
    vllm_api = VLLMAPI(base_url=f"http://{args.url}/v1", model_name="Qwen")
    
    schema = json.load(open("utils/schema.json", "r"))
    
    save_dir_path = "registration"
    extraction_save_path = f"{save_dir_path}/registration_step2_{args.worker_id}.jsonl"
    extraction_save_big_path = f"{save_dir_path}/registration_step2.jsonl"

    raw_datas = [json.loads(line) for line in open(f"{save_dir_path}/registration_step1.jsonl", "r")][args.worker_id::args.worker_num]
    
    extraction_fw = open(extraction_save_path, "a")
    existing_paper_ids = [data['id'] for data in [json.loads(line) for line in open(extraction_save_path, "r")]]
    print(f"existing paper ids: {len(existing_paper_ids)}")
    
    if os.path.exists(extraction_save_big_path):
        existing_paper_ids += [data['id'] for data in [json.loads(line) for line in open(extraction_save_big_path, "r")]]
        existing_paper_ids = list(set(existing_paper_ids))
        print(f"after loading big extraction, existing paper ids: {len(existing_paper_ids)}")
    
    for raw_data in tqdm.tqdm(raw_datas, desc="paper"):            
        print('\n' + raw_data['id'])
        if raw_data['id'] in existing_paper_ids:
            print("already extracted")
            continue
        
        try:
            
            registration = raw_data["registration"]
            
            first_nodes = []
            for first_key in list(registration.keys()):
                second_nodes = []
                for second_key in list(registration[first_key].keys()):
                    children = registration[first_key][second_key]
                    node_desc = "summary of " + " and ".join([child['node_desc'] for child in children])
                    second_nodes.append({
                        "node_name": second_key,
                        "node_desc": node_desc,
                        "node_value": "",
                        "children": children
                    })
                
                children = second_nodes
                node_desc = "overview of " + " and ".join([child['node_desc'] for child in children])
                first_nodes.append({
                    "node_name": first_key,
                    "node_desc": node_desc,
                    "node_value": "",
                    "children": second_nodes
                })
            
            registration = copy.deepcopy(first_nodes) # get a list
            
            first_nodes = []
            for first_node in registration:
                second_nodes = []
                for second_node in first_node["children"]:
                    root_name = second_node["node_name"]
                    children = second_node["children"]
                    content = get_root_value(root_name, children)
                    if "root_value" in content:
                        second_node["node_value"] = content["root_value"]
                    else:
                        second_node["node_value"] = " and ".join([child['node_value'] for child in children])
                    second_nodes.append({
                        "node_name": second_node["node_name"],
                        "node_desc": second_node["node_desc"],
                        "node_value": second_node["node_value"],
                        "children": second_node["children"]
                    })
                
                root_name = first_node["node_name"]
                children = copy.deepcopy(second_nodes)
                for child in children:
                    child.pop("children")
                content = get_root_value(root_name, children)
                if "root_value" in content:
                    first_node["node_value"] = content["root_value"]
                else:
                    first_node["node_value"] = " and ".join([child['node_value'] for child in children])
                first_nodes.append({
                    "node_name": first_node["node_name"],
                    "node_desc": first_node["node_desc"],
                    "node_value": first_node["node_value"],
                    "children": second_nodes
                })
            
            registration = copy.deepcopy(first_nodes) # get a list
        
            raw_data['registration'] = registration
            extraction_fw.write(json.dumps(raw_data, ensure_ascii=False) + "\n")
            extraction_fw.flush()
        
        except Exception as e:
            print(f"error: {e}")
            continue
        
    extraction_fw.close()
    
    print("done")