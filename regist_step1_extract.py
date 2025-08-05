import os
import json
import tqdm
import time
import argparse
from utils.vllm_api import VLLMAPI


classification_prompt = """
Determine the category of the paper based on its abstract. The category options are as follows:  

Algorithm Innovation: Proposes new systems, new models, new training methods, new inference approaches, new data organization methods, etc.  
Benchmark Construction: Introduces a new benchmark.  
Mechanism Exploration: Investigates and analyzes the mechanisms of existing systems, algorithms, phenomena, or functionalities.  
Theory Proof: Proves a certain theory or formula.  
Survey and Review: Summarizes the research landscape of a particular field.  

Your final output should be only the best correct category of the paper, do not contain any other information or explanation.  

<abstract>  
{abstract}
"""


extraction_prompt = """
You are an content extraction expert, particularly skilled at extracting structured records from academic papers. Below, I need you to perform extraction based on a given schema. Please note the following:

1. Your extraction does not need to preserve the original text verbatim. You may paraphrase or summarize the content to make the extracted content more comprehensive and fluent.  
2. Not all field names in the schema will have corresponding content in the paper. If you cannot find a precise match in the paper, leave that field empty.  
3. Your final output must strictly adhere to the original schema in JSON format, starting with '```json' and ending with '```'.  

<schema>  
{schema}  

<paper>  
{paper}  
"""


def get_clear_paper(raw_paper):
    last_name = ""
    name_2_paragraphs = {}
    for p, paragraph in enumerate(raw_paper['_source']["files"][0]["content"].split("##")):
        
        paragraph = paragraph.strip()
        paragraph_line0 = paragraph.split("\n")[0].strip()
        paragraph_token0 = paragraph_line0.split(" ")[0].strip('.')
        paragraph_line999 = paragraph.replace(paragraph_line0, "").strip()
        
        if is_number(paragraph_token0) == 'other':
            continue
        elif is_number(paragraph_token0) == 'int': # big paragraph_title, such as Method, Experiment
            last_name = paragraph_line0
            name_2_paragraphs[last_name] = [paragraph_line999]
        else:
            if last_name == "":
                continue
            name_2_paragraphs[last_name].append(paragraph_line999)

    sections = []
    for section_name, paragraphs in name_2_paragraphs.items():
        # if "Introduct" in section_name \
        # or "Relate" in section_name \
        # or "Background" in section_name:
        #     continue
        if "Relate" in section_name \
        or "Background" in section_name:
            continue
        
        section = ""
        for line in (" ".join(paragraphs)).split("\n"):
            if '|' in line:
                continue
            section += line + "\n"
        sections.append(f"{section_name}\n{section}")
        
        if "Conclusion" in section_name:
            break

    paper = "\n".join(sections)
    
    return name_2_paragraphs, sections, paper

    

def is_number(s):
    if s.isdigit():  
        return "int"
    else:
        try:
            float(s)
            return 'float'
        except ValueError:  
            return 'other'


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
    category_save_path = f"{save_dir_path}/category_{args.worker_id}.jsonl"
    extraction_save_path = f"{save_dir_path}/registration_step1_{args.worker_id}.jsonl"
    extraction_save_big_path = f"{save_dir_path}/registration_step1.jsonl"
    
    raw_papers = json.load(open(f"{save_dir_path}/raw_papers.json", "r"))[args.worker_id::args.worker_num]
    
    category_fw = open(category_save_path, "w")
    extraction_fw = open(extraction_save_path, "a")
    existing_paper_ids = [data['id'] for data in [json.loads(line) for line in open(extraction_save_path, "r")]]
    
    print(f"existing paper ids: {len(existing_paper_ids)}, raw papers: {len(raw_papers)}")
    
    if os.path.exists(extraction_save_big_path):
        existing_paper_ids += [data['id'] for data in [json.loads(line) for line in open(extraction_save_big_path, "r")]]
        existing_paper_ids = list(set(existing_paper_ids))
        print(f"after loading big extraction, existing paper ids: {len(existing_paper_ids)}")

    for raw_paper in tqdm.tqdm(raw_papers, desc="paper"):    
        try:
            idx = raw_paper["_id"]
            year = raw_paper['_source']["year"]
            title = raw_paper['_source']["title"]
            autors = raw_paper['_source']["author"]
            abstract = raw_paper['_source']["description"]

            if idx in existing_paper_ids:
                print(f'\nalready exists: {idx}')
                continue

            completion, content = vllm_api.get_response(
                input_text=classification_prompt.format(abstract=(title+abstract)), 
                return_complete_completion=True, enable_thinking=True
            )
            for category in [
                "Algorithm Innovation", "Benchmark Construction", "Mechanism Exploration", "Theory Proof", "Survey and Review", 
                "error"
            ]:
                try:
                    content = content.split('\n')[-1]
                except Exception as e:
                    print(f"{idx} error {e}")
                    content = "error"
                if category in content:
                    break
            category_fw.write(json.dumps({"idx": idx, "category": category, "title": title, "content": content}, ensure_ascii=False) + "\n")
            category_fw.flush()
            
            wanted_type_list = [
                "Algorithm Innovation", "Benchmark Construction", "Mechanism Exploration", 
                "Theory Proof", "Survey and Review"
            ]
            if category not in wanted_type_list:
                print(f"{idx} is {category}, not in {wanted_type_list}")
                continue

            name_2_paragraphs, sections, paper = get_clear_paper(raw_paper)

            for try_times in range(5):
                try:
                    completion, content = vllm_api.get_response(
                        input_text=extraction_prompt.format(schema=schema[category], paper=paper), 
                        return_complete_completion=True, enable_thinking=True
                    )
                    content = content.split('```json')[-1]
                    content = json.loads(content.strip().strip("```").strip())
                    break
                except Exception as e:
                    content = e
                    try_times += 1
                    print(f"\n\ntry times: {try_times}, {e}")
        
            extraction_fw.write(json.dumps({
                "id": idx,
                "category": category,
                "registration": content,
                "year": year,
                "title": title,
                "author": autors,
                "abstract": abstract
            }, ensure_ascii=False) + "\n")
            extraction_fw.flush()

        except Exception as e:
            print(f"{idx} error {e}")
            continue
        
    category_fw.close()
    extraction_fw.close()
    
    print("done")