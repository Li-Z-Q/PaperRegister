import json
import tqdm
import torch
import spacy
from utils.embedder import Embedder


nlp = spacy.load("en_core_web_sm")
embedder = Embedder(device='cuda:1')


def do_spacy(text):
    tokenized_text = nlp(text.lower())
    filtered_tokenized_text = [token.lemma_ for token in tokenized_text if not token.is_stop and not token.is_punct]
    return filtered_tokenized_text


def transform_record_2_db(field_2_kb, field_name, node, record_id):
    if node['node_value'] != "":
        if isinstance(node['node_value'], list):
            node_value = " ".join(node['node_value'])
        elif isinstance(node['node_value'], str):
            node_value = node['node_value']
        else:
            raise ValueError(f"node['node_value'] is not str or list, but {type(node['node_value'])}")
        field_2_kb['paper_id'][field_name].append(record_id)
        field_2_kb['content'][field_name].append(node['node_value'])
        field_2_kb['tokens'][field_name].append(do_spacy(node_value))
        field_2_kb['embedding'][field_name].append(embedder.get_embedding(node_value))

    return field_2_kb
    

if __name__ == "__main__":
    
    schema = json.load(open("utils/schema.json", "r"))  
    
    save_dir_path = "registration"
    raw_papers = json.load(open(f"{save_dir_path}/raw_papers.json", "r"))
    id_2_total_paper = {raw_paper['_id']: raw_paper['_source']['files'][0]['content'] for raw_paper in raw_papers}
    for idx in tqdm.tqdm(id_2_total_paper.keys(), desc="get clear total paper"):
        clear_content = "\n".join(id_2_total_paper[idx].split("##")[2:])
        id_2_total_paper[idx] = clear_content

    total_fild_names = []
    for type_key in schema.keys():
        for first_key in schema[type_key].keys():
            total_fild_names.append(f"{type_key} --> {first_key}")
            for second_key in schema[type_key][first_key].keys():
                total_fild_names.append(f"{type_key} --> {first_key} --> {second_key}")
                for nodes in schema[type_key][first_key][second_key]:
                    total_fild_names.append(f"{type_key} --> {first_key} --> {second_key} --> {nodes['node_name']}")
    total_fild_names += ['title', 'abstract', 'total_paper']
    
    field_2_kb = {
        "paper_id": {total_field_name: [] for total_field_name in total_fild_names},
        "content": {total_field_name: [] for total_field_name in total_fild_names},
        "tokens": {total_field_name: [] for total_field_name in total_fild_names},
        "embedding": {total_field_name: [] for total_field_name in total_fild_names}
    }
    print("len(total_fild_names)", len(total_fild_names))
    
    records = [json.loads(line) for line in open(f"{save_dir_path}/registration_step2.jsonl", "r")]
    for record in tqdm.tqdm(records[:]):
        
        good_record = True
        
        for first_node in record['registration']:
            if 'node_name' in first_node:
                field_name = f"{record['category']} --> {first_node['node_name']}"
            else:
                print(f"there is no node_name in {field_name} of {record['id']}")
                good_record = False
                continue
            
            if field_name not in total_fild_names:
                print(f"{field_name} not in total_fild_names")
                good_record = False
                continue
            
            if 'node_value' in first_node:
                field_2_kb = transform_record_2_db(field_2_kb, field_name, first_node, record['id'])
            else:
                print(f"there is no node_value in {field_name} of {record['id']}")
                good_record = False
            
            for second_node in first_node['children']:
                if 'node_name' in second_node:
                    field_name = f"{record['category']} --> {first_node['node_name']} --> {second_node['node_name']}"
                else:
                    print(f"there is no node_name in {field_name} of {record['id']}")
                    good_record = False
                    continue

                if field_name not in total_fild_names:
                    print(f"{field_name} not in total_fild_names")
                    good_record = False
                    continue
            
                if 'node_value' in second_node:
                    field_2_kb = transform_record_2_db(field_2_kb, field_name, second_node, record['id'])
                else:
                    print(f"there is no node_value in {field_name} of {record['id']}")
                    good_record = False
                
                for third_node in second_node['children']:
                    if 'node_name' in third_node:
                        field_name = f"{record['category']} --> {first_node['node_name']} --> {second_node['node_name']} --> {third_node['node_name']}"
                    else:
                        print(f"there is no node_name in {field_name} of {record['id']}")
                        good_record = False
                        continue

                    if field_name not in total_fild_names:
                        print(f"{field_name} not in total_fild_names")
                        good_record = False
                        continue
            
                    if 'node_value' in third_node:
                        field_2_kb = transform_record_2_db(field_2_kb, field_name, third_node, record['id'])
                    else:
                        print(f"there is no node_value in {field_name} of {record['id']}")
                        good_record = False
        
        if good_record:
            field_2_kb['paper_id']['title'].append(record['id'])
            field_2_kb['content']['title'].append(record['title'])
            field_2_kb['tokens']['title'].append(do_spacy(record['title']))
            field_2_kb['embedding']['title'].append(embedder.get_embedding(record['title']))
            
            field_2_kb['paper_id']['abstract'].append(record['id'])
            field_2_kb['content']['abstract'].append(record['abstract'])
            field_2_kb['tokens']['abstract'].append(do_spacy(record['abstract']))
            field_2_kb['embedding']['abstract'].append(embedder.get_embedding(record['abstract']))
            
            field_2_kb['paper_id']['total_paper'].append(record['id'])
            field_2_kb['content']['total_paper'].append(id_2_total_paper[record['id']])
            field_2_kb['tokens']['total_paper'].append(do_spacy(id_2_total_paper[record['id']]))
            field_2_kb['embedding']['total_paper'].append(embedder.get_embedding(id_2_total_paper[record['id']], max_length=12800))
        
    json.dump(field_2_kb['paper_id'], open(f"{save_dir_path}/db/field_2_paper_id.json", "w"))
    json.dump(field_2_kb['content'], open(f"{save_dir_path}/db/field_2_content.json", "w"))
    json.dump(field_2_kb['tokens'], open(f"{save_dir_path}/db/field_2_tokens.json", "w"))
    for field_name in field_2_kb['embedding'].keys():
        field_2_kb['embedding'][field_name] = torch.stack(field_2_kb['embedding'][field_name], dim=0).squeeze(1)
    torch.save(field_2_kb['embedding'], f"{save_dir_path}/db/field_2_embedding.pt")
    
    print("done")