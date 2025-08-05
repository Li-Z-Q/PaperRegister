import json
import tqdm
import torch
import spacy
import argparse
from rank_bm25 import BM25Okapi
from utils.embedder import Embedder


nlp = spacy.load("en_core_web_sm")


def get_match_score(query_representation, paper_ids, bm25_db, embedding_db, match_method):
    if match_method == "bm25":
        scores = bm25_db.get_scores(query_representation).tolist()
    elif match_method == "embedding":
        scores = torch.cosine_similarity(query_representation, embedding_db.to(query_representation.device)).cpu().tolist()
    else:
        raise NotImplementedError(f"not implemented match method: {match_method}")
    
    if not len(paper_ids) == len(scores):
        print(f"!!!! WARNING!!!! len(paper_ids) {len(paper_ids)} != len(scores) {len(scores)}")
    
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    sorted_scores = [scores[i] for i in sorted_indices]
    sorted_paper_ids = [paper_ids[i] for i in sorted_indices]
    
    return sorted_paper_ids, sorted_scores    


def do_spacy(text):
    tokenized_text = nlp(text.lower())
    filtered_tokenized_text = [token.lemma_ for token in tokenized_text if not token.is_stop and not token.is_punct]
    return filtered_tokenized_text


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker_id", type=int, default=0)
    parser.add_argument("--worker_num", type=int, default=1)
    parser.add_argument("--data_split", type=str, default="test", choices=['dev', 'test'])
    parser.add_argument("--intent", type=str, default="/141nfs/username/paper_register/result/plus_test_Qwen3-0.6B_SFT-AUG-GRPO-Both_2.0_40/inference_.jsonl") # ['goldn', 'title', 'abstract', 'total_paper', 'path']
    parser.add_argument("--match_method", type=str, default="bm25", choices=['bm25', 'embedding'])
    parser.add_argument("--score_method", type=str, default="_max", choices=['', '_max'])
    args = parser.parse_args()
    for k, v in vars(args).items():
        print(f"{k}: {v}")
        
    # if args.data_split == "test" and args.intent == "goldn": # now test also has goldn intent
    #     raise NotImplementedError("test data doesn't have goldn intent")
    
    if args.data_split == "test":
        test_datas = json.load(open("data_test/datas.json", "r"))[args.worker_id::args.worker_num]
    elif args.data_split == "dev":
        test_datas = json.load(open("data_train/datas_dev.json", "r"))[args.worker_id::args.worker_num]
    else:
        raise NotImplementedError(f"not implemented data_split: {args.data_split}")
    
    if args.intent in ["goldn", "title", "abstract", "total_paper"]:
        save_path = f"result/{args.data_split}_baseline_{args.intent}_{args.match_method}_{args.worker_id}.jsonl"
        qid_2_intents = None
    else:
        field_name_map = json.load(open("utils/plus_field_name_map.json", "r"))
        
        save_path = f"{args.intent}_new_db{args.score_method}_{args.match_method}_{args.worker_id}.jsonl"
        qid_2_intents = {
            data["qid"]: [
                field_name_map[intent] for intent in data["predictions"]
            ]  for data in [json.loads(line) for line in open(f"{args.intent}", "r")]    
        }
        if not len(test_datas) == len(qid_2_intents):
            print(f"!!!! WARNING!!!! len(test_datas) {len(test_datas)} != len(qid_2_intents) {len(qid_2_intents)}")

    print(f"save_path: {save_path}")
        
    if args.match_method == "embedding":
        embedder = Embedder(device='cuda:0')
    elif args.match_method == "bm25":
        embedder = None
    else:
        raise NotImplementedError(f"not implemented match method: {args.match_method}")
    schema = json.load(open("utils/schema.json", "r"))
    
    db_dir_path = "registration/db"
    field_2_paper_id = json.load(open(f"{db_dir_path}/field_2_paper_id.json"))
    field_2_content = json.load(open(f"{db_dir_path}/field_2_content.json"))
    field_2_bm25 = {
        field_name: BM25Okapi(tokens) for field_name, tokens in json.load(open(f"{db_dir_path}/field_2_tokens.json")).items()
    }
    field_2_embedding = torch.load(f"{db_dir_path}/field_2_embedding.pt")
    
    result_fw = open(save_path, "a")
    existing_query_ids = [data['qid'] for data in [json.loads(line) for line in open(save_path, "r")]]
    for data in tqdm.tqdm(test_datas):
        
        if data['qid'] in existing_query_ids:
            print(f"\n{data['qid']} already exists")
            continue
        
        if data['qid'] not in qid_2_intents:
            print(f"\n{data['qid']} not in qid_2_intents")
            continue
        
        if args.match_method == "bm25":
            query_representation = do_spacy(data["query"])
        else:
            query_representation = embedder.get_embedding(data["query"])
        
        if args.intent in ["title", "abstract", "total_paper"]:
            sorted_paper_ids, sorted_scores = get_match_score(
                query_representation, field_2_paper_id[args.intent],
                field_2_bm25[args.intent], field_2_embedding[args.intent], args.match_method
            )
        else:
            if args.intent == "goldn":
                intents = data["intents"]
            else:
                intents = qid_2_intents[data["qid"]]
                
            paper_id_2_total_score_list = {}
            for intent in intents:
                sorted_paper_ids, sorted_scores = get_match_score(
                    query_representation, field_2_paper_id[intent],
                    field_2_bm25[intent], field_2_embedding[intent], args.match_method
                )
                for paper_id, score in zip(sorted_paper_ids, sorted_scores):
                    if paper_id not in paper_id_2_total_score_list:
                        paper_id_2_total_score_list[paper_id] = [score]
                    else:
                        paper_id_2_total_score_list[paper_id].append(score)
            
            sorted_scores_by_avg = [sum(scores) / len(scores) for paper_id, scores in sorted(paper_id_2_total_score_list.items(), key=lambda x: sum(x[1]) / len(x[1]), reverse=True)]
            sorted_paper_ids_by_avg = [paper_id for paper_id, scores in sorted(paper_id_2_total_score_list.items(), key=lambda x: sum(x[1]) / len(x[1]), reverse=True)]
            sorted_scores_by_max = [max(scores) for paper_id, scores in sorted(paper_id_2_total_score_list.items(), key=lambda x: max(x[1]), reverse=True)]
            sorted_paper_ids_by_max = [paper_id for paper_id, scores in sorted(paper_id_2_total_score_list.items(), key=lambda x: max(x[1]), reverse=True)]
            
            if args.score_method == "_max":
                sorted_paper_ids = sorted_paper_ids_by_max
                sorted_scores = sorted_scores_by_max
            else:
                sorted_paper_ids = sorted_paper_ids_by_avg
                sorted_scores = sorted_scores_by_avg
                        
        data['predicetion'] = {
            paper_id: score for paper_id, score in zip(sorted_paper_ids[:100], sorted_scores[:100])
        }
        
        result_fw.write(json.dumps(data, ensure_ascii=False) + "\n")
        result_fw.flush()
        
    result_fw.close()
    
    print(f"done ! result_file_path: {save_path}")