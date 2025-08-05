import os
import json
import numpy
import pytrec_eval
import pandas as pd


if __name__ == '__main__':
    data_split = "test" # ['dev', 'test']
    match_method = "embedding" # ['embedding', 'bm25']
        
    intent_list = [
        # "goldn",
        "title",
        "abstract",
        "total_paper",
        f"result/plus_{data_split}_Qwen3-0.6B_SFT-AUG-GRPO-Tree_2.0_2/inference_.jsonl_new_db_max",
    ]
    total_results = {
        "intents": [intent[-40:] for intent in intent_list], # get intent file 's name

        "dep1_rec_5": [], "dep1_rec_10": [], "dep1_rec_20": [], "dep1_map": [], "dep1_ndc": [], 
        "dep2_rec_5": [], "dep2_rec_10": [], "dep2_rec_20": [], "dep2_map": [], "dep2_ndc": [], 
        "dep3_rec_5": [], "dep3_rec_10": [], "dep3_rec_20": [], "dep3_map": [], "dep3_ndc": [], 
    }

    for intent in intent_list:
        if intent in ["goldn", "title", "abstract", "total_paper"]:
            result_file_path = f"result/{data_split}_baseline_{intent}_{match_method}_0.jsonl"
        else:
            result_file_path = f"{intent}_{match_method}_0.jsonl"

        if not os.path.exists(result_file_path):
            print(f"File not found: {result_file_path}")
            for key in total_results:
                if key.startswith("dep"):
                    total_results[key].append("NA")
            continue

        results = [json.loads(line) for line in open(result_file_path, "r")]
        print(f"data_split: plus_{data_split} intent: {intent}, match_method: {match_method}, len(results): {len(results)}")
        
        cut_num = 5
        
        recall_matrix = numpy.zeros((4, 4))
        map_cut_matrix = numpy.zeros((4, 4))
        ndcg_cut_matrix = numpy.zeros((4, 4))
        
        recall_matrix_5 = numpy.zeros((4, 4))
        recall_matrix_10 = numpy.zeros((4, 4))
        recall_matrix_20 = numpy.zeros((4, 4))
        
        for layer_level in [1, 2, 3]:
            for combination_level in [1]:
                
                level_3_stat = {}
                
                qrel = {}
                run = {}
                for result in results:
                    
                    if not (result['layer'] == layer_level and result['combination'] == combination_level):
                        continue 
                    
                    qrel[str(result['qid'])] = {
                        result['goldn_answer']: 1
                    }
                    run[str(result['qid'])] = {
                        paper_id: score for paper_id, score in result['predicetion'].items()
                    }
                
                evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'ndcg_cut', 'recall', 'map_cut'})
                scores = evaluator.evaluate(run)

                recall = []
                map_cut = []
                ndcg_cut = []
                
                recall_5 = []
                recall_10 = []
                recall_20 = []
                
                for qid in scores:
                    recall.append(scores[qid][f'recall_{str(cut_num)}'])
                    map_cut.append(scores[qid][f'map_cut_{str(cut_num)}'])
                    ndcg_cut.append(scores[qid][f'ndcg_cut_{str(cut_num)}'])
                    
                    recall_5.append(scores[qid]['recall_5'])
                    recall_10.append(scores[qid]['recall_10'])
                    recall_20.append(scores[qid]['recall_20'])
                    
                    if layer_level == 3:
                        level_3_stat[qid] = scores[qid]['recall_5']

                recall_matrix[layer_level][combination_level] = round(sum(recall) / len(recall) * 100, 1)
                map_cut_matrix[layer_level][combination_level] = round(sum(map_cut) / len(map_cut) * 100, 1)
                ndcg_cut_matrix[layer_level][combination_level] = round(sum(ndcg_cut) / len(ndcg_cut) * 100, 1)
                
                recall_matrix_5[layer_level][combination_level] = round(sum(recall_5) / len(recall_5) * 100, 1)
                recall_matrix_10[layer_level][combination_level] = round(sum(recall_10) / len(recall_10) * 100, 1)
                recall_matrix_20[layer_level][combination_level] = round(sum(recall_20) / len(recall_20) * 100, 1)
                
        recall_matrix = recall_matrix[1:, 1:]
        map_cut_matrix = map_cut_matrix[1:, 1:]
        ndcg_cut_matrix = ndcg_cut_matrix[1:, 1:]
        
        recall_matrix_5 = recall_matrix_5[1:, 1:]
        recall_matrix_10 = recall_matrix_10[1:, 1:]
        recall_matrix_20 = recall_matrix_20[1:, 1:]
                
        # print(f"recall_matrix:{recall_matrix[:, 0]}; map_cut_matrix:{map_cut_matrix[:, 0]}; ndcg_cut_matrix:{ndcg_cut_matrix[:, 0]}")
        
        total_results["dep1_map"].append(map_cut_matrix[0][0])
        total_results["dep2_map"].append(map_cut_matrix[1][0])
        total_results["dep3_map"].append(map_cut_matrix[2][0])
        
        total_results["dep1_ndc"].append(ndcg_cut_matrix[0][0])
        total_results["dep2_ndc"].append(ndcg_cut_matrix[1][0])
        total_results["dep3_ndc"].append(ndcg_cut_matrix[2][0])
        
        total_results["dep1_rec_5"].append(recall_matrix_5[0][0])
        total_results["dep1_rec_10"].append(recall_matrix_10[0][0])
        total_results["dep1_rec_20"].append(recall_matrix_20[0][0])
        
        total_results["dep2_rec_5"].append(recall_matrix_5[1][0])
        total_results["dep2_rec_10"].append(recall_matrix_10[1][0])
        total_results["dep2_rec_20"].append(recall_matrix_20[1][0])
        
        total_results["dep3_rec_5"].append(recall_matrix_5[2][0])
        total_results["dep3_rec_10"].append(recall_matrix_10[2][0])
        total_results["dep3_rec_20"].append(recall_matrix_20[2][0])
        
        # print(f"recall_matrix: \n{recall_matrix[0]}{recall_matrix[1]}{recall_matrix[2]}")
        # print(f"map_cut_matrix: \n{map_cut_matrix[0]}{map_cut_matrix[1]}{map_cut_matrix[2]}")
        # print(f"ndcg_cut_matrix: \n{ndcg_cut_matrix[0]}{ndcg_cut_matrix[1]}{ndcg_cut_matrix[2]}")
    
    results_table = pd.DataFrame(total_results)
    print(results_table)
    
    results_table.to_csv(f"plus_get_score_{data_split}_{match_method}.csv")