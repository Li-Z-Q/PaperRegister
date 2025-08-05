import os
import json
import torch
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import warnings
import pytrec_eval
from torch import nn
from rank_bm25 import BM25Okapi
from plus_router_get_tree import tree_str
from online import do_spacy, get_match_score
from typing import Any, Callable, Optional, Union
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.extras.profiling import profiling_context, profiling_decorator
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_trainer import nanstd
from trl.trainer.utils import pad
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="Qwen3-0.6B-SFT")
parser.add_argument("--output_dir", type=str, default="tmp")
parser.add_argument("--temperature", type=float, default=2.0)
parser.add_argument("--train_data_path", type=str, default="")
parser.add_argument("--per_device_train_batch_size", type=int, default=10)
parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
parser.add_argument("--reward_method", type=str, default="tmp", choices=['Tree', 'Retrieval', 'Both', "ZeroOne"]) 
args = parser.parse_args()
for k, v in vars(args).items():
    print(f"{k}: {v}")
    
tokenizer = AutoTokenizer.from_pretrained(f"{args.model_path}")
candidate_intents = [intent.strip().replace('<|im_end|>', "") for intent in tree_str.split("\n")]
field_2_paper_id = json.load(open(f"registration/db/field_2_paper_id.json"))
field_2_content = json.load(open(f"registration/db/field_2_content.json"))
field_2_bm25 = {field_name: BM25Okapi(tokens) for field_name, tokens in json.load(open(f"registration/db/field_2_tokens.json")).items()}
field_name_map = json.load(open("utils/plus_field_name_map.json", "r"))
query_2_paperid = {
    data['query']: data['goldn_answer'] for data in json.load(open("data_train/datas_all.json", "r"))
}
plus_datas = []
for i in range(3):
    plus_datas += [json.loads(line) for line in open(f"data_train/plus_datas_train.jsonl_aug.jsonl.tmp_{i}", "r").readlines()]
for data in plus_datas:
    if 'title' in data:
        content_index = field_2_content['title'].index(data['title'])
        paper_id = field_2_paper_id['title'][content_index]
    elif 'abstract' in data:
        content_index = field_2_content['abstract'].index(data['abstract'])
        paper_id = field_2_paper_id['abstract'][content_index]
    else:
        raise ValueError(f"data {data} does not contain title or abstract")
    query_2_paperid[data['query']] = paper_id
raw_logs = open("router_get_train_data_aug.log_used", "r").readlines()
current_seed_query = "INIT"
for log_str in raw_logs:
    if log_str.startswith("seed query:"):
        seed_query = log_str.split("seed query:")[-1].strip()
        assert seed_query in query_2_paperid, f"seed query {seed_query} not in query_2_paperid"
        current_seed_query = seed_query
    elif log_str.startswith("new query:"):
        new_query = log_str.split("new query:")[-1].strip()
        assert current_seed_query in query_2_paperid, f"current_seed_query {current_seed_query} not in query_2_paperid"
        query_2_paperid[new_query] = query_2_paperid[current_seed_query]
    else:
        continue
print(f"query_2_paperid: {len(query_2_paperid)} queries loaded")

START_STR = " **"
VOCAB_SIZE = tokenizer.vocab_size
EOS_TOKEN_ID = tokenizer.eos_token_id
CONSTRAIN_DICT = json.load(open("utils/plus_tree.json"))
CONSTRAIN_FUNC_FLAG_DICT = {
    gpu_num: ["init" for _ in range(args.gradient_accumulation_steps*args.per_device_train_batch_size)] for gpu_num in range(torch.cuda.device_count())
}

def constrain_func(batch_id, input_ids):
    global CONSTRAIN_FUNC_FLAG_DICT
    if CONSTRAIN_FUNC_FLAG_DICT[input_ids.device.index][batch_id] == "first":
        CONSTRAIN_FUNC_FLAG_DICT[input_ids.device.index][batch_id] = "not-first"
    elif CONSTRAIN_FUNC_FLAG_DICT[input_ids.device.index][batch_id] == "not-first":
        if input_ids[-1] in [EOS_TOKEN_ID, tokenizer(START_STR, add_special_tokens=False)["input_ids"][0]]:
            CONSTRAIN_FUNC_FLAG_DICT[input_ids.device.index][batch_id] = "stop"
            return [tokenizer.pad_token_id]
    elif CONSTRAIN_FUNC_FLAG_DICT[input_ids.device.index][batch_id] == "stop":
        return [tokenizer.pad_token_id]
    else:
        raise f"{CONSTRAIN_FUNC_FLAG_DICT} {input_ids.device.index}, {batch_id}, {CONSTRAIN_FUNC_FLAG_DICT[input_ids.device.index][batch_id]} is not not-first or first"
   
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


def reward_funcs(completions, original_inputs, **kwargs):
    
    for c in range(len(completions)):
        completions[c] = completions[c].strip("**").strip()
        assert completions[c] in candidate_intents, f"completions {completions[c]} not in candidate_intents {candidate_intents}"
    
    rewards = []
    both_rewards = []
    assert args.reward_method in ["Tree", "Retrieval", "Both", "ZeroOne"], f"reward_method {args.reward_method} not in ['Tree', 'Retrieval', 'Both', ZeroOne]"
    for com, ori_inp in zip(completions, original_inputs):
        tree_reward = 0
        retrieval_reward = 0
        
        if args.reward_method in ["Retrieval", "Both"]:
            query = ori_inp['prompt'].split('<Query>')[-1].strip().split('<|im_end|>')[0]
            assert query in query_2_paperid, f"query {query} not in query_2_paperid"
            sorted_paper_ids, sorted_scores = get_match_score(
                query_representation=do_spacy(query),
                paper_ids= field_2_paper_id[field_name_map[com]],
                bm25_db= field_2_bm25[field_name_map[com]],
                embedding_db=None,  # No embedding database used in this case
                match_method="bm25"
            )
            goldn_paper_id = query_2_paperid[query]

            # 把 sorted_scores 中的值映射到 0-2 区间，需要避免 division by zero
            sorted_scores = [
                2.0 * (score - min(sorted_scores) + 1e-6) / (max(sorted_scores) - min(sorted_scores) + 1e-6) 
                for score in sorted_scores
            ]
            
            qrel = {'0': {goldn_paper_id: 1}}
            run = {'0': {paper_id: score for paper_id, score in zip(sorted_paper_ids, sorted_scores)}}
            evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'ndcg_cut', 'recall', 'map_cut'})
            reall_at_5 = evaluator.evaluate(run)['0'][f'recall_5']
            
            retrieval_reward = run['0'][goldn_paper_id] if goldn_paper_id in run['0'] else 0.0
            rewards.append(retrieval_reward)
            
        if args.reward_method in ["Tree", "Both"]:
            for goldn_com in [
                ori_com.replace("<|im_end|>", "").strip() for ori_com in ori_inp['completion'].split("**")
            ]:
                overlap_num = 0
                com_word_list = com.split(' --> ')
                goldn_com_word_list = goldn_com.split(' --> ')
                for i in range(min(len(com_word_list), len(goldn_com_word_list))):
                    if com_word_list[i] == goldn_com_word_list[i]:
                        overlap_num += 1
                    else:
                        break
                this_reward = overlap_num / len(com_word_list) + overlap_num / len(goldn_com_word_list)
                if this_reward > tree_reward:
                    tree_reward = this_reward
            
            rewards.append(tree_reward)

        both_rewards.append((retrieval_reward + tree_reward) / 2)
    
        if args.reward_method == "ZeroOne":
            for goldn_com in [
                ori_com.replace("<|im_end|>", "").strip() for ori_com in ori_inp['completion'].split("**")
            ]:
                overlap_num = 0
                com_word_list = com.split(' --> ')
                goldn_com_word_list = goldn_com.split(' --> ')
                for i in range(min(len(com_word_list), len(goldn_com_word_list))):
                    if com_word_list[i] == goldn_com_word_list[i]:
                        overlap_num += 1
                    else:
                        break
                # this_reward = overlap_num / len(com_word_list) + overlap_num / len(goldn_com_word_list)
                if overlap_num == len(com_word_list) and overlap_num == len(goldn_com_word_list):
                    this_reward = 1.0
                else:
                    this_reward = 0.0
                if this_reward > tree_reward:
                    tree_reward = this_reward
            
            rewards.append(tree_reward)
        
    if args.reward_method == "Both":
        rewards = both_rewards
    
    print(f"\ncompletions, original_inputs, len=={len(completions)}")
    
    total_show = {}
    for i in range(len(completions)):
        reward = rewards[i]
        com = completions[i]
        ori_inp = original_inputs[i]

        query = ori_inp['prompt'].split('<Query>')[-1]
        if query not in total_show:
            total_show[query] = {
                "goldn": ori_inp['completion'],
                "preds": []
            }
        total_show[query]['preds'].append(f"reward: '{reward}'; com: '{com}'")
        
    print(json.dumps(total_show, indent=2))
    
    return rewards


class MyGRPOTrainer(GRPOTrainer):
    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        mode = "eval" if self.control.should_evaluate else "train"

        prompts = [x["prompt"] for x in inputs] # num is num_gpu * batch_size * sum_grad_steps
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(
            text=prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = super(GRPOTrainer, self)._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Generate completions using either vLLM or regular generation
        if self.use_vllm:
            # First, have main process load weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            all_prompts_text = gather_object(prompts_text)
            if self.accelerator.is_main_process:
                # Since 'prompts' contains 'num_generations' duplicates, we first take unique prompts, and generate
                # num_generations outputs for each one. This is faster than generating outputs for each duplicate
                # prompt individually.
                ordered_set_of_prompts = all_prompts_text[:: self.num_generations]
                with profiling_context(self, "vLLM.generate"):
                    completion_ids = self.vllm_client.generate(
                        prompts=ordered_set_of_prompts,
                        n=self.num_generations,
                        repetition_penalty=self.repetition_penalty,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        top_k=-1 if self.top_k is None else self.top_k,
                        min_p=0.0 if self.min_p is None else self.min_p,
                        max_tokens=self.max_completion_length,
                        guided_decoding_regex=self.guided_decoding_regex,
                    )
            else:
                completion_ids = [None] * len(all_prompts_text)
            # Broadcast the completions from the main process to all processes, ensuring each process receives its
            # corresponding slice.
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
            completion_ids = completion_ids[process_slice]

            # Pad the completions, and concatenate them with the prompts
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else:
            # Regular generation path
            with unwrap_model_for_generation(
                self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
            ) as unwrapped_model:
                global CONSTRAIN_FUNC_FLAG_DICT
                for gn in CONSTRAIN_FUNC_FLAG_DICT:
                    CONSTRAIN_FUNC_FLAG_DICT[gn] = ["first" for _ in range(len(CONSTRAIN_FUNC_FLAG_DICT[gn]))]
                prompt_completion_ids = unwrapped_model.generate(
                    prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config,
                    prefix_allowed_tokens_fn=constrain_func, 
                )

            # Compute prompt length and extract completion ids
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
        if self.mask_truncated_completions:
            truncated_completions = ~is_eos.any(dim=1)
            completion_mask = completion_mask * (~truncated_completions).unsqueeze(1).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        with torch.no_grad():
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's
            # computation here, and use per_token_logps.detach() instead.
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep, batch_size
                )
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep, batch_size
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep, batch_size
                    )

        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes, self.reward_func_names)
        ):
            with profiling_context(self, reward_func_name):
                if isinstance(
                    reward_func, nn.Module
                ):  # Module instead of PretrainedModel for compat with compiled models
                    if is_conversational(inputs[0]):
                        messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                        texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions)]
                    reward_inputs = reward_processing_class(
                        text=texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                    )
                    reward_inputs = super(GRPOTrainer, self)._prepare_inputs(reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
                else:
                    # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                    keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                    reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                    output_reward_func = reward_func(prompts=prompts, completions=completions, original_inputs=inputs, **reward_kwargs)
                    # Convert None values to NaN
                    output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]

                    rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()}
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards
        if self.scale_rewards:
            advantages = advantages / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Log the metrics
        if mode == "train":
            self.state.num_input_tokens_seen += self.accelerator.gather_for_metrics(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # log completion lengths, mean, min, max
        agg_completion_mask = self.accelerator.gather_for_metrics(completion_mask.sum(1))
        self._metrics[mode]["completions/mean_length"].append(agg_completion_mask.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_mask.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_mask.float().max().item())

        # identify sequences that terminated with EOS and log their lengths
        agg_terminated_with_eos = self.accelerator.gather_for_metrics(is_eos.any(dim=1))
        term_completion_mask = agg_completion_mask[agg_terminated_with_eos]
        clipped_completions_ratio = 1 - len(term_completion_mask) / len(agg_completion_mask)
        self._metrics[mode]["completions/clipped_ratio"].append(clipped_completions_ratio)
        if len(term_completion_mask) == 0:
            # edge case where no completed sequences are found
            term_completion_mask = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_mask.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_mask.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_mask.float().max().item())

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        # Log prompt and completion texts
        self._textual_logs["prompt"].extend(gather_object(prompts_text))
        self._textual_logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._textual_logs["rewards"][name].extend(rewards_per_func[:, i].tolist())

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
        }


def main():
    # load jsonl dataset
    train_dataset = load_dataset("json", data_files=args.train_data_path, split="train")

    # Train model
    training_args = GRPOConfig(
        output_dir=f"{args.output_dir}",
        logging_steps=2,
        # learning_rate=2e-5,
        bf16=True,
        use_liger_kernel=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # max_length=8192,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        # dataset_num_proc=32,
        num_train_epochs=2,
        warmup_ratio=0.1,
        save_strategy="epoch",
        use_vllm=False,
        top_p=0.99,
        top_k=999999,
        temperature=args.temperature,
        num_generations=5
    )
    print("training_args:\n", training_args)
    
    trainer = MyGRPOTrainer(
        model=args.model_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
    )
    trainer.train()

    trainer.save_model()


if __name__ == "__main__":
    main()
    
    print("done")