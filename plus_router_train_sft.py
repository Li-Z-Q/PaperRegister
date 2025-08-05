import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import argparse


"""
Train Gemma-3 on the Codeforces COTS dataset.

accelerate launch --config_file examples/accelerate_configs/deepspeed_zero3.yaml examples/scripts/sft_gemma3.py
"""

from datasets import load_dataset
from transformers import AutoModelForCausalLM

from trl import SFTConfig, SFTTrainer


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="Qwen3-0.6B")
parser.add_argument("--output_dir", type=str, default="models")
parser.add_argument("--train_data_path", type=str, default="train_data/datas_train.jsonl")
parser.add_argument("--per_device_train_batch_size", type=int, default=1)
parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
args = parser.parse_args()
for k, v in vars(args).items():
    print(f"{k}: {v}")
    

def main():
    # load jsonl dataset
    train_dataset = load_dataset("json", data_files=args.train_data_path, split="train")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(args.model_path)

    # Train model
    training_args = SFTConfig(
        output_dir=f"{args.output_dir}",
        logging_steps=2,
        bf16=True,
        use_liger_kernel=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=8192,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataset_num_proc=32,
        num_train_epochs=5,
        warmup_ratio=0.1,
        save_strategy="epoch"
    )
    print("training_args:\n", training_args)
    
    trainer = SFTTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
    )
    trainer.train()
    
    trainer.save_model()

    # # Push to hub
    # trainer.push_to_hub(dataset_name="open-r1/codeforces-cots")


if __name__ == "__main__":
    main()
    
    print("done")