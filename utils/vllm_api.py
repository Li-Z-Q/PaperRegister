from openai import OpenAI
from transformers import AutoTokenizer


class VLLMAPI():
    def __init__(self, base_url="http:///v1", model_name="Qwen"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained("/141nfs/username/hf_models/gpt2")
        self.client = OpenAI(base_url=base_url, api_key="sk-jlspXzMhnhze2fF9ds3WvVXK8LeaJIFW2ueGIiP8q6mFHy2s")
        print(f"use my vllm api, base url: {base_url}, model_name: {model_name}")
        
    def get_response(self, input_text, max_completion_tokens=4096, return_complete_completion=False, show_usage=False, enable_thinking=False):
        
        total_tokens = self.tokenizer.encode(input_text)
        total_token_num = len(total_tokens)
        if total_token_num > 128000:
            print(f"input text too long, truncate from {total_token_num} to 128000")
            input_text = self.tokenizer.decode(total_tokens[:128000])
        
        for try_time in range(5):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": input_text}
                    ],
                    max_completion_tokens=max_completion_tokens,
                    presence_penalty=1.5,
                    temperature=0.7,
                    top_p=0.8,
                    extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
                )
                
                if show_usage:
                    print("usage:", completion.usage)
                
                if return_complete_completion:
                    return completion, completion.choices[0].message.content
                else:
                    return completion.choices[0].message.content
                
            except Exception as e:
                print(f"try_time in vllm_api.py {try_time}, error: {e}")
                continue
            
        if return_complete_completion:
            return "error", "error"
        else:
            return "error"