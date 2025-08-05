import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
from torch import Tensor
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


class Embedder:
    def __init__(self, device='cuda'):
        model_path = "/141nfs/username/hf_models/gte-Qwen2-7B-instruct"
        print(f"Loading embedder model from {model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        if device == 'cpu':
            self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, device_map=device)
        else:
            self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16, device_map=device)
        self.model.eval()

    @torch.no_grad()
    def get_embedding(self, text, max_length=4096):
        assert isinstance(text, str), "text must be a string"
        batch_dict = self.tokenizer([text], max_length=max_length, padding=True, truncation=True, return_tensors='pt').to(self.model.device)
        outputs = self.model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        
        return embeddings.to('cpu')
