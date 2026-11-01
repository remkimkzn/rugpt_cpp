import torch
import tiktoken
from src.gpt_model import GPTModel

# Та же конфигурация, которую мы обучали в run_pretraining.py
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": True
}

def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

# 1. Создаем пустую модель и загружаем наши ОБУЧЕННЫЕ веса
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(torch.load("gpt2_small_scratch.pth", weights_only=True))
model.eval()

tokenizer = tiktoken.get_encoding("gpt2")
start_context = "Every effort moves you"
encoded = torch.tensor(tokenizer.encode(start_context)).unsqueeze(0)

out_ids = generate_text_simple(model, encoded, max_new_tokens=15, context_size=256)
print("Ответ модели ПОСЛЕ 5 эпох предобучения:")
print(tokenizer.decode(out_ids.squeeze(0).tolist()))