import torch
import tiktoken
from transformers import GPT2Model as HF_GPT2Model
from src.gpt_model import GPTModel

BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True,
    "emb_dim": 768,
    "n_layers": 12,
    "n_heads": 12
}


def generate(model, idx, max_new_tokens, context_size,
             temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )

        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if idx_next == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def load_hf_weights_into_gpt(gpt, hf_gpt):
    with torch.no_grad():
        # 1. Эмбеддинги
        gpt.tok_emb.weight.copy_(hf_gpt.wte.weight)
        gpt.pos_emb.weight.copy_(hf_gpt.wpe.weight)

        # 2. Блоки трансформера
        for b in range(len(hf_gpt.h)):
            hf_b = hf_gpt.h[b]
            my_b = gpt.trf_blocks[b]

            my_b.norm1.scale.copy_(hf_b.ln_1.weight)
            my_b.norm1.shift.copy_(hf_b.ln_1.bias)

            q_w, k_w, v_w = torch.chunk(hf_b.attn.c_attn.weight, 3, dim=-1)
            my_b.att.W_query.weight.copy_(q_w.T)
            my_b.att.W_key.weight.copy_(k_w.T)
            my_b.att.W_value.weight.copy_(v_w.T)

            q_b, k_b, v_b = torch.chunk(hf_b.attn.c_attn.bias, 3, dim=-1)
            my_b.att.W_query.bias.copy_(q_b)
            my_b.att.W_key.bias.copy_(k_b)
            my_b.att.W_value.bias.copy_(v_b)

            my_b.att.out_proj.weight.copy_(hf_b.attn.c_proj.weight.T)
            my_b.att.out_proj.bias.copy_(hf_b.attn.c_proj.bias)

            my_b.norm2.scale.copy_(hf_b.ln_2.weight)
            my_b.norm2.shift.copy_(hf_b.ln_2.bias)

            my_b.ff.layers[0].weight.copy_(hf_b.mlp.c_fc.weight.T)
            my_b.ff.layers[0].bias.copy_(hf_b.mlp.c_fc.bias)
            my_b.ff.layers[2].weight.copy_(hf_b.mlp.c_proj.weight.T)
            my_b.ff.layers[2].bias.copy_(hf_b.mlp.c_proj.bias)

        # 3. Финальная нормализация и голова выхода
        gpt.final_norm.scale.copy_(hf_gpt.ln_f.weight)
        gpt.final_norm.shift.copy_(hf_gpt.ln_f.bias)
        gpt.out_head.weight.copy_(hf_gpt.wte.weight)


print("1. Скачиваем официальные веса GPT-2 (124M)...")
hf_gpt = HF_GPT2Model.from_pretrained("openai-community/gpt2")

print("2. Переносим веса в наш собственный класс GPTModel...")
model_openai = GPTModel(BASE_CONFIG)
load_hf_weights_into_gpt(model_openai, hf_gpt)
model_openai.eval()

torch.save(model_openai.state_dict(), "gpt2_124m_openai.pth")
print("[OK] Веса успешно перенесены и сохранены в файл gpt2_124m_openai.pth!")

tokenizer = tiktoken.get_encoding("gpt2")
start_context = "Every effort moves you"
encoded = torch.tensor(tokenizer.encode(start_context)).unsqueeze(0)

print("\n3. Генерируем текст моделью с оригинальными весами:")
torch.manual_seed(123)
out_tokens = generate(
    model=model_openai,
    idx=encoded,
    max_new_tokens=25,
    context_size=BASE_CONFIG["context_length"],
    top_k=50,
    temperature=1.0
)
print("Ответ модели:\n", tokenizer.decode(out_tokens.squeeze(0).tolist()))