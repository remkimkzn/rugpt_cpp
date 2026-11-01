import torch
import tiktoken
from src.gpt_model import GPTModel

# Та же конфигурация компактной модели, которую мы обучали
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": True
}


def generate(model, idx, max_new_tokens, context_size,
             temperature=0.0, top_k=None, eos_id=None):
    """
    Продвинутая генерация текста (Листинг 5.4 из книги):
    поддерживает температуру и отсечение top-k.
    """
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # 1. Фильтруем логиты с помощью выборки top-k
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )

        # 2. Применяем температурное масштабирование и вероятностную выборку
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            # Выбираем токен пропорционально вероятностям
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            # Жадный выбор (при T=0.0 выбирается токен с наибольшим баллом)
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if idx_next == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx


# 1. Загружаем модель с сохраненными после 5 эпох весами
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(torch.load("gpt2_small_scratch.pth", weights_only=True))
model.eval()

tokenizer = tiktoken.get_encoding("gpt2")
start_context = "Every effort moves you"
encoded = torch.tensor(tokenizer.encode(start_context)).unsqueeze(0)

print("=== Сравнение режимов генерации ===\n")

# Режим 1: Бухгалтер (Жадный выбор, T=0.0)
torch.manual_seed(123)
out_greedy = generate(model, encoded, max_new_tokens=15, context_size=256, temperature=0.0)
print("1. Бухгалтер (Жадный выбор, T=0.0):")
print(tokenizer.decode(out_greedy.squeeze(0).tolist()))

# Режим 2: Джазовый музыкант (T=1.2, top_k=25)
torch.manual_seed(123)
out_creative = generate(model, encoded, max_new_tokens=15, context_size=256, temperature=1.2, top_k=25)
print("\n2. Джазовый музыкант (T=1.2, top_k=25):")
print(tokenizer.decode(out_creative.squeeze(0).tolist()))

# Режим 3: Другая случайная импровизация (меняем сид)
torch.manual_seed(456)
out_creative2 = generate(model, encoded, max_new_tokens=15, context_size=256, temperature=1.2, top_k=25)
print("\n3. Джазовый музыкант (T=1.2, другой сид 456):")
print(tokenizer.decode(out_creative2.squeeze(0).tolist()))