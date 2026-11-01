import torch
import tiktoken
from src.gpt_model import GPTModel

# Конфигурация для компактной GPT (как в книге Себастьяна Рашки)
GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Размер словаря BPE
    "context_length": 256,  # Контекстное окно токенов
    "emb_dim": 256,         # Размерность вектора
    "n_heads": 4,           # Количество голов внимания
    "n_layers": 4,          # Количество блоков трансформера
    "drop_rate": 0.0,       # Dropout
    "qkv_bias": False       # Без смещений в Q, K, V
}

# 1. Создаем саму модель
model = GPTModel(GPT_CONFIG_124M)

# 2. Берем исходный текст и превращаем в токены (первый этап)
tokenizer = tiktoken.get_encoding("gpt2")
text = "Every effort moves you forward"
inputs = torch.tensor([tokenizer.encode(text)])

print("Входной текст:", text)
print("Форма входного тензора:", inputs.shape)

# 3. Прогоняем через модель
out = model(inputs)

print("Форма выходного тензора (логитов):", out.shape)
print(f"Пояснение формы {out.shape}: [1 батч, {inputs.shape[1]} токенов, {GPT_CONFIG_124M['vocab_size']} вероятностных оценок в словаре]")