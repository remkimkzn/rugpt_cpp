import torch
import tiktoken
from src.gpt_model import GPTModel

torch.manual_seed(123)

# Конфигурация нашей компактной модели
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 256,
    "n_heads": 4,
    "n_layers": 4,
    "drop_rate": 0.0,
    "qkv_bias": False
}


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """Генерирует новые токены по одному за шаг"""
    for _ in range(max_new_tokens):
        # 1. Обрезаем контекст, если он длиннее окна модели
        idx_cond = idx[:, -context_size:]
        
        # 2. Получаем предсказания модели (логиты)
        with torch.no_grad():
            logits = model(idx_cond)
        
        # 3. Берем оценки только для самого последнего слова: [batch, 50257]
        logits = logits[:, -1, :]
        
        # 4. Переводим оценки в вероятности (от 0 до 1)
        probas = torch.softmax(logits, dim=-1)
        
        # 5. Выбираем ID слова с самой высокой вероятностью
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        
        # 6. Приклеиваем угаданное слово к нашей фразе и идем на следующий круг
        idx = torch.cat((idx, idx_next), dim=1)
        
    return idx


# Инициализируем модель и токенизатор
model = GPTModel(GPT_CONFIG_124M)
model.eval()  # Режим использования (отключает случайности вроде dropout)
tokenizer = tiktoken.get_encoding("gpt2")

# Начальная фраза
start_context = "Every effort moves you"
encoded_tensor = torch.tensor(tokenizer.encode(start_context)).unsqueeze(0)

print("Начало фразы:", start_context)

# Просим модель дописать 10 новых слов
out_ids = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"]
)

# Переводим числа обратно в текст
decoded_text = tokenizer.decode(out_ids.squeeze(0).tolist())
print("\nЧто сочинила необученная модель:")
print(decoded_text)