import time
import json
import torch
import torch.nn as nn
from functools import partial
from torch.utils.data import DataLoader
import tiktoken

from src.gpt_model import GPTModel
from step9_decode_strategies import generate
from step12_instruction_dataloader import (
    InstructionDataset,
    custom_collate_fn,
    format_input
)

# 1. Конфигурация модели GPT-2 (124M)
BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True,
    "emb_dim": 768,
    "n_layers": 12,
    "n_heads": 12
}

device = torch.device("cpu")

# 2. Загружаем датасет и делим на выборки (Листинг 7.3)
with open("instruction-data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

train_portion = int(len(data) * 0.85)  # 935
val_portion = int(len(data) * 0.1)     # 110

train_data = data[:train_portion]
val_data = data[train_portion:train_portion + val_portion]
test_data = data[train_portion + val_portion:]

tokenizer = tiktoken.get_encoding("gpt2")

customized_collate_fn = partial(
    custom_collate_fn,
    device=device,
    allowed_max_length=1024
)

train_loader = DataLoader(
    InstructionDataset(train_data, tokenizer),
    batch_size=4,
    collate_fn=customized_collate_fn,
    shuffle=True,
    drop_last=True
)

val_loader = DataLoader(
    InstructionDataset(val_data, tokenizer),
    batch_size=4,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False
)

# 3. Инициализируем модель и загружаем веса OpenAI
print("Загружаем оригинальные веса OpenAI из gpt2_124m_openai.pth...")
model = GPTModel(BASE_CONFIG)
model.load_state_dict(torch.load("gpt2_124m_openai.pth", map_location=device, weights_only=True))
model.to(device)

# 4. Проверяем ответ ДО тонкой настройки
test_prompt = format_input(val_data[0]) + "\n\n### Response:\n"
encoded_prompt = torch.tensor(tokenizer.encode(test_prompt)).unsqueeze(0).to(device)

print("\n" + "=" * 50)
print("ОТВЕТ БАЗОВОЙ МОДЕЛИ ДО ОБУЧЕНИЯ (Раздел 7.5):")
print("=" * 50)
model.eval()
with torch.no_grad():
    token_ids = generate(
        model=model,
        idx=encoded_prompt,
        max_new_tokens=30,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256
    )
print(tokenizer.decode(token_ids.squeeze(0).tolist())[len(test_prompt):].strip())
print("=" * 50 + "\n")

# 5. Функции вычисления потерь с учётом маски -100
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = nn.functional.cross_entropy(
        logits.flatten(0, 1),
        target_batch.flatten(0, 1),
        ignore_index=-100
    )
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

# 6. Запуск тонкой настройки (Листинг 7.8)
print("Начинаем тонкую настройку (Instruction Fine-Tuning)...")
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)
num_epochs = 2

start_time = time.time()
global_step = -1

for epoch in range(num_epochs):
    model.train()
    for input_batch, target_batch in train_loader:
        optimizer.zero_grad()
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        loss.backward()
        optimizer.step()
        global_step += 1

        if global_step % 25 == 0:
            model.eval()
            with torch.no_grad():
                train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
                val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)
            print(f"Эпоха {epoch+1} (Шаг {global_step:04d}): Train loss {train_loss:.3f} | Val loss {val_loss:.3f}")
            model.train()

end_time = time.time()
print(f"\nОбучение завершено за {(end_time - start_time) / 60:.2f} минут!")

# Сохраняем модель-ассистента
torch.save(model.state_dict(), "gpt2_124m_instruction.pth")
print("[OK] Дообученные веса сохранены в 'gpt2_124m_instruction.pth'!")

# 7. Проверяем ответ ПОСЛЕ тонкой настройки
print("\n" + "=" * 50)
print("ОТВЕТ МОДЕЛИ ПОСЛЕ ТОНКОЙ НАСТРОЙКИ:")
print("=" * 50)
model.eval()
with torch.no_grad():
    token_ids = generate(
        model=model,
        idx=encoded_prompt,
        max_new_tokens=30,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256
    )
print(tokenizer.decode(token_ids.squeeze(0).tolist())[len(test_prompt):].strip())
print("=" * 50)