import json
import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken
from functools import partial

# 1. Загружаем датасет
with open("instruction-data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Разделение данных: 85% train, 10% val, 5% test (Листинг 7.3)
train_portion = int(len(data) * 0.85)  # 935 примеров
val_portion = int(len(data) * 0.1)     # 110 примеров

train_data = data[:train_portion]
val_data = data[train_portion:train_portion + val_portion]
test_data = data[train_portion + val_portion:]

print(f"Обучающая выборка: {len(train_data)} записей")
print(f"Проверочная выборка: {len(val_data)} записей")
print(f"Тестовая выборка: {len(test_data)} записей")

# 2. Функция форматирования (Листинг 7.2)
def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry.get("input") else ""
    return instruction_text + input_text

# 3. Класс набора данных (Листинг 7.4)
class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(tokenizer.encode(full_text))

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)

# 4. Пользовательская функция упаковки батча с паддингом и маской -100 (Листинг 7.5)
def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=1024,
    device="cpu"
):
    # Находим самую длинную последовательность в текущем пакете + 1 токен для сдвига цели
    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        # Дополняем токены конца текста до batch_max_length
        padded = (
            new_item + [pad_token_id] * (batch_max_length - len(new_item))
        )

        # inputs: входная строка (все токены кроме последнего)
        inputs = torch.tensor(padded[:-1])
        # targets: целевая строка (сдвинутая на 1 вправо)
        targets = torch.tensor(padded[1:])

        # Заменяем все дополняющие токены pad_token_id в целях кроме первого на -100
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # Ограничиваем максимальную длину при необходимости
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor

# 5. Проверяем работу загрузчика DataLoader
tokenizer = tiktoken.get_encoding("gpt2")
device = torch.device("cpu")

customized_collate_fn = partial(
    custom_collate_fn,
    device=device,
    allowed_max_length=1024
)

train_dataset = InstructionDataset(train_data, tokenizer)
train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    collate_fn=customized_collate_fn,
    shuffle=True,
    drop_last=True
)

# Берем первый батч для проверки
for inputs, targets in train_loader:
    print("\n[OK] Первый тестовый батч успешно сформирован:")
    print("Форма входного тензора inputs:", inputs.shape)
    print("Форма целевого тензора targets:", targets.shape)
    print("\nПример части целевого тензора первой строки (видны -100 в конце):")
    print(targets[0][-15:].tolist())
    break