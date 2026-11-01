from src.dataset import create_dataloader_v1

# Тестовый текст подлиннее
raw_text = (
    "Alice was beginning to get very tired of sitting by her sister on the bank, "
    "and of having nothing to do: once or twice she had peeped into the book her sister was reading, "
    "but it had no pictures or conversations in it, 'and what is the use of a book,' thought Alice "
    "'without pictures or conversations?'"
)

# Создаем загрузчик:
# batch_size=2 (по 2 примера в пачке), max_length=4 (длина контекста 4 токена), stride=4 (сдвигаем без наложений)
dataloader = create_dataloader_v1(
    raw_text, 
    batch_size=2, 
    max_length=4, 
    stride=4, 
    shuffle=False
)

print(f"Всего батчей: {len(dataloader)}\n")

for batch_idx, (inputs, targets) in enumerate(dataloader):
    print(f"=== Батч #{batch_idx + 1} ===")
    print("Входы (Inputs - размер тензора", inputs.shape, "):")
    print(inputs)
    print("Цели (Targets - размер тензора", targets.shape, "):")
    print(targets)
    print("-" * 40)