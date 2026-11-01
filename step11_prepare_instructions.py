import json
import urllib.request
from pathlib import Path

# 1. Скачиваем официальный набор данных инструкций
file_path = Path("instruction-data.json")
url = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
    "/main/ch07/01_main-chapter-code/instruction-data.json"
)

if not file_path.exists():
    print("Скачиваем instruction-data.json...")
    urllib.request.urlretrieve(url, file_path)
    print("Готово!")

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Всего записей в датасете: {len(data)}")

# 2. Функция форматирования в стиле Alpaca (Листинг 7.2)
def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry.get("input") else ""
    return instruction_text + input_text

# 3. Посмотрим на два разных примера:
print("\n" + "=" * 50)
print("ПРИМЕР 1 (с входным контекстом Input):")
print("=" * 50)
entry_with_input = data[0]
full_text_1 = format_input(entry_with_input) + f"\n\n### Response:\n{entry_with_input['output']}"
print(full_text_1)

print("\n" + "=" * 50)
print("ПРИМЕР 2 (без Input, чистая инструкция):")
print("=" * 50)
entry_without_input = data[50]
full_text_2 = format_input(entry_without_input) + f"\n\n### Response:\n{entry_without_input['output']}"
print(full_text_2)