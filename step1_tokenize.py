import tiktoken

#1. Загружаем официальный токенизатор BPE для GPT-2
tokenizer = tiktoken.get_encoding("gpt2");

#2. Тестовый фрагмент текста
text = "Alice was beginning to get very tired of sitting" 

#3. Преобразуем строку в последовательность идентификаторов токенов
token_ids = tokenizer.encode(text)

print("Исходный текст:", text)
print("Идентификаторы токенов (Token IDs):", token_ids)
print(f"Всего токенов: {len(token_ids)}")

# 4. Показываем соответствие каждого токена его числу
print("\nРазбор по токенам:")
for tid in token_ids:
    print(f"ID {tid:<6} -> '{tokenizer.decode([tid])}'")
