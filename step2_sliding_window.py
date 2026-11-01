import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
text = "Alice was beginning to get very tired of sitting"
token_ids = tokenizer.encode(text)

# Задаем размер контекстного окна  сколько токенов модель держит в поле зрения
context_size = 4

print(f"Полная цепочка токенов: {token_ids}\n")

# Возьмем первые два токена для входа (x) и 4 токена со сдвигом на 1 для цели (y)
x=token_ids[:context_size]
y=token_ids[1:context_size+1]

print(f"x (вход): {x}")
print(f"y (цель): {y}\n")

# Посмотрим, как это выглядит для человека
print("Как обучается модель шаг за шагом:")
for i in range(1, context_size+1):
    context = token_ids[:i]
    target = token_ids[i]
    context_words= tokenizer.decode(context)
    target_word = tokenizer.decode([target])
    print(f"Контекст: {str(context):<20} ('{context_words}') -----> Цель: {target} ('{target_word}') ")