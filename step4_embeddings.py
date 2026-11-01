import torch

# Зафиксируем генератор случайных чисел для повторяемости
torch.manual_seed(123)

# 1. Задаем параметры (как в архитектуре GPT)
vocab_size = 50257     # Общий размер словаря GPT-2
max_context_len = 4    # Длина контекста (в нашем тесте 4 токена)
output_dim = 256       # Размерность скрытого вектора (d_in)

# 2. Создаем два слоя эмбеддингов: для слов и для их позиций
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
pos_embedding_layer = torch.nn.Embedding(max_context_len, output_dim)

# 3. Возьмем первый батч входов из нашего предыдущего шага:
# Два примера по 4 токена:
inputs = torch.tensor([
    [44484,   373,  3726,   284],
    [  651,   845, 10032,   286]
])

print("Форма входного тензора токенов:", inputs.shape)  # torch.Size([2, 4])

# 4. Превращаем номера токенов в векторы смысла
token_embeddings = token_embedding_layer(inputs)
print("Форма эмбеддингов токенов:", token_embeddings.shape)  # torch.Size([2, 4, 256])

# 5. Создаем позиционные метки для позиций 0, 1, 2, 3
positions = torch.arange(max_context_len)
pos_embeddings = pos_embedding_layer(positions)
print("Форма позиционных эмбеддингов:", pos_embeddings.shape)  # torch.Size([4, 256])

# 6. Складываем их вместе (PyTorch автоматически добавит pos_embeddings к каждой строке батча)
input_embeddings = token_embeddings + pos_embeddings
print("\nФорма итогового входного тензора модели:", input_embeddings.shape)
print("Пример первых 5 чисел первого токена первого предложения:")
print(input_embeddings[0, 0, :5])