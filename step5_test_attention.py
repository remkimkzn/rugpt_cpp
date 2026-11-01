import torch
from src.attention import MultiHeadAttention

torch.manual_seed(123)

# Параметры теста
batch_size = 2
seq_len = 4
emb_dim = 256
num_heads = 4

# Создаем случайный батч эмбеддингов (аналогично выходу предыдущего шага)
x = torch.rand(batch_size, seq_len, emb_dim)

# Инициализируем слой многоголового внимания
mha = MultiHeadAttention(
    d_in=emb_dim,
    d_out=emb_dim,
    context_length=seq_len,
    dropout=0.0,
    num_heads=num_heads
)

# Прогоняем данные через механизм внимания
out = mha(x)

print("Форма входного тензора:", x.shape)
print("Форма выходного тензора после внимания:", out.shape)
print("Пример значений первого вектора на выходе:")
print(out[0, 0, :5])