import torch
import tiktoken
from src.gpt_model import GPTModel
from step9_decode_strategies import generate

# Конфигурация модели GPT-2 (124M)
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
tokenizer = tiktoken.get_encoding("gpt2")

print("Загружаем вашего обученного ассистента из gpt2_124m_instruction.pth...")
model = GPTModel(BASE_CONFIG)
model.load_state_dict(torch.load("gpt2_124m_instruction.pth", map_location=device, weights_only=True))
model.to(device)
model.eval()
print("[OK] Ассистент готов к работе!\n")

def ask_assistant(instruction, input_context="", temperature=0.7, top_k=25):
    # Форматирование в стиль Alpaca
    prompt = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}"
    )
    if input_context:
        prompt += f"\n\n### Input:\n{input_context}"
    prompt += "\n\n### Response:\n"

    encoded = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)

    with torch.no_grad():
        token_ids = generate(
            model=model,
            idx=encoded,
            max_new_tokens=60,
            context_size=BASE_CONFIG["context_length"],
            temperature=temperature,
            top_k=top_k,
            eos_id=50256  # Остановка на токене конца текста
        )

    full_output = tokenizer.decode(token_ids.squeeze(0).tolist())
    # Вырезаем только ответ модели
    response = full_output[len(prompt):].split("###")[0].strip()
    return response

# Проверим модель на трёх совершенно новых вопросах:
questions = [
    ("Identify the correct spelling of the following word.", "Bannana"),
    ("What is the primary function of a smartphone?", ""),
    ("Write a short polite greeting to a friend.", "")
]

for inst, inp in questions:
    print("-" * 50)
    print(f"Инструкция: {inst}")
    if inp:
        print(f"Входные данные: {inp}")
    answer = ask_assistant(inst, inp)
    print(f"Ответ ассистента:\n>> {answer}")
print("-" * 50)