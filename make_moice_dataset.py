import json
from tqdm import tqdm

# входной файл
INPUT = "data/train_raw.txt"

# выходные файлы
TRAIN_JSON = "data/train_data.json"
EVAL_JSON = "data/eval_data.json"

# сколько символов на один sample
CHUNK = 1024   # можно увеличить до 2048

# какой процент уходит в eval
EVAL_SPLIT = 0.02

# читаем сырой текст
with open(INPUT, "r", encoding="utf-8") as f:
    text = f.read()

samples = []

# разбиваем на куски
for i in tqdm(range(0, len(text) - 2*CHUNK, CHUNK)):
    user_chunk = text[i : i + CHUNK]
    assistant_chunk = text[i + CHUNK : i + 2*CHUNK]

    sample = {
        "messages": [
            {"role": "user", "content": user_chunk},
            {"role": "assistant", "content": assistant_chunk}
        ]
    }

    samples.append(sample)

# делим на train / eval
split = int(len(samples) * (1 - EVAL_SPLIT))
train_data = samples[:split]
eval_data = samples[split:]

# сохраняем
with open(TRAIN_JSON, "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

with open(EVAL_JSON, "w", encoding="utf-8") as f:
    json.dump(eval_data, f, ensure_ascii=False, indent=2)

print(f"Создано {len(train_data)} train samples и {len(eval_data)} eval samples.")

