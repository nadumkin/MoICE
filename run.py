from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_path = "./models/qwen2-0.5b"

bnb_config = BitsAndBytesConfig(load_in_8bit=True)

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto"
)

inputs = tokenizer("Explain the MoICE method.", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

