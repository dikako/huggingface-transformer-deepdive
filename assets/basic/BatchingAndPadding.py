from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

texts = ["I love NML", "I am learning machine learning"]

inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

print(inputs.input_ids)