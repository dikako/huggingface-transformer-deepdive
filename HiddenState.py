from transformers import AutoModel, AutoTokenizer

checkpoint = "distilbert-base-uncased"
model = AutoModel.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

sentence = "I am learning Machine learning"

inputs = tokenizer(sentence, return_tensors="pt")

print(inputs)

outputs = model(**inputs)

print(outputs.last_hidden_state.shape)

