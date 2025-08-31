from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

sentence = "I am learning Machine learning"

inputs = tokenizer(sentence, return_tensors="pt")

logits = model(**inputs).logits

predicted_class_id = torch.argmax(logits).item()

print("POSITIVE" if predicted_class_id == 1 else "NEGATIVE")
