from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

sentences = ["I am learning Machine learning",
             "I love ML and HuggingFace",
             "I am not great at learning human science",
             "Transformers are great at understanding natural language"]

inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)

logits = model(**inputs).logits

predicted_class_id = torch.argmax(logits, dim=-1).tolist()

labels = ["NEGATIVE", "POSITIVE"]

for sentences, predicted_class_id in zip(sentences, predicted_class_id):
    print(f"Sentence: {sentences}")
    print(f"Predicted class: {labels[predicted_class_id]}")