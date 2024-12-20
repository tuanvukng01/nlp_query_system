import json
import csv
import random
from langdetect import detect
import fasttext

random.seed(42)

categories = ["Billing", "Technical Support", "General Inquiry"]
sentiments = ["Positive", "Neutral", "Negative"]
issues = ["my latest invoice", "a charge on my account", "accessing my account",
           "connecting to Wi-Fi", "resetting my password",
           "understanding the new feature", "improving the app performance"]
templates = [
     "I need help with {}.",
     "Can someone assist me with {}?",
     "I have a question about {}.",
     "I’m concerned about {}.",
     "Could you clarify {} for me?",
     "I’m having trouble with {}.",
     "What can you tell me about {}?"
 ]

# Additional languages for augmentation
translated_templates_es = [
    "Necesito ayuda con {}.",
    "¿Puede alguien ayudarme con {}?",
    "Tengo una pregunta sobre {}.",
    "Estoy preocupado por {}.",
    "¿Podría aclararme {}?",
    "Tengo problemas con {}.",
    "¿Qué me puede decir sobre {}?"
]

num_samples = 2000
data = []
for _ in range(num_samples):
     cat = random.choice(categories)
     sent = random.choice(sentiments)
     issue = random.choice(issues)
     template = random.choice(templates)
     query = template.format(issue)

    # 20% probability to translate to Spanish for multilingual data
     if random.random() < 0.2:
        query = random.choice(translated_templates_es).format(issue)

     data.append({
         "query": query,
         "category": cat,
         "sentiment": sent
     })

with open("synthetic_data.json", "w") as f:
     json.dump(data, f, indent=2)

with open("synthetic_data.csv", "w") as f:
     writer = csv.DictWriter(f, fieldnames=["query", "category", "sentiment"])
     writer.writeheader()
     for d in data:
         writer.writerow(d)