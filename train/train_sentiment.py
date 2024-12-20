import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import yaml
from models.sentiment_model import SentimentModel
from models.utils import TextDataset, set_seed

with open("train_config.yaml", "r") as f:
    config = yaml.safe_load(f)

set_seed(42)

df = pd.read_csv("../data/synthetic_data.csv")

sentiment_map = {"Positive":0, "Neutral":1, "Negative":2}
df['sent_label'] = df['sentiment'].map(sentiment_map)

train_df, val_df = train_test_split(df, test_size=1-config['split_ratio'], random_state=42)

train_dataset = TextDataset(train_df['query'].tolist(), train_df['sent_label'].tolist(), config['model_name'], max_length=config['max_length'])
val_dataset = TextDataset(val_df['query'].tolist(), val_df['sent_label'].tolist(), config['model_name'], max_length=config['max_length'])

train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

sent_model = SentimentModel(model_name=config['model_name'], num_labels=3)
sent_model.train(train_loader, val_loader, epochs=config['epochs'], lr=float(config['lr']))
sent_model.save("../models/sentiment_model.pt")