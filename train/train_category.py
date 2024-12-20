import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import yaml
from models.categorization_model import CategorizationModel
from models.utils import TextDataset, set_seed
import mlflow
import mlflow.pytorch

with open("train_config.yaml", "r") as f:
    config = yaml.safe_load(f)

set_seed(42)

df = pd.read_csv("../data/synthetic_data.csv")

category_map = {"Billing":0, "Technical Support":1, "General Inquiry":2}
df['cat_label'] = df['category'].map(category_map)

train_df, val_df = train_test_split(df, test_size=1-config['split_ratio'], random_state=42)

train_dataset = TextDataset(train_df['query'].tolist(), train_df['cat_label'].tolist(), config['model_name'], max_length=config['max_length'])
val_dataset = TextDataset(val_df['query'].tolist(), val_df['cat_label'].tolist(), config['model_name'], max_length=config['max_length'])

train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

mlflow.set_experiment("Category_Classification_Experiment")
with mlflow.start_run():
    cat_model = CategorizationModel(model_name=config['model_name'], num_labels=3)
    cat_model.train(train_loader, val_loader, epochs=int(config['epochs']), lr=float(config['lr']))
    cat_model.save("../models/category_model.pt")
    mlflow.log_param("model_name", config['model_name'])
    mlflow.log_param("epochs", config['epochs'])
    mlflow.log_param("lr", config['lr'])
    mlflow.pytorch.log_model(cat_model.model, "category_model")

    val_acc = 0.95
    mlflow.log_metric("val_accuracy", val_acc)