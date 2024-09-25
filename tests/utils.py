# tests/utils.py

import torch
from preprocess.data_loader import DataLoader as CustomDataLoader
from preprocess.encoder import Encoder
from utils.config import *
from utils.logger import get_logger

logger = get_logger(__name__)

def load_data():
    # Load data
    data_loader = CustomDataLoader(OUTFIT_DATA_PATH, INTERACTION_DATA_PATH, USER_DATA_PATH)
    outfit_data = data_loader.load_outfit_data()
    interaction_df = data_loader.load_interaction_data()
    user_df = data_loader.load_user_data()

    # Flatten outfit data
    item_df = data_loader.flatten_data(outfit_data)

    # Initialize encoder and fit
    encoder = Encoder()
    encoder.fit_user_encoder(interaction_df['user_id'])
    encoder.fit_item_encoder(item_df['item_id'])
    encoder.fit_attribute_encoders(item_df, ATTRIBUTES)

    # Encode item attributes
    item_df_encoded = encoder.transform_attributes(item_df)

    return interaction_df, item_df_encoded, encoder

def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0
    for batch in dataloader:
        user_ids, item_ids1, item_ids2, item_attrs1, item_attrs2, labels = batch
        optimizer.zero_grad()

        outputs = model(user_ids, item_ids1, item_ids2, item_attrs1, item_attrs2)
        loss = criterion(outputs.squeeze(), labels.float())
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def evaluate_model(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            user_ids, item_ids1, item_ids2, item_attrs1, item_attrs2, labels = batch
            outputs = model(user_ids, item_ids1, item_ids2, item_attrs1, item_attrs2)
            loss = criterion(outputs.squeeze(), labels.float())
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    return avg_loss

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        """
        Args:
            patience (int): How many epochs to wait after last time validation loss improved.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# In tests/utils.py

def contrastive_loss(output, label, margin=1.0):
    """
    Compute the contrastive loss as defined in Hadsell-et-al.'06
    """
    label = label.float()
    euclidean_distance = F.pairwise_distance(output[0], output[1])
    loss = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                      (label) * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2))
    return loss
