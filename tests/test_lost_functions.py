# tests/test_loss_functions.py

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from models.embeddings import EmbeddingLayer
from models.siamese_network import SiameseNetwork
from preprocess.dataset import OutfitCompatibilityDataset
from preprocess.encoder import Encoder
from utils.config import *
from utils.logger import get_logger
from tests.utils import load_data, evaluate_model

logger = get_logger(__name__)

def train_one_epoch_contrastive(model, dataloader, optimizer, margin=1.0):
    model.train()
    total_loss = 0
    for batch in dataloader:
        user_ids, item_ids1, item_ids2, item_attrs1, item_attrs2, labels = batch
        optimizer.zero_grad()

        outputs = model(user_ids, item_ids1, item_ids2, item_attrs1, item_attrs2)
        loss = contrastive_loss(outputs, labels, margin)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def test_loss_functions():
    # Load data and prepare datasets
    # ...

    # Initialize model
    embedding_layer = EmbeddingLayer(num_users, num_items, num_attributes, embedding_dim)
    model = SiameseNetwork(embedding_layer, embedding_dim)
    model.train()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch_contrastive(model, train_dataloader, optimizer)
        logger.info(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss:.4f}")

        # Optionally, evaluate model
        # val_loss = evaluate_model(model, val_dataloader, criterion)
        # logger.info(f"Validation Loss: {val_loss:.4f}")

    # Save results or model as needed

if __name__ == "__main__":
    test_loss_functions()
