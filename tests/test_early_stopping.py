# tests/test_early_stopping.py

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.embeddings import EmbeddingLayer
from models.siamese_network import SiameseNetwork
from preprocess.dataset import OutfitCompatibilityDataset
from preprocess.encoder import Encoder
from utils.config import *
from utils.logger import get_logger
from tests.utils import load_data, train_one_epoch, evaluate_model, EarlyStopping

logger = get_logger(__name__)

def test_early_stopping():
    # Load data
    interaction_df, item_df_encoded, encoder = load_data()

    # Split data into training and validation sets
    # For simplicity, we'll use a simple split here
    train_df = interaction_df.sample(frac=0.8, random_state=42)
    val_df = interaction_df.drop(train_df.index)

    # Prepare datasets
    train_dataset = OutfitCompatibilityDataset(
        train_df, item_df_encoded, encoder.user_encoder, encoder.item_encoder, encoder
    )
    val_dataset = OutfitCompatibilityDataset(
        val_df, item_df_encoded, encoder.user_encoder, encoder.item_encoder, encoder
    )

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model parameters
    num_users = len(encoder.user_encoder.classes_)
    num_items = len(encoder.item_encoder.classes_)
    num_attributes = {}
    for attr in ATTRIBUTES:
        if attr in encoder.attribute_encoders:
            num_attributes[attr] = len(encoder.attribute_encoders[attr].classes_)
        elif attr in encoder.multi_label_encoders:
            num_attributes[attr] = len(encoder.multi_label_encoders[attr].classes_)
        else:
            continue
    embedding_dim = EMBEDDING_DIM

    # Initialize model
    embedding_layer = EmbeddingLayer(num_users, num_items, num_attributes, embedding_dim)
    model = SiameseNetwork(embedding_layer, embedding_dim)
    model.train()

    # Loss and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Early stopping
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_dataloader, criterion, optimizer)
        val_loss = evaluate_model(model, val_dataloader, criterion)
        logger.info(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Check early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            logger.info("Early stopping triggered.")
            break

    # Save results or model as needed

if __name__ == "__main__":
    test_early_stopping()
