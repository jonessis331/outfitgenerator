# tests/test_dropout_weight_decay.py

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.embeddings import EmbeddingLayer
from models.siamese_network import SiameseNetwork
from preprocess.dataset import OutfitCompatibilityDataset
from preprocess.encoder import Encoder
from utils.config import *
from utils.logger import get_logger
from tests.utils import load_data, train_one_epoch, evaluate_model

logger = get_logger(__name__)

def test_dropout_weight_decay(dropout_rates, weight_decays):
    # Load data
    interaction_df, item_df_encoded, encoder = load_data()

    # Prepare dataset
    dataset = OutfitCompatibilityDataset(
        interaction_df, item_df_encoded, encoder.user_encoder, encoder.item_encoder, encoder
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

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

    for dropout_rate in dropout_rates:
        for weight_decay in weight_decays:
            logger.info(f"Testing with dropout_rate={dropout_rate}, weight_decay={weight_decay}")

            # Initialize model
            embedding_layer = EmbeddingLayer(num_users, num_items, num_attributes, embedding_dim)
            model = SiameseNetwork(embedding_layer, embedding_dim, dropout_rate=dropout_rate)
            model.train()

            # Loss and optimizer
            criterion = torch.nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=weight_decay)

            # Training loop
            for epoch in range(NUM_EPOCHS):
                train_loss = train_one_epoch(model, dataloader, criterion, optimizer)
                logger.info(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {train_loss:.4f}")

            # Evaluate model (optional)
            # val_loss = evaluate_model(model, val_dataloader, criterion)
            # logger.info(f"Validation Loss: {val_loss:.4f}")

            # Save results
            # Save model or log results as needed

if __name__ == "__main__":
    dropout_rates = [0.3, 0.5, 0.7]
    weight_decays = [1e-5, 1e-4, 1e-3]
    test_dropout_weight_decay(dropout_rates, weight_decays)
