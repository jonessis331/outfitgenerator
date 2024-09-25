# tests/test_learning_rate_scheduler.py

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, ExponentialLR
from models.embeddings import EmbeddingLayer
from models.siamese_network import SiameseNetwork
from preprocess.dataset import OutfitCompatibilityDataset
from preprocess.encoder import Encoder
from utils.config import *
from utils.logger import get_logger
from tests.utils import load_data, train_one_epoch, evaluate_model

logger = get_logger(__name__)

def test_learning_rate_schedulers():
    # Load data and prepare datasets (similar to previous scripts)
    # ...

    # Define different schedulers to test
    schedulers = {
        'ReduceLROnPlateau': ReduceLROnPlateau,
        'StepLR': StepLR,
        'ExponentialLR': ExponentialLR
    }

    for scheduler_name, scheduler_class in schedulers.items():
        logger.info(f"Testing with scheduler: {scheduler_name}")

        # Initialize model
        embedding_layer = EmbeddingLayer(num_users, num_items, num_attributes, embedding_dim)
        model = SiameseNetwork(embedding_layer, embedding_dim)
        model.train()

        # Loss and optimizer
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # Initialize scheduler
        if scheduler_name == 'ReduceLROnPlateau':
            scheduler = scheduler_class(optimizer, mode='min', factor=0.1, patience=3)
        elif scheduler_name == 'StepLR':
            scheduler = scheduler_class(optimizer, step_size=5, gamma=0.1)
        elif scheduler_name == 'ExponentialLR':
            scheduler = scheduler_class(optimizer, gamma=0.95)

        # Training loop
        for epoch in range(NUM_EPOCHS):
            train_loss = train_one_epoch(model, train_dataloader, criterion, optimizer)
            val_loss = evaluate_model(model, val_dataloader, criterion)
            logger.info(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Update scheduler
            if scheduler_name == 'ReduceLROnPlateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()

            # Optionally, log learning rate
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Current Learning Rate: {current_lr}")

        # Save results or model as needed

if __name__ == "__main__":
    test_learning_rate_schedulers()
