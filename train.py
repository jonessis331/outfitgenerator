# train.py

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from preprocess.data_loader import DataLoader as CustomDataLoader
from models.embeddings import EmbeddingLayer
from models.siamese_network import SiameseNetwork
from preprocess.dataset import OutfitCompatibilityDataset
from preprocess.encoder import Encoder
from utils.config import *
from utils.logger import get_logger
import pandas as pd
import random

logger = get_logger(__name__)

def train_model():
    logger.info("Starting training process...")

    # Load data
    data_loader = CustomDataLoader(OUTFIT_DATA_PATH, INTERACTION_DATA_PATH, USER_DATA_PATH)
    outfit_data = data_loader.load_outfit_data()
    interaction_df = data_loader.load_interaction_data()
    user_df = data_loader.load_user_data()

    # Flatten outfit data
    logger.info("Flattening outfit data...")
    item_df = data_loader.flatten_data(outfit_data)

    # Initialize encoder and fit
    logger.info("Initializing and fitting encoders...")
    encoder = Encoder()
    encoder.fit_user_encoder(interaction_df['user_id'])
    encoder.fit_item_encoder(item_df['item_id'])
    encoder.fit_attribute_encoders(item_df, ATTRIBUTES)

    # Prepare dataset
    logger.info("Preparing dataset...")
    dataset = OutfitCompatibilityDataset(
        interaction_df, item_df, encoder.user_encoder, encoder.item_encoder, encoder
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

    # Initialize model
    logger.info("Initializing model...")
    # Determine single-valued and multi-valued attributes
    single_valued_attrs = list(encoder.attribute_encoders.keys())
    multi_valued_attrs = list(encoder.multi_label_encoders.keys())

    embedding_layer = EmbeddingLayer(
        num_users, num_items, num_attributes, embedding_dim, single_valued_attrs, multi_valued_attrs
    )

    model = SiameseNetwork(embedding_layer, embedding_dim)
    model.train()

    # Loss and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    logger.info("Starting training loop...")
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for batch in dataloader:
            user_ids, item_ids1, item_ids2, item_attrs1, item_attrs2, labels = batch

            user_ids = user_ids.long()
            item_ids1 = item_ids1.long()
            item_ids2 = item_ids2.long()
            labels = labels.float()

            # Ensure item attributes are of correct data type
            item_attrs1 = {k: v.long() if k in single_valued_attrs else v.float() for k, v in item_attrs1.items()}
            item_attrs2 = {k: v.long() if k in single_valued_attrs else v.float() for k, v in item_attrs2.items()}

            optimizer.zero_grad()
            outputs = model(user_ids, item_ids1, item_ids2, item_attrs1, item_attrs2)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {avg_loss:.4f}")

    # Save model
    logger.info("Saving model...")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    encoder.save_encoders(ENCODERS_PATH)
    logger.info("Training completed successfully.")

if __name__ == "__main__":
    train_model()


# In train.py or a separate utils file
def custom_collate_fn(batch):
    user_ids, item_ids1, item_ids2, item_attrs1_list, item_attrs2_list, labels = zip(*batch)

    user_ids = torch.stack(user_ids)
    item_ids1 = torch.stack(item_ids1)
    item_ids2 = torch.stack(item_ids2)
    labels = torch.stack(labels)


    # Initialize dictionaries to hold batched attributes
    item_attrs1 = {attr: [] for attr in ATTRIBUTES}
    item_attrs2 = {attr: [] for attr in ATTRIBUTES}

    # Collect and stack attribute tensors
    for item_attrs_list, item_attrs in zip([item_attrs1_list, item_attrs2_list], [item_attrs1, item_attrs2]):
        for attr in ATTRIBUTES:
            attr_values = [item_attrs_dict[attr] for item_attrs_dict in item_attrs_list]
            attr_values = torch.stack(attr_values, dim=0)
            item_attrs[attr] = attr_values

    return user_ids, item_ids1, item_ids2, item_attrs1, item_attrs2, labels
