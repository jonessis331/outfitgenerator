# main.py

import pandas as pd
from preprocess.data_loader import DataLoader
from preprocess.encoder import Encoder
from utils.config import *
from utils.logger import get_logger

logger = get_logger(__name__)

def main():
    logger.info("Starting data preprocessing...")

    # Initialize DataLoader
    data_loader = DataLoader(DATA_PATH, INTERACTION_DATA_PATH, USER_DATA_PATH)

    # Load data
    logger.info("Loading outfit data...")
    outfit_data = data_loader.load_outfit_data()

    logger.info("Loading interaction data...")
    interaction_df = data_loader.load_interaction_data()

    logger.info("Loading user data...")
    user_df = data_loader.load_user_data()

    # Flatten outfit data into items DataFrame
    logger.info("Flattening outfit data into items DataFrame...")
    item_df = data_loader.flatten_data(outfit_data)

    # Save items DataFrame for future use
    item_df.to_csv(ITEM_DATA_PATH, index=False)
    logger.info(f"Items data saved to {ITEM_DATA_PATH}")

    # Initialize Encoder
    encoder = Encoder()

    # Fit encoders
    logger.info("Fitting user encoder...")
    encoder.fit_user_encoder(interaction_df['user_id'])

    logger.info("Fitting item encoder...")
    encoder.fit_item_encoder(item_df['item_id'])

    logger.info("Fitting attribute encoders...")
    encoder.fit_attribute_encoders(item_df, ATTRIBUTES)

    # Transform interaction data
    logger.info("Encoding interaction data...")
    interaction_df['user_id_encoded'] = encoder.transform_user(interaction_df['user_id'])
    interaction_df['item_ids_encoded'] = interaction_df['item_ids'].apply(
        lambda x: [encoder.transform_item([item_id])[0] for item_id in eval(x)]
    )

    # Save encoded interaction data
    interaction_df.to_csv(ENCODED_INTERACTIONS_PATH, index=False)
    logger.info(f"Encoded interaction data saved to {ENCODED_INTERACTIONS_PATH}")

    # Transform item attributes
    logger.info("Encoding item attributes...")
    item_df_encoded = encoder.transform_attributes(item_df)

    # Save encoded item data
    item_df_encoded.to_csv(ENCODED_ITEMS_PATH, index=False)
    logger.info(f"Encoded item data saved to {ENCODED_ITEMS_PATH}")

    # Save encoders
    encoder.save_encoders(ENCODERS_PATH)
    logger.info(f"Encoders saved to {ENCODERS_PATH}")

    logger.info("Data preprocessing completed successfully.")

if __name__ == "__main__":
    main()
