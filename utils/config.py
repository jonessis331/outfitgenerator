# utils/config.py

DATA_PATH = 'data/raw/outfits.json'
PROCESSED_DATA_PATH = 'data/processed/outfits_processed.csv'
ENCODERS_PATH = 'models/encoders.joblib'
SCALER_PATH = 'models/scaler.joblib'
MODEL_PATH = 'models/kmeans_model.joblib'

DATA_PATH = 'data/raw/outfits.json'
INTERACTION_DATA_PATH = 'data/raw/interactions.csv'
USER_DATA_PATH = 'data/raw/users.csv'
ITEM_DATA_PATH = 'data/processed/items.csv'
ENCODERS_PATH = 'models/encoders.joblib'
MODEL_SAVE_PATH = 'models/siamese_model.pth'
ATTRIBUTES = ['colors', 'materials', 'pattern', 'fit', 'category', 'item_type']
NUM_EPOCHS = 10

DATA_DIR = 'data'
RAW_DATA_DIR = f'{DATA_DIR}/raw'
PROCESSED_DATA_DIR = f'{DATA_DIR}/processed'

OUTFIT_DATA_PATH = f'{RAW_DATA_DIR}/outfits.json'
INTERACTION_DATA_PATH = f'{RAW_DATA_DIR}/interactions.csv'
USER_DATA_PATH = f'{RAW_DATA_DIR}/users.csv'

ITEM_DATA_PATH = f'{PROCESSED_DATA_DIR}/items.csv'
ENCODED_ITEMS_PATH = f'{PROCESSED_DATA_DIR}/items_encoded.csv'
ENCODED_INTERACTIONS_PATH = f'{PROCESSED_DATA_DIR}/interactions_encoded.csv'

ENCODERS_PATH = 'models/encoders.joblib'
MODEL_SAVE_PATH = 'models/siamese_model.pth'

#ATTRIBUTES = ['category', 'colors', 'materials', 'pattern', 'fit', 'item_type']
NUM_EPOCHS = 10
BATCH_SIZE = 32
EMBEDDING_DIM = 64
LEARNING_RATE = 0.001
TOP_K = 10
