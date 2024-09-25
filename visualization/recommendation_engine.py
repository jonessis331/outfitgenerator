# recommendation_engine.py

import torch
from models.embeddings import EmbeddingLayer
from models.siamese_network import SiameseNetwork
from preprocess.encoder import Encoder
from utils.config import *
import pandas as pd

# Load model and encoders
encoder = Encoder()
encoder.load_encoders(ENCODERS_PATH)

num_users = len(encoder.user_encoder.classes_)
num_items = len(encoder.item_encoder.classes_)
single_valued_attrs = list(encoder.attribute_encoders.keys())
multi_valued_attrs = list(encoder.multi_label_encoders.keys())
num_attributes = {}
for attr in single_valued_attrs:
    num_attributes[attr] = len(encoder.attribute_encoders[attr].classes_)
for attr in multi_valued_attrs:
    num_attributes[attr] = len(encoder.multi_label_encoders[attr].classes_)
embedding_dim = EMBEDDING_DIM

embedding_layer = EmbeddingLayer(
    num_users, num_items, num_attributes, embedding_dim, single_valued_attrs, multi_valued_attrs
)

model = SiameseNetwork(embedding_layer, embedding_dim)
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.eval()

# Load item data
item_df = pd.read_csv(ITEM_DATA_PATH)

def generate_recommendations(data):
    # ... (same as before)
    return {'recommendations': recommendations}
