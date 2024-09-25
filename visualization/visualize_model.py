# visualization/visualize_model.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torchviz import make_dot
from models.embeddings import EmbeddingLayer
from models.siamese_network import SiameseNetwork
from preprocess.encoder import Encoder
from utils.config import *


def visualize_model():
    # Load encoders
    encoder = Encoder()
    encoder.load_encoders(ENCODERS_PATH)

    # Model parameters
    num_users = len(encoder.user_encoder.classes_)
    num_items = len(encoder.item_encoder.classes_)
    num_attributes = {}
    single_valued_attrs = list(encoder.attribute_encoders.keys())
    multi_valued_attrs = list(encoder.multi_label_encoders.keys())
    for attr in single_valued_attrs:
        num_attributes[attr] = len(encoder.attribute_encoders[attr].classes_)
    for attr in multi_valued_attrs:
        num_attributes[attr] = len(encoder.multi_label_encoders[attr].classes_)
    embedding_dim = EMBEDDING_DIM

    # Initialize model
    embedding_layer = EmbeddingLayer(
        num_users, num_items, num_attributes, embedding_dim, single_valued_attrs, multi_valued_attrs
    )
    model = SiameseNetwork(embedding_layer, embedding_dim)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()

    # Sample input data
    user_ids = torch.tensor([0], dtype=torch.long)
    item_ids1 = torch.tensor([0], dtype=torch.long)
    item_ids2 = torch.tensor([1], dtype=torch.long)
    item_attrs1 = {}
    item_attrs2 = {}
    # Prepare dummy attributes
    for attr in single_valued_attrs:
        item_attrs1[attr] = torch.tensor([0], dtype=torch.long)
        item_attrs2[attr] = torch.tensor([0], dtype=torch.long)
    for attr in multi_valued_attrs:
        num_classes = num_attributes[attr]
        item_attrs1[attr] = torch.zeros(1, num_classes)
        item_attrs2[attr] = torch.zeros(1, num_classes)

    # Forward pass
    output = model(user_ids, item_ids1, item_ids2, item_attrs1, item_attrs2)

    # Visualize
    dot = make_dot(output, params=dict(model.named_parameters()))
    dot.format = 'png'
    output_path = os.path.join('visualization', 'siamese_network')
    dot.render(output_path)
    print(f"Model architecture visualized and saved to {output_path}.png")

if __name__ == '__main__':
    visualize_model()
