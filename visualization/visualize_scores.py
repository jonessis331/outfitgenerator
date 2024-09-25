# visualization/visualize_scores.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from models.embeddings import EmbeddingLayer
from models.siamese_network import SiameseNetwork
from preprocess.encoder import Encoder
from utils.config import *

def visualize_scores():
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

    # Get all item combinations
    item_ids = encoder.item_encoder.classes_
    num_items = len(item_ids)
    compatibility_matrix = np.zeros((num_items, num_items))

    # Prepare dummy user ID and attributes
    user_id = torch.tensor([0], dtype=torch.long)
    item_attrs_list = []
    for i in range(num_items):
        item_attrs = {}
        for attr in single_valued_attrs:
            item_attrs[attr] = torch.tensor([0], dtype=torch.long)
        for attr in multi_valued_attrs:
            num_classes = num_attributes[attr]
            item_attrs[attr] = torch.zeros(1, num_classes)
        item_attrs_list.append(item_attrs)

    with torch.no_grad():
        for i in range(num_items):
            item_id1 = torch.tensor([i], dtype=torch.long)
            item_attrs1 = item_attrs_list[i]
            for j in range(num_items):
                item_id2 = torch.tensor([j], dtype=torch.long)
                item_attrs2 = item_attrs_list[j]
                score = model(
                    user_id,
                    item_id1,
                    item_id2,
                    item_attrs1,
                    item_attrs2
                )
                compatibility_matrix[i, j] = score.item()

    # Plot Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(compatibility_matrix, xticklabels=item_ids, yticklabels=item_ids, cmap='viridis')
    plt.title('Compatibility Scores between Items')
    plt.xlabel('Item ID')
    plt.ylabel('Item ID')
    plt.savefig('visualization/compatibility_heatmap.png')
    plt.close()
    print("Compatibility heatmap saved to visualization/compatibility_heatmap.png")

if __name__ == '__main__':
    visualize_scores()
