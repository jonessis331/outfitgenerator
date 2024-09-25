# visualization/visualize_embeddings.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from models.embeddings import EmbeddingLayer
from models.siamese_network import SiameseNetwork
from preprocess.encoder import Encoder
from utils.config import *
import numpy as np

def visualize_embeddings():
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

    # Extract item embeddings
    item_embeddings = model.embedding_layer.item_embedding.weight.data.cpu().numpy()
    item_ids = encoder.item_encoder.inverse_transform(np.arange(num_items))

    # PCA Visualization
    pca = PCA(n_components=2)
    item_embeddings_pca = pca.fit_transform(item_embeddings)

    plt.figure(figsize=(8, 6))
    plt.scatter(item_embeddings_pca[:, 0], item_embeddings_pca[:, 1])
    for i, item_id in enumerate(item_ids):
        plt.annotate(item_id, (item_embeddings_pca[i, 0], item_embeddings_pca[i, 1]))
    plt.title('Item Embeddings Visualized with PCA')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.savefig('visualization/item_embeddings_pca.png')
    plt.close()
    print("Item embeddings PCA visualization saved to visualization/item_embeddings_pca.png")

    # t-SNE Visualization
    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    item_embeddings_tsne = tsne.fit_transform(item_embeddings)

    plt.figure(figsize=(8, 6))
    plt.scatter(item_embeddings_tsne[:, 0], item_embeddings_tsne[:, 1])
    for i, item_id in enumerate(item_ids):
        plt.annotate(item_id, (item_embeddings_tsne[i, 0], item_embeddings_tsne[i, 1]))
    plt.title('Item Embeddings Visualized with t-SNE')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.savefig('visualization/item_embeddings_tsne.png')
    plt.close()
    print("Item embeddings t-SNE visualization saved to visualization/item_embeddings_tsne.png")

if __name__ == '__main__':
    visualize_embeddings()
