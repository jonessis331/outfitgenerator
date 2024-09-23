# models/siamese_network.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseNetwork(nn.Module):
    def __init__(self, embedding_layer, embedding_dim):
        super(SiameseNetwork, self).__init__()
        self.embedding_layer = embedding_layer
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, user_ids, item_ids1, item_ids2, item_attributes1, item_attributes2):
        user_emb, item_emb1 = self.embedding_layer(user_ids, item_ids1, item_attributes1)
        _, item_emb2 = self.embedding_layer(user_ids, item_ids2, item_attributes2)

        # Compute element-wise product of item embeddings
        item_pair_emb = item_emb1 * item_emb2

        # Print shapes for debugging
        print(f"user_emb shape: {user_emb.shape}")
        print(f"item_pair_emb shape: {item_pair_emb.shape}")

        combined_emb = torch.cat([item_pair_emb, user_emb], dim=1)

        # Print combined_emb shape
        print(f"combined_emb shape: {combined_emb.shape}")

        # Compute compatibility score
        score = self.fc(combined_emb)
        return score
