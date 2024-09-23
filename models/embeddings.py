# models/embeddings.py

import torch
import torch.nn as nn

class EmbeddingLayer(nn.Module):
    def __init__(self, num_users, num_items, num_attributes, embedding_dim, single_valued_attrs, multi_valued_attrs):
        super(EmbeddingLayer, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.attribute_embeddings = nn.ModuleDict({
            attr_name: nn.Embedding(num_categories, embedding_dim)
            for attr_name, num_categories in num_attributes.items()
        })
        self.single_valued_attrs = single_valued_attrs
        self.multi_valued_attrs = multi_valued_attrs


    def forward(self, user_ids, item_ids, item_attributes):
        print(f"user_ids shape before embedding: {user_ids.shape}")
        print(f"item_ids shape before embedding: {item_ids.shape}")

        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)

        print(f"user_emb shape after embedding: {user_emb.shape}")
        print(f"item_emb shape after embedding: {item_emb.shape}")

       
        attr_embs = []
        for attr_name, attr_values in item_attributes.items():
            if attr_name in self.single_valued_attrs:
                # Single-valued attribute
                attr_emb = self.attribute_embeddings[attr_name](attr_values)
                attr_embs.append(attr_emb)
            elif attr_name in self.multi_valued_attrs:
                # Multi-valued attribute
                # attr_values: [batch_size, num_classes]
                embedding_matrix = self.attribute_embeddings[attr_name].weight  # [num_classes, embedding_dim]
                attr_emb = torch.matmul(attr_values.float(), embedding_matrix)  # [batch_size, embedding_dim]
                attr_embs.append(attr_emb)
            else:
                continue  # Skip unknown attributes
        # Sum attribute embeddings
        item_attr_emb = torch.stack(attr_embs, dim=0).sum(dim=0)
        item_total_emb = item_emb + item_attr_emb
        return user_emb, item_total_emb

