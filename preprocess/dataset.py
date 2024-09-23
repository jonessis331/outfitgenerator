# preprocess/dataset.py

import torch
from torch.utils.data import Dataset
import pandas as pd
import random
from utils.config import ATTRIBUTES
import numpy as np

class OutfitCompatibilityDataset(Dataset):
    def __init__(self, interaction_df, item_df, user_encoder, item_encoder, full_encoder, negative_sample_ratio=1):
        self.interaction_df = interaction_df
        self.item_df = item_df
        self.user_encoder = user_encoder
        self.item_encoder = item_encoder
        self.encoder = full_encoder
        self.negative_sample_ratio = negative_sample_ratio
        self.prepare_samples()

    def prepare_samples(self):
        positive_samples = []
        negative_samples = []
        
        # For positive samples
        for idx, row in self.interaction_df[self.interaction_df['interaction'] == 1].iterrows():
            item_ids = eval(row['item_ids'])
            user_id = row['user_id']
            # Generate all possible pairs within the outfit
            for i in range(len(item_ids)):
                for j in range(i + 1, len(item_ids)):
                    positive_samples.append({
                        'user_id': user_id,
                        'item_id1': item_ids[i],
                        'item_id2': item_ids[j],
                        'label': 1
                    })
        
        # For negative samples
        for idx, row in self.interaction_df[self.interaction_df['interaction'] == 0].iterrows():
            item_ids = eval(row['item_ids'])
            user_id = row['user_id']
            # Randomly pair items (ensure they are not in the same outfit)
            # You may need to implement logic to select truly negative pairs
            for i in range(len(item_ids)):
                negative_samples.append({
                    'user_id': user_id,
                    'item_id1': item_ids[i],
                    'item_id2': random.choice(self.item_df['item_id'].unique()),
                    'label': 0
                })
        
        # Combine and shuffle samples
        self.samples = positive_samples + negative_samples
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        user_id = sample['user_id']
        item_id1 = sample['item_id1']
        item_id2 = sample['item_id2']
        label = sample['label']

        # Encode user and items
        user_encoded = self.encoder.transform_user([user_id])[0]
        item_encoded1 = self.encoder.transform_item([item_id1])[0]
        item_encoded2 = self.encoder.transform_item([item_id2])[0]

        # Correct tensor creation
        user_tensor = torch.tensor(user_encoded, dtype=torch.long)
        item_tensor1 = torch.tensor(item_encoded1, dtype=torch.long)
        item_tensor2 = torch.tensor(item_encoded2, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.float)


        # Retrieve attributes
        item_attrs1 = self.get_item_attributes(item_id1)
        item_attrs2 = self.get_item_attributes(item_id2)

        return user_tensor, item_tensor1, item_tensor2, item_attrs1, item_attrs2, label_tensor

    def get_item_attributes(self, item_id):
        item_row = self.item_df[self.item_df['item_id'] == item_id].iloc[0]
        item_attrs = {}
        for attr in ATTRIBUTES:
            attr_value = item_row[attr]
            if attr in self.encoder.attribute_encoders:
                # Single-valued attribute
                attr_encoded = self.encoder.attribute_encoders[attr].transform([str(attr_value)])[0]
                item_attrs[attr] = torch.tensor(attr_encoded, dtype=torch.long)
            elif attr in self.encoder.multi_label_encoders:
                # Multi-valued attribute
                attr_values = attr_value if isinstance(attr_value, list) else [attr_value]
                attr_encoded = self.encoder.multi_label_encoders[attr].transform([attr_values])[0]
                item_attrs[attr] = torch.tensor(attr_encoded, dtype=torch.float)
        return item_attrs

