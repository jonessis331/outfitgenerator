# preprocess/data_loader.py
import json
import pandas as pd

class DataLoader:
    def __init__(self, outfit_data_path, interaction_data_path, user_data_path=None):
        self.outfit_data_path = outfit_data_path
        self.interaction_data_path = interaction_data_path
        self.user_data_path = user_data_path

    def load_outfit_data(self):
        with open(self.outfit_data_path, 'r') as f:
            outfit_data = json.load(f) or pd.read_csv(f)

        return outfit_data

    def load_interaction_data(self):
        interaction_df = pd.read_csv(self.interaction_data_path)
        return interaction_df

    def load_user_data(self):
        if self.user_data_path:
            user_df = pd.read_csv(self.user_data_path)
            return user_df
        else:
            return None

    def flatten_data(self, data):
        # Flatten nested JSON into a DataFrame
        rows = []
        for outfit in data:
            for item in outfit['items']:
                row = {
                    'outfit_id': outfit['outfit_id'],
                    'item_id': item['item_id'],
                    'category': item['category'],
                    'colors': item['colors'],
                    'materials': item['materials'],
                    'pattern': item['pattern'],
                    'fit': item['fit'],
                    'item_type': item['item_id']
                }
                rows.append(row)
        df = pd.DataFrame(rows)
        return df
