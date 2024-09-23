# preprocess/encoder.py

from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
import pandas as pd
import joblib

class Encoder:
    def __init__(self):
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.attribute_encoders = {}
        self.multi_label_encoders = {}

    def fit_user_encoder(self, user_ids):
        self.user_encoder.fit(user_ids)

    def fit_item_encoder(self, item_ids):
        self.item_encoder.fit(item_ids)

    def fit_attribute_encoders(self, df, attributes):
        for attr in attributes:
            if df[attr].dtype == 'object':
                # Check if the column contains lists
                if df[attr].apply(lambda x: isinstance(x, list)).any():
                    # Multi-valued attribute
                    mlb = MultiLabelBinarizer()
                    # Flatten the lists and fit the MultiLabelBinarizer
                    mlb.fit(df[attr].apply(lambda x: x if isinstance(x, list) else [x]))
                    self.multi_label_encoders[attr] = mlb
                else:
                    # Single-valued attribute
                    unique_values = df[attr].unique()
                    print(f"Fitting encoder for {attr} with unique values: {unique_values}")
                    le = LabelEncoder()
                    le.fit(df[attr].astype(str))
                    self.attribute_encoders[attr] = le

    def transform_user(self, user_ids):
        return self.user_encoder.transform(user_ids)

    def transform_item(self, item_ids):
        return self.item_encoder.transform(item_ids)

    def transform_attributes(self, df):
        df_encoded = df.copy()
        for attr in self.attribute_encoders:
            known_classes = set(self.attribute_encoders[attr].classes_)
            df[attr] = df[attr].apply(lambda x: x if x in known_classes else None)
            df_encoded[attr] = self.attribute_encoders[attr].transform(df[attr].astype(str))
        for attr in self.multi_label_encoders:
            mlb = self.multi_label_encoders[attr]
            df_encoded[attr] = df[attr].apply(lambda x: mlb.transform([x])[0])
        return df_encoded



    def save_encoders(self, path):
        joblib.dump({
            'user_encoder': self.user_encoder,
            'item_encoder': self.item_encoder,
            'attribute_encoders': self.attribute_encoders,
            'multi_label_encoders': self.multi_label_encoders
        }, path)

    def load_encoders(self, path):
        data = joblib.load(path)
        self.user_encoder = data['user_encoder']
        self.item_encoder = data['item_encoder']
        self.attribute_encoders = data['attribute_encoders']
        self.multi_label_encoders = data['multi_label_encoders']
     