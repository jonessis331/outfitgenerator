from flask import Flask, request, jsonify
import torch
from models.embeddings import EmbeddingLayer
from models.siamese_network import SiameseNetwork
from preprocess.encoder import Encoder
from utils.config import *
import pandas as pd

app = Flask(__name__)

# Load model and encoders
encoder = Encoder()
encoder.load_encoders(ENCODERS_PATH)

num_users = len(encoder.user_encoder.classes_)
num_items = len(encoder.item_encoder.classes_)

# Determine single-valued and multi-valued attributes
single_valued_attrs = list(encoder.attribute_encoders.keys())
multi_valued_attrs = list(encoder.multi_label_encoders.keys())

# Build num_attributes dictionary
num_attributes = {}
for attr in single_valued_attrs:
    num_categories = len(encoder.attribute_encoders[attr].classes_)
    num_attributes[attr] = num_categories
for attr in multi_valued_attrs:
    num_categories = len(encoder.multi_label_encoders[attr].classes_)
    num_attributes[attr] = num_categories

embedding_dim = EMBEDDING_DIM

embedding_layer = EmbeddingLayer(
    num_users, num_items, num_attributes, embedding_dim, single_valued_attrs, multi_valued_attrs
)

model = SiameseNetwork(embedding_layer, embedding_dim)
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.eval()

# Load item data
item_df = pd.read_csv(ITEM_DATA_PATH)

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    user_id = data['user_id']
    item_ids = data['item_ids']  # Items the user wants to include

    # Encode user and items
    user_encoded = torch.tensor(encoder.transform_user([user_id]), dtype=torch.long)
    item_encoded = torch.tensor(encoder.transform_item(item_ids), dtype=torch.long)

    # Prepare attributes for the items
    item_attrs = {}
    # For single-valued attributes
    for attr in single_valued_attrs:
        attr_values = item_df[item_df['item_id'].isin(item_ids)][attr].values
        attr_encoded = encoder.attribute_encoders[attr].transform(attr_values)
        item_attrs[attr] = torch.tensor(attr_encoded, dtype=torch.long)
    # For multi-valued attributes
    for attr in multi_valued_attrs:
        attr_values = item_df[item_df['item_id'].isin(item_ids)][attr].values
        attr_values = [vals if isinstance(vals, list) else [vals] for vals in attr_values]
        attr_encoded = encoder.multi_label_encoders[attr].transform(attr_values)
        item_attrs[attr] = torch.tensor(attr_encoded, dtype=torch.float)

    # Find compatible items
    # Compute compatibility scores with other items in the inventory
    candidate_items = item_df[~item_df['item_id'].isin(item_ids)]['item_id'].values
    candidate_encoded = torch.tensor(encoder.transform_item(candidate_items), dtype=torch.long)

    # Prepare candidate attributes
    candidate_attrs = {}
    # For single-valued attributes
    for attr in single_valued_attrs:
        attr_values = item_df[item_df['item_id'].isin(candidate_items)][attr].values
        attr_encoded = encoder.attribute_encoders[attr].transform(attr_values)
        candidate_attrs[attr] = torch.tensor(attr_encoded, dtype=torch.long)
    # For multi-valued attributes
    for attr in multi_valued_attrs:
        attr_values = item_df[item_df['item_id'].isin(candidate_items)][attr].values
        attr_values = [vals if isinstance(vals, list) else [vals] for vals in attr_values]
        attr_encoded = encoder.multi_label_encoders[attr].transform(attr_values)
        candidate_attrs[attr] = torch.tensor(attr_encoded, dtype=torch.float)

    # Compute scores
    with torch.no_grad():
        scores = []
        for i, candidate_item in enumerate(candidate_encoded):
            # Prepare attributes for candidate item
            candidate_attr = {}
            for attr in ATTRIBUTES:
                candidate_attr[attr] = candidate_attrs[attr][i].unsqueeze(0)

            # Compute score for each item in item_encoded
            item_scores = []
            for j, item_id in enumerate(item_encoded):
                item_attr = {}
                for attr in ATTRIBUTES:
                    item_attr[attr] = item_attrs[attr][j].unsqueeze(0)

                score = model(
                    user_encoded,
                    item_id.unsqueeze(0),
                    candidate_item.unsqueeze(0),
                    item_attr,
                    candidate_attr
                )
                item_scores.append(score.item())
            # Aggregate scores
            avg_score = sum(item_scores) / len(item_scores)
            scores.append((candidate_items[i], avg_score))

   # Sort candidates by score
    scores.sort(key=lambda x: x[1], reverse=True)
    top_candidates = scores[:TOP_K]

    # Prepare recommendations with scores
    recommendations = []
    for item_id, score in top_candidates:
        item_info = item_df[item_df['item_id'] == item_id].to_dict(orient='records')[0]
        item_info['compatibility_score'] = score
        recommendations.append(item_info)

    return jsonify({'recommendations': recommendations})

if __name__ == '__main__':
    app.run(debug=True)
