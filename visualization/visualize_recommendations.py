# visualization/visualize_recommendations.py

import torch
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
from api.app import recommend
from utils.config import ITEM_DATA_PATH

def visualize_recommendations():
    # Sample input
    data = {
        'user_id': 'u1',
        'item_ids': ['i1', 'i2']  # User's selected items
    }

    # Call the recommend function directly (modify the function to accept data as a parameter)
    recommendations = recommend(data)

    # Extract recommended item IDs
    recommended_items = recommendations.get_json()['recommendations']
    recommended_item_ids = [item['item_id'] for item in recommended_items]

    # Display images
    plt.figure(figsize=(15, 5))
    for idx, item_id in enumerate(recommended_item_ids):
        plt.subplot(1, len(recommended_item_ids), idx + 1)
        image_path = os.path.join('path_to_images', f'{item_id}.jpg')
        if os.path.exists(image_path):
            img = Image.open(image_path)
            plt.imshow(img)
        else:
            plt.text(0.5, 0.5, f'No Image\n{item_id}', horizontalalignment='center', verticalalignment='center')
        plt.title(f'Item ID: {item_id}')
        plt.axis('off')
    plt.savefig('visualization/sample_recommendations.png')
    plt.close()
    print("Sample recommendations visualization saved to visualization/sample_recommendations.png")

if __name__ == '__main__':
    visualize_recommendations()
