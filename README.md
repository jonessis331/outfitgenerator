# Outfit Recommendation System

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Features](#features)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Model Training](#model-training)
- [API Usage](#api-usage)
- [Configuration](#configuration)
- [Logging](#logging)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The Outfit Recommendation System is a machine learning project designed to provide personalized outfit recommendations to users based on their preferences and past interactions. It leverages a Siamese neural network to learn compatibility between items and users, enabling the system to suggest items that complement each other and align with user tastes.

## Project Structure

├── data │ ├── raw │ │ ├── outfits.json │ │ ├── interactions.csv │ │ └── users.csv │ └── processed │ ├── items.csv │ ├── items_encoded.csv │ └── interactions_encoded.csv ├── models │ ├── base_model.py │ ├── embeddings.py │ ├── siamese_network.py │ └── encoders.joblib ├── preprocess │ ├── data_loader.py │ ├── dataset.py │ ├── encoder.py │ └── scaler.py ├── utils │ ├── config.py │ ├── helper_functions.py │ └── logger.py ├── main.py ├── train.py ├── api │ └── app.py ├── setup.py ├── requirements.txt └── README.md

## Features

- Data Preprocessing: Load and preprocess data from JSON and CSV files.
- Encoding: Transform categorical attributes into numerical representations.
- Siamese Neural Network: Learn compatibility between item pairs in the context of user preferences.
- Model Training: Train the model using encoded data.
- API: Serve recommendations via a RESTful API.
- Logging: Comprehensive logging for monitoring and debugging.

## Installation

### Prerequisites

- Python 3.6 or higher
- pip (Python package installer)

### Clone the Repository

```bash
git clone https://github.com/your_username/outfit-recommendation-system.git
cd outfit-recommendation-system
Install Dependencies
You can install the required packages using pip:

bash
Copy code
pip install -r requirements.txt
Alternatively, if you have setup.py, you can install the package:

bash

python setup.py install
Data Preparation

Data Files
Place your data files in the data/raw/ directory:

outfits.json: Contains outfit and item details.
interactions.csv: User interactions with outfits (e.g., likes, dislikes).
users.csv: User profile information.
Directory Structure
kotlin
Copy code
data/
├── raw/
│   ├── outfits.json
│   ├── interactions.csv
│   └── users.csv
└── processed/
Running Data Preprocessing
Execute the main.py script to preprocess the data:

bash
Copy code
python main.py
This script will:

Load and flatten the outfit data into items.
Fit encoders for users, items, and attributes.
Encode interaction data.
Save encoded data to data/processed/.
Save encoders to models/encoders.joblib.
Model Training

Training the Model
Run the train.py script to train the Siamese neural network:

bash
Copy code
python train.py
This script will:

Load and encode data.
Prepare the dataset and data loader.
Initialize the Siamese network model.
Train the model over the specified number of epochs.
Save the trained model to models/siamese_model.pth.
Configuration

You can adjust training parameters and paths in utils/config.py:

python
Copy code
NUM_EPOCHS = 10
BATCH_SIZE = 32
EMBEDDING_DIM = 64
LEARNING_RATE = 0.001
API Usage

Starting the API
After training the model, you can serve recommendations via the API:

bash
Copy code
python -m api.app
This will start the Flask API server on http://localhost:5000.

API Endpoints
POST /recommend

Get outfit recommendations for a user based on selected items.

Request Body:

json
Copy code
{
  "user_id": "u1",
  "item_ids": ["i1", "i2"]
}
Response:

json
Copy code
{
  "recommendations": [
    {
      "item_id": "i16",
      "compatibility_score": 1.4015,
      "category": "bottom",
      "colors": "['blue']",
      "materials": "['denim']",
      "pattern": "solid",
      "fit": "regular",
      "item_type": "i16",
      "outfit_id": "o8"
    }
    // ... more recommendations
  ]
}
Example Request
Using curl:

bash
Copy code
curl -X POST -H "Content-Type: application/json" -d '{"user_id": "u1", "item_ids": ["i1", "i2"]}' http://localhost:5000/recommend
Configuration

All configurable parameters are stored in utils/config.py:

python
Copy code
# Data paths
DATA_DIR = 'data'
RAW_DATA_DIR = f'{DATA_DIR}/raw'
PROCESSED_DATA_DIR = f'{DATA_DIR}/processed'

OUTFIT_DATA_PATH = f'{RAW_DATA_DIR}/outfits.json'
INTERACTION_DATA_PATH = f'{RAW_DATA_DIR}/interactions.csv'
USER_DATA_PATH = f'{RAW_DATA_DIR}/users.csv'

ITEM_DATA_PATH = f'{PROCESSED_DATA_DIR}/items.csv'
ENCODED_ITEMS_PATH = f'{PROCESSED_DATA_DIR}/items_encoded.csv'
ENCODED_INTERACTIONS_PATH = f'{PROCESSED_DATA_DIR}/interactions_encoded.csv'

ENCODERS_PATH = 'models/encoders.joblib'
MODEL_SAVE_PATH = 'models/siamese_model.pth'

# Attributes to encode
ATTRIBUTES = ['category', 'colors', 'materials', 'pattern', 'fit', 'item_type']

# Training parameters
NUM_EPOCHS = 10
BATCH_SIZE = 32
EMBEDDING_DIM = 64
LEARNING_RATE = 0.001
TOP_K = 10
Logging

Logging is set up using Python's logging module, configured in utils/logger.py. Logs provide detailed information about the execution flow, which is helpful for debugging.

Example log output:
markdown
Copy code
2024-09-19 11:26:03,760 - __main__ - INFO - Starting training process...
2024-09-19 11:26:03,762 - __main__ - INFO - Flattening outfit data...
2024-09-19 11:26:03,762 - __main__ - INFO - Initializing and fitting encoders...
...
Troubleshooting

Common Issues
ValueError: y contains previously unseen labels

Cause:
This error occurs when the LabelEncoder encounters a value during transformation that it wasn't fitted on.

Solution:
Ensure all possible values are included when fitting the encoder. Convert attribute values to consistent data types (e.g., strings) during both fitting and transformation.

UserWarning: unknown class(es) [0, 1] will be ignored

Cause:
This warning indicates that the LabelEncoder encountered unknown classes.

Solution:
Check that multi-valued attributes are being processed with MultiLabelBinarizer. Ensure that the data types are consistent and that all classes are known to the encoder.

Tips
Data Consistency: Ensure your data is clean and consistent. Inconsistent data types or unexpected values can cause encoding errors.
Print Statements: Use print statements or logging to debug values during encoding and transformation.
Encoder Classes: After fitting encoders, print out the classes to verify all expected values are included.
Contributing

Contributions are welcome! Please follow these steps:

Fork the repository.
Create a new branch for your feature or bugfix.
Make your changes and commit them with clear messages.
Submit a pull request detailing your changes.
License

This project is licensed under the MIT License. See the LICENSE file for details.

Contact

For any questions or issues, please open an issue on the repository or contact the project maintainer.
```
