Here's an updated README file reflecting the use of Docker for the entire environment setup and build process, including instructions for the Makefile.

---

# TrellisLaw Document Classification

## Project Overview

This project aims to develop a robust document classification system using machine learning techniques. The system reads and processes text documents from various categories, trains multiple classification models, evaluates their performance, and deploys the best model as an API. The project leverages technologies such as Scikit-Learn, SpaCy, FastAPI, and Docker to ensure scalability, efficiency, and ease of deployment.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Folder Structure](#folder-structure)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Data Processing](#data-processing)
6. [Model Training and Evaluation](#model-training-and-evaluation)
7. [API Deployment](#api-deployment)
8. [Contributing](#contributing)
9. [License](#license)

## Folder Structure

```
trellis_law/
│
├── data/
│   ├── 01_raw/
│   │   └── trellis_assesment_data/
│   │       ├── food/
│   │       ├── sport/
│   │       ├── space/
│   │       ├── medical/
│   │       ├── business/
│   │       ├── politics/
│   │       ├── graphics/
│   │       ├── historical/
│   │       ├── technologie/
│   │       └── entertainment/
│   ├── 02_intermediate/
│   ├── 03_primary/
│   ├── 04_feature/
│   ├── 05_model_input/
│   ├── 06_models/
│   ├── 07_model_output/
│   └── 08_reporting/
│
├── notebooks/
│   ├── eda.ipynb
│   └── eda2.ipynb
│
├── src/
│   ├── trellis_law/
│   │   ├── __init__.py
│   │   ├── nodes/
│   │   │   ├── data_engineering.py
│   │   │   ├── model_training.py
│   │   │   └── utils.py
│   │   ├── pipelines/
│   │   └── api.py
│   ├── tests/
│   ├── pipeline_registry.py
│   └── settings.py
│
├── conf/
│   ├── base/
│   │   ├── catalog.yml
│   │   ├── logging.yml
│   │   ├── parameters.yml
│   │   └── spark.yml
│   ├── local/
│   │   └── credentials.yml
│
├── docker-compose.yml
├── Dockerfile
├── Makefile
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/trellis-law.git
   cd trellis-law
   ```

2. **Build and Run Docker Container:**
   ```bash
   make build
   make run
   ```

## Usage

### Data Processing

1. **Run Data Preprocessing:**
   ```bash
   make preprocess_data
   ```

### Model Training and Evaluation

1. **Train Models:**
   ```bash
   make train_models
   ```

### API Deployment

1. **Build and Run Docker Container:**
   ```bash
   make build
   make run
   ```

2. **Stop Docker Container:**
   ```bash
   make stop
   ```

3. **Test API:**
   The API can be accessed at `http://localhost:8000/docs` for Swagger UI.

## Data Processing

The `data_engineering.py` script handles data reading and preprocessing:
- Reads text files from specified directories.
- Preprocesses the text by removing stop words, punctuation, and lowercasing.
- Combines and returns the processed data as a pandas DataFrame.

## Model Training and Evaluation

The `model_training.py` script handles model training and evaluation:
- Splits the data into training and testing sets.
- Trains multiple models including Naive Bayes, Logistic Regression, SVM, LSTM, and XGBoost.
- Evaluates models using metrics like accuracy, precision, recall, and F1-score.
- Saves the best-performing model.

## API Deployment

The `api.py` script sets up a FastAPI server:
- Loads the best-trained model.
- Provides an endpoint to classify document text.
- Uses a threshold-based classification to determine if a document should be categorized as 'Other'.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any changes or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

Feel free to customize any part of this README to better fit your project specifics and preferences.