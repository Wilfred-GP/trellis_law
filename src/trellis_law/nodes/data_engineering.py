import os
import pandas as pd
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def read_text_files_from_directory(directory):
    texts = []
    files = os.listdir(directory)
    for file in files:
        if file.endswith('.txt'):  # Ensure only .txt files are processed
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path):
                with open(file_path, 'r') as f:
                    texts.append(f.read())
    print(f"Read {len(texts)} files from {directory}")
    return pd.DataFrame({"content": texts})

def preprocess_text(text):
    text = re.sub(r'\W+', ' ', text)
    words = text.lower().split()
    words = [word for word in words if word not in ENGLISH_STOP_WORDS]
    return ' '.join(words)

def preprocess_data() -> pd.DataFrame:
    base_path = "./data/01_raw/trellis_assesment_data"
    print(f"Base path: {base_path}")
    
    # Read and concatenate text files from each category
    categories = ["food", "sport", "space", "medical", "business", "politics", "graphics", "historical", "technologie", "entertainment"]
    
    dfs = []
    for category in categories:
        directory = os.path.join(base_path, category)
        print(f"Reading directory: {directory}")
        if os.path.exists(directory) and os.path.isdir(directory):
            df = read_text_files_from_directory(directory)
            df["category"] = category
            print(f"{category} data: {df.shape[0]} records")
            dfs.append(df)
        else:
            print(f"Directory {directory} does not exist or is not a directory.")
    
    if dfs:
        all_data = pd.concat(dfs, ignore_index=True)
        print(f"All data combined: {all_data.shape[0]} records")

        # Preprocess text data
        all_data["content"] = all_data["content"].apply(preprocess_text)
        print("Data preprocessing completed")

        return all_data
    else:
        print("No data to process.")
        return pd.DataFrame()  # Return an empty DataFrame if no data is found
