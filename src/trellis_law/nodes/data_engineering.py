import os
import pandas as pd
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def read_text_files_from_directory(directory: str) -> pd.DataFrame:
    """
    Reads all text files from a given directory and returns their content in a DataFrame.

    Args:
        directory: The path to the directory containing text files.

    Returns:
        A DataFrame with a single column 'content' containing the text from each file.
    """
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

def preprocess_text(text: str) -> str:
    """
    Preprocesses a given text by removing non-alphanumeric characters, converting to lowercase,
    and removing English stop words.

    Args:
        text: The text to preprocess.

    Returns:
        The preprocessed text.
    """
    text = re.sub(r'\W+', ' ', text)
    words = text.lower().split()
    words = [word for word in words if word not in ENGLISH_STOP_WORDS]
    return ' '.join(words)

def preprocess_data(base_path: str = "./data/01_raw/trellis_assesment_data") -> pd.DataFrame:
    """
    Reads and preprocesses text files from multiple categories located in subdirectories under the base path.

    Args:
        base_path: The base directory containing category subdirectories.

    Returns:
        A DataFrame with preprocessed text data and their associated categories.
    """
    print(f"Base path: {base_path}")

    # Define categories
    categories = [
        "food", "sport", "space", "medical", "business",
        "politics", "graphics", "historical", "technologie", "entertainment"
    ]

    # Read and concatenate text files from each category
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
