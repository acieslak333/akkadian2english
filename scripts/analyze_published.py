import pandas as pd
import os

def analyze():
    file_path = 'data/published_texts.csv'
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return
    
    # Use low_memory=False to avoid DtypeWarning
    df = pd.read_csv(file_path, low_memory=False)
    print(f"Total rows in published_texts.csv: {len(df)}")
    
    # Filter for rows that have both content and translation
    valid_df = df[df['transliteration'].notna() & df['AICC_translation'].notna()]
    print(f"Rows with both content and translation: {len(valid_df)}")
    
    # Check for oare_id overlap with train.csv
    train_df = pd.read_csv('data/train.csv')
    train_ids = set(train_df['oare_id'].unique())
    published_ids = set(df['oare_id'].unique())
    
    overlap = train_ids.intersection(published_ids)
    print(f"Number of OARE IDs in train.csv: {len(train_ids)}")
    print(f"Number of OARE IDs in published_texts.csv: {len(published_ids)}")
    print(f"Overlap OARE IDs: {len(overlap)}")
    
    # Sample some translations
    if len(valid_df) > 0:
        print("\nSample segment (published_texts):")
        print(valid_df[['transliteration', 'AICC_translation']].head(3))

if __name__ == "__main__":
    analyze()
