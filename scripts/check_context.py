import pandas as pd

def check():
    train_df = pd.read_csv('data/train.csv')
    pub_df = pd.read_csv('data/published_texts.csv', low_memory=False)
    
    # Merge on oare_id
    merged = pd.merge(train_df, pub_df[['oare_id', 'transliteration']], on='oare_id', suffixes=('_train', '_pub'))
    
    # Compare lengths
    merged['len_train'] = merged['transliteration_train'].str.len()
    merged['len_pub'] = merged['transliteration_pub'].str.len()
    
    print("Length Comparison (Train vs Published):")
    print((merged['len_pub'] - merged['len_train']).describe())
    
    # Find cases where pub is significantly longer
    longer = merged[merged['len_pub'] > merged['len_train'] + 10]
    print(f"\nNumber of tablets where Published is >10 chars longer: {len(longer)}")
    
    if len(longer) > 0:
        print("\nSample of longer context:")
        idx = longer.index[0]
        print(f"ID: {longer.loc[idx, 'oare_id']}")
        print(f"Train: {longer.loc[idx, 'transliteration_train'][:100]}...")
        print(f"Pub:   {longer.loc[idx, 'transliteration_pub'][:100]}...")

if __name__ == "__main__":
    check()
