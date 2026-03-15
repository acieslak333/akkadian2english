from transformers import AutoTokenizer
import pandas as pd
import numpy as np

def analyze_token_lengths():
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    df = pd.read_csv('data/processed/train_cleaned_norm.csv')
    
    # Fill NaN
    df['transliteration'] = df['transliteration'].fillna('')
    df['translation'] = df['translation'].fillna('')
    
    print("Tokenizing transliterations...")
    trans_tokens = tokenizer(df['transliteration'].tolist(), add_special_tokens=True)['input_ids']
    trans_lens = [len(x) for x in trans_tokens]
    
    print("Tokenizing translations...")
    label_tokens = tokenizer(df['translation'].tolist(), add_special_tokens=True)['input_ids']
    label_lens = [len(x) for x in label_tokens]
    
    stats_df = pd.DataFrame({
        'transliteration_tokens': trans_lens,
        'translation_tokens': label_lens
    })
    
    print("\nToken Length Statistics:")
    print(stats_df.describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]))
    
    for length in [128, 256, 512]:
        p_trans = (np.array(trans_lens) > length).mean() * 100
        p_label = (np.array(label_lens) > length).mean() * 100
        print(f"\nTruncation at {length} tokens:")
        print(f"  Transliteration: {p_trans:.2f}% truncated")
        print(f"  Translation:     {p_label:.2f}% truncated")

if __name__ == "__main__":
    analyze_token_lengths()
