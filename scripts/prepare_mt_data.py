import pandas as pd
import os
import re
import random

def add_tags(df, tag):
    df['transliteration'] = tag + " " + df['transliteration'].astype(str)
    return df

def safe_augment_formula(df):
    """
    Safely augment only the highly predictable 'um-ma ... a-na ...' formula.
    """
    # Simplified search for the formula
    formula_pattern = r'um-ma ([\w-]+) a-na ([\w-]+)'
    
    aug_rows = []
    # Identify potential names from the dataset itself to keep it consistent
    all_names = re.findall(r'um-ma ([\w-]+)', " ".join(df['transliteration'].astype(str)))
    all_names += re.findall(r'a-na ([\w-]+)', " ".join(df['transliteration'].astype(str)))
    unique_names = list(set(all_names))
    
    if len(unique_names) < 2:
        return pd.DataFrame(columns=df.columns)

    for _, row in df.iterrows():
        match = re.search(formula_pattern, row['transliteration'])
        if match:
            # Create 2 augmented versions with random name swaps
            for _ in range(2):
                new_name1 = random.choice(unique_names)
                new_name2 = random.choice(unique_names)
                
                new_trans = row['transliteration'].replace(match.group(1), new_name1).replace(match.group(2), new_name2)
                # Note: We don't have a reliable way to swap names in English translation without NLP/NER
                # So we only augment the transliteration if we can also identify them in English.
                # Since that's risky, let's stick to sub-word masking for general augmentation.
                pass
    
    return pd.DataFrame(aug_rows)

def subword_masking(text, mask_prob=0.15):
    tokens = text.split()
    if len(tokens) < 3:
        return text
    
    new_tokens = []
    for t in tokens:
        if random.random() < mask_prob and len(t) > 2:
            new_tokens.append("<gap>")
        else:
            new_tokens.append(t)
    return " ".join(new_tokens)

def prepare_data():
    base_dir = 'datasets/base'
    aug_dir = 'datasets/augmented'
    processed_dir = 'data/processed'
    
    train_path = os.path.join(processed_dir, 'train_cleaned.csv')
    val_path = os.path.join(processed_dir, 'val_cleaned.csv')
    
    if not os.path.exists(train_path):
        print("Error: train_cleaned.csv not found. Run clean_data.py first.")
        return

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    # --- BASE VERSION (Tagged) ---
    print("Creating Base (Tagged) version...")
    # We'll assume everything in train_cleaned is a sentence for now
    # In a more complex setup, we'd track the source of each row.
    # Since we merged them in clean_data.py, let's just use <<SENTENCE>>.
    base_train = add_tags(train_df.copy(), "<<SENTENCE>>")
    base_val = add_tags(val_df.copy(), "<<SENTENCE>>")
    
    base_train.to_csv(os.path.join(base_dir, 'train.csv'), index=False)
    base_val.to_csv(os.path.join(base_dir, 'val.csv'), index=False)
    
    # --- AUGMENTED VERSION (Tagged + Masked) ---
    print("Creating Augmented version...")
    # Add original tagged data
    aug_train = base_train.copy()
    
    # Generate augmented samples via masking
    masked_samples = base_train.copy()
    masked_samples['transliteration'] = masked_samples['transliteration'].apply(lambda x: subword_masking(x))
    
    # Combine
    aug_train = pd.concat([aug_train, masked_samples], ignore_index=True)
    aug_train = aug_train.drop_duplicates().reset_index(drop=True)
    
    aug_train.to_csv(os.path.join(aug_dir, 'train.csv'), index=False)
    # Validation stays the same as base (usually don't augment validation)
    base_val.to_csv(os.path.join(aug_dir, 'val.csv'), index=False)
    
    print(f"Base train size: {len(base_train)}")
    print(f"Augmented train size: {len(aug_train)}")

if __name__ == "__main__":
    prepare_data()
