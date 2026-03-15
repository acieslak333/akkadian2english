import pandas as pd
import os
import json

def build_lexicon_map(lexicon_file):
    df = pd.read_csv(lexicon_file)
    # We want to map 'form' to 'norm'
    # 'form' are often things like 'a-na' or 'um-ma'
    # 'norm' are the lexical forms
    
    # Filter for PN and words we want to normalize
    # For now, let's focus on PN (Proper Nouns) as they have many variants
    
    lex_map = {}
    for _, row in df.iterrows():
        form = str(row['form']).strip()
        norm = str(row['norm']).strip()
        if form and norm and form != norm:
            lex_map[form] = norm
            
    return lex_map

def apply_lexicon_normalization(df, lex_map):
    def normalize(text):
        if not isinstance(text, str):
            return ""
        # Split by tokens (hyphenated tokens and spaces)
        # Note: This is a bit complex as transliteration has 'um-ma'
        # Lexicon has 'a-ba-a-a' -> 'Abā'
        
        # Simple token-based replacement for now
        tokens = re.split(r'(\s+|-)', text)
        new_tokens = []
        for token in tokens:
            if token in lex_map:
                new_tokens.append(lex_map[token])
            else:
                new_tokens.append(token)
        return "".join(new_tokens)

    # Note: re is needed
    import re
    df['transliteration'] = df['transliteration'].apply(normalize)
    return df

if __name__ == "__main__":
    lex_file = 'data/OA_Lexicon_eBL.csv'
    train_file = 'data/processed/train_cleaned.csv'
    val_file = 'data/processed/val_cleaned.csv'
    test_file = 'data/processed/test_cleaned.csv'
    
    print("Building lexicon map...")
    lex_map = build_lexicon_map(lex_file)
    print(f"Loaded {len(lex_map)} mapping entries.")
    
    # Save mapping for reference
    with open('data/processed/lexicon_map.json', 'w', encoding='utf-8') as f:
        json.dump(lex_map, f, ensure_ascii=False, indent=2)
        
    for file_path in [train_file, val_file, test_file]:
        print(f"Normalizing {file_path}...")
        df = pd.read_csv(file_path)
        df_norm = apply_lexicon_normalization(df, lex_map)
        df_norm.to_csv(file_path.replace('.csv', '_norm.csv'), index=False)
