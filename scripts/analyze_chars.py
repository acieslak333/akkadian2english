import pandas as pd
from collections import Counter
import re

def analyze_chars(file_path):
    df = pd.read_csv(file_path)
    all_text = " ".join(df['transliteration'].dropna().astype(str))
    chars = Counter(all_text)
    
    # Sort by frequency
    sorted_chars = sorted(chars.items(), key=lambda x: x[1], reverse=True)
    
    print(f"Total unique characters in {file_path}: {len(chars)}")
    print("Top 50 characters:")
    for char, freq in sorted_chars[:50]:
        print(f"'{char}': {freq}")
        
    # Check for specific patterns
    subscripts = re.findall(r'[₀-₉]', all_text)
    print(f"\nSubscripts found: {Counter(subscripts)}")
    
    accented = re.findall(r'[áéíóúàèìòùâêîôûāēīōūṣšḫṭ]', all_text, re.IGNORECASE)
    print(f"Accented characters found: {Counter(accented)}")

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'data/train.csv'
    analyze_chars(path)
