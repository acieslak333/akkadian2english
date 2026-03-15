import pandas as pd
import re
from sklearn.model_selection import train_test_split
import os

def clean_text_gaps(text):
    # Gap Normalization Patterns
    gap_patterns = [
        r'\[x\]', r'x', r'\(break\)', r'\(large break\)',
        r'\(n broken lines\)', r'\.\.\.', r'…', r'\[\.\.\.\]'
    ]
    for pattern in gap_patterns:
        text = re.sub(pattern, '<gap>', text)
    
    # Deduplicate gaps
    text = re.sub(r'(<gap>\s*)+', '<gap> ', text)
    text = text.replace('-<gap>', '- <gap>').replace('<gap>-', '<gap> -') # Handle hyphenated gaps
    return text.strip()

def clean_transliteration(text):
    if not isinstance(text, str):
        return ""
    
    text = clean_text_gaps(text)
    
    # Ḫ → H, ḫ → h
    text = text.replace('Ḫ', 'H').replace('ḫ', 'h')
    
    # KÙ.B. —> KÙ.BABBAR
    text = text.replace('KÙ.B.', 'KÙ.BABBAR')
    
    # Subscripts to normal integers
    subscript_map = {
        '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4',
        '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9'
    }
    for sub, digit in subscript_map.items():
        text = text.replace(sub, digit)
        
    # Decimals to Fractions for transliterations
    frac_map = {
        '0.5': '½', '0.25': '¼', '0.3333': '⅓', '0.8333': '⅚',
        '0.625': '⅝', '0.6666': '⅔', '0.75': '¾', '0.1666': '⅙'
    }
    for dec, frac in frac_map.items():
        text = text.replace(dec, frac)
        
    # Determinatives alignment
    text = text.replace('(d)', '{d}').replace('(ki)', '{ki}').replace('(TÚG)', 'TÚG')
    
    # Normalize whitespace
    text = " ".join(text.split())
    return text

def clean_translation(text):
    if not isinstance(text, str):
        return ""
    
    text = clean_text_gaps(text)
    
    # Replace PN with <gap>
    text = text.replace('PN', '<gap>')
    
    # Remove unwanted marks
    removals = ['fem.', 'sing.', 'pl.', 'plural', r'\(\?\)', r'\.\.', r'\?']
    for r in removals:
        text = re.sub(r, '', text)
        
    # Remove stray marks (but keep gap tags)
    text = re.sub(r'<<\s*>>|<<|>>|<(?!!gap>)|(?<!<gap)>', '', text)
    
    # Specific term replacements
    term_map = {
        '-gold': 'pašallum gold',
        '-tax': 'šadduātum tax',
        '-textiles': 'kutānum textiles',
        '1 / 12 (shekel)': '15 grains',
        '5 / 12 shekel': '⅓ shekel 15 grains',
        '5 11 / 12 shekels': '6 shekels less 15 grains',
        '7 / 12 shekel': '½ shekel 15 grains'
    }
    for old, new in term_map.items():
        text = text.replace(old, new)
    for old, new in term_map.items():
        text = text.replace(old, new)
        
    # Decimals to Fractions for translations
    frac_map = {
        '0.5': '½', '0.25': '¼', '0.3333': '⅓', '0.8333': '⅚',
        '0.625': '⅝', '0.6666': '⅔', '0.75': '¾', '0.1666': '⅙'
    }
    for dec, frac in frac_map.items():
        text = text.replace(dec, frac)
        
    # Roman numerals for months (month V -> month 5)
    roman_to_int = {
        'month XII': 'month 12', 'month XI': 'month 11', 'month X': 'month 10',
        'month IX': 'month 9', 'month VIII': 'month 8', 'month VII': 'month 7',
        'month VI': 'month 6', 'month V': 'month 5', 'month IV': 'month 4',
        'month III': 'month 3', 'month II': 'month 2', 'month I': 'month 1'
    }
    for r, i in roman_to_int.items():
        text = text.replace(r, i)
        
    # Choose first option for / translations (e.g., "you / she" -> "you")
    text = re.sub(r'(\w+)\s*/\s*(\w+)', r'\1', text)
    
    # Shorten long floats
    text = re.sub(r'(\d+\.\d{4})\d+', r'\1', text)
    
    # Final cleanup
    text = " ".join(text.split())
    return text

def align_gaps(trans_text, transla_text):
    # Ensure if one has <gap>, the other has at least one too
    if '<gap>' in trans_text and '<gap>' not in transla_text:
        transla_text = '<gap> ' + transla_text
    elif '<gap>' in transla_text and '<gap>' not in trans_text:
        trans_text = '<gap> ' + trans_text
    
    # Re-deduplicate just in case
    trans_text = re.sub(r'(<gap>\s*)+', '<gap> ', trans_text).strip()
    transla_text = re.sub(r'(<gap>\s*)+', '<gap> ', transla_text).strip()
    
    return trans_text, transla_text

def load_and_standardize(file_path, trans_col, translation_col):
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found.")
        return pd.DataFrame(columns=['transliteration', 'translation'])
    
    df = pd.read_csv(file_path, low_memory=False)
    # Filter for necessary columns and rename
    df = df[[trans_col, translation_col]].rename(columns={
        trans_col: 'transliteration',
        translation_col: 'translation'
    })
    return df

def preprocess_data(output_dir):
    sources = [
        ('data/train.csv', 'transliteration', 'translation'),
        ('data/Sentences_Oare_FirstWord_LinNum.csv', 'sentence_obj_in_text', 'translation'),
        ('data/eBL_Dictionary.csv', 'word', 'definition')
    ]
    
    dfs = []
    for path, trans_col, transla_col in sources:
        print(f"Loading {path}...")
        dfs.append(load_and_standardize(path, trans_col, transla_col))
    
    df = pd.concat(dfs, ignore_index=True)
    print(f"Combined data size: {len(df)}")
    
    # Remove duplicates
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Size after deduplication: {len(df)}")
    
    print("Cleaning text...")
    df['transliteration'] = df['transliteration'].apply(clean_transliteration)
    df['translation'] = df['translation'].apply(clean_translation)
    
    # Gap alignment
    print("Aligning gaps...")
    aligned = df.apply(lambda row: align_gaps(row['transliteration'], row['translation']), axis=1)
    df['transliteration'] = [a[0] for a in aligned]
    df['translation'] = [a[1] for a in aligned]
    
    # Remove empty or too short entries
    df = df[df['transliteration'].str.len() > 1]
    df = df[df['translation'].str.len() > 1]
    
    # Final deduplication after cleaning (some might become identical)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Final cleaned data size: {len(df)}")
    
    # Split into train and validation (90/10)
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    
    # Save processed files
    train_df.to_csv(os.path.join(output_dir, 'train_cleaned.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val_cleaned.csv'), index=False)
    
    print(f"Saved {len(train_df)} training and {len(val_df)} validation samples to {output_dir}")

def preprocess_test(input_file, output_dir):
    if not os.path.exists(input_file):
        print(f"Warning: {input_file} not found.")
        return
    df = pd.read_csv(input_file)
    df['transliteration'] = df['transliteration'].apply(clean_transliteration)
    df.to_csv(os.path.join(output_dir, 'test_cleaned.csv'), index=False)
    print(f"Saved cleaned test data to {output_dir}")

if __name__ == "__main__":
    os.makedirs('data/processed', exist_ok=True)
    preprocess_data('data/processed')
    preprocess_test('data/test.csv', 'data/processed')
