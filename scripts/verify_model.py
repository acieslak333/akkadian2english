import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def verify_env():
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.2f} MB")

def verify_tokenizer():
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    test_text = "um-ma kà-ru-um kà-ni-ia-ma ša-ra-nim ṣé-er ṭup-pá-kà ḫi-ba-at"
    tokens = tokenizer.encode(test_text)
    decoded = tokenizer.decode(tokens, skip_special_tokens=True)
    
    print(f"\nOriginal: {test_text}")
    print(f"Tokens: {tokens}")
    print(f"Decoded: {decoded}")
    
    # Check for specific chars
    special_chars = "šḫṣṭáàíìúùÉ"
    for char in special_chars:
        t = tokenizer.encode(char, add_special_tokens=False)
        d = tokenizer.decode(t)
        print(f"Char '{char}' -> Tokens {t} -> Decoded '{d}'")

if __name__ == "__main__":
    verify_env()
    verify_tokenizer()
