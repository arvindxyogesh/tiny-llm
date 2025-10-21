from tokenizers import ByteLevelBPETokenizer
import os

data_path = "data/tiny_text.txt"       # Your text data
tokenizer_dir = "scripts/train_tokenizer"  # Where to save vocab.json & merges.txt

# Make directory if it doesn't exist
os.makedirs(tokenizer_dir, exist_ok=True)

# Read all text
with open(data_path, "r", encoding="utf-8") as f:
    text = f.read()

# Train tokenizer
tokenizer = ByteLevelBPETokenizer()
tokenizer.train_from_iterator([text], vocab_size=2000, min_frequency=2)

# Save tokenizer files
tokenizer.save_model(tokenizer_dir)

print(f"Tokenizer trained and saved in {tokenizer_dir}")
