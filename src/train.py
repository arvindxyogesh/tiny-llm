# src/data.py
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
import os

class HFDataset(Dataset):
    def __init__(self, dataset_name="wikitext", subset="wikitext-2-raw-v1",
                 split="train", tokenizer_path="data/tokenizer",
                 block_size=16, subset_size=None):  # ← subset_size included
        print(f"Loading dataset {dataset_name}/{subset}, split={split}...")
        ds = load_dataset(dataset_name, subset, split=split)

        if subset_size is not None:
            ds = ds.select(range(subset_size))
            print(f"Using subset_size={subset_size} samples for quick testing.")

        text = "\n".join(ds["text"])
        os.makedirs(tokenizer_path, exist_ok=True)

        # Train or load tokenizer
        vocab_file = os.path.join(tokenizer_path, "vocab.json")
        merges_file = os.path.join(tokenizer_path, "merges.txt")

        if not os.path.exists(vocab_file):
            print("Training new ByteLevelBPETokenizer...")
            tokenizer = ByteLevelBPETokenizer()
            tokenizer.train_from_iterator([text], vocab_size=2000, min_frequency=2)
            tokenizer.save_model(tokenizer_path)
        else:
            tokenizer = ByteLevelBPETokenizer(vocab_file, merges_file)
            print("Loaded existing tokenizer.")

        self.tokenizer = tokenizer
        self.block_size = block_size

        # Tokenize full dataset
        encoded = tokenizer.encode(text)
        self.tokens = torch.tensor(encoded.ids, dtype=torch.long)
        print(f"Tokenized dataset length: {len(self.tokens)}")

    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx):
        chunk = self.tokens[idx: idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y
