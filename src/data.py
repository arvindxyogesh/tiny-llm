# src/data.py
import os
import torch
from torch.utils.data import Dataset

class ByteDataset(Dataset):
    """
    Very simple byte-level dataset: reads a text file and converts bytes->integers.
    Good for baseline quick experiments.
    """
    def __init__(self, path, block_size=128):
        with open(path, "rb") as f:
            data = f.read()
        # keep as integers 0..255
        self.vocab_size = 256
        self.data = torch.tensor(list(data), dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return max(0, (len(self.data) - self.block_size))

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + 1 + self.block_size]
        return x, y
    
    # src/hf_data.py
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
import torch
from torch.utils.data import Dataset

class HFDataset(Dataset):
    def __init__(self, split="train", tokenizer_dir="tokenizer", block_size=128, subset=None):
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        if subset:
            ds = ds.select(range(subset))
        txts = [x["text"] for x in ds if x["text"] and x["text"].strip()]
        self.tokenizer = ByteLevelBPETokenizer(f"{tokenizer_dir}/vocab.json", f"{tokenizer_dir}/merges.txt")
        self.ids = []
        for t in txts:
            enc = self.tokenizer.encode(t).ids
            self.ids.extend(enc + [self.tokenizer.token_to_id("</s>")])  # join docs with EOS
        self.ids = torch.tensor(self.ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return max(0, (len(self.ids) - self.block_size))

    def __getitem__(self, idx):
        x = self.ids[idx: idx + self.block_size]
        y = self.ids[idx + 1: idx + 1 + self.block_size]
        return x, y

    
