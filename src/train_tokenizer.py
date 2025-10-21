"""train_tokenizer.py
Train a BPE tokenizer on your corpus using Hugging Face tokenizers library.
Run: python src/train_tokenizer.py --input data/tiny_text.txt --out_tokenizer models/tokenizer.json --vocab_size 2000
"""
import argparse
from tokenizers import ByteLevelBPETokenizer

def main(args):
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=[args.input], vocab_size=args.vocab_size, min_frequency=2, special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
    tokenizer.save(args.out_dir)
    print("Saved tokenizer to", args.out_dir)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--out_dir', default='models/tokenizer')
    p.add_argument('--vocab_size', type=int, default=2000)
    args = p.parse_args()
    main(args)
