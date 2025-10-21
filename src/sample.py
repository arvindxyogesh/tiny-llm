# src/sample.py
import torch
from model import GPT, GPTConfig
from data import HFDataset

def load_model(model_path="checkpoints/final.pt", subset_size=100):
    # Load a small subset dataset to get tokenizer
    ds = HFDataset(split="train",tokenizer_dir="scripts/train_tokenizer",block_size=16)
    vocab_size = ds.tokenizer.get_vocab_size()
    block_size = ds.block_size

    # Initialize model
    cfg = GPTConfig(vocab_size=vocab_size, block_size=block_size,
                    n_layer=4, n_head=4, n_embd=256)
    model = GPT(cfg)

    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model, ds

def generate_text(model, ds, prompt, max_new_tokens=50, temperature=1.0, top_k=10):
    tokens = ds.tokenizer.encode(prompt).ids
    x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits, _ = model(x, x)
            logits = logits[0, -1] / temperature
            top_logits, top_indices = torch.topk(logits, top_k)
            probs = torch.softmax(top_logits, dim=-1)
            next_token = top_indices[torch.multinomial(probs, 1)]
            x = torch.cat([x, next_token.unsqueeze(0)], dim=1)

    return ds.tokenizer.decode(x[0].tolist())

def main():
    print("📦 Loading model and tokenizer...")
    model, ds = load_model()
    print("✅ Model ready. Type 'quit' to exit.")

    while True:
        prompt = input("\nPrompt: ")
        if prompt.lower() in ["quit", "exit"]:
            break
        output = generate_text(model, ds, prompt)
        print("\nGenerated:\n", output)

if __name__ == "__main__":
    main()
