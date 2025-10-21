import torch
import torch.nn.functional as F

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
    assert logits.dim() == 1
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][-1]
        logits[indices_to_remove] = filter_value
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        # shift the indices to the right to keep first token above threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

def sample_sequence(model, tokenizer, start_ids, length=100, temperature=1.0, top_k=50, top_p=0.95, device='cpu'):
    model.eval()
    generated = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)
    for _ in range(length):
        with torch.no_grad():
            logits = model(generated)[0, -1, :].squeeze()
        logits = logits / temperature
        filtered_logits = top_k_top_p_filtering(logits.clone(), top_k=top_k, top_p=top_p)
        probs = F.softmax(filtered_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
        if generated.size(1) > model.pos_emb.size(1):
            generated = generated[:, -model.pos_emb.size(1):]
    return generated.squeeze().tolist()
