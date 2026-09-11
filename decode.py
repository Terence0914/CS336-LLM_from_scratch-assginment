import torch

def generate(model, tokenizer, prompt, max_new_tokens, temperature = 1.0, top_p= 1.0):
    token_ids = tokenizer.encode(prompt)

    tokens = torch.tensor(
        token_ids,
        dtype = torch.long,
        device = next(model.parameters()).device
    ).unsqueeze(0)

    with torch.no_grad():
        eos_token_id = tokenizer.special_to_id.get("<|endoftext|>")
        for _ in range(max_new_tokens):
            model_input = tokens[:, -model.context_length:]
            logits = model(model_input)
            next_logits = logits[:, -1, :]
            next_logits = next_logits / temperature
            probs = torch.softmax(next_logits, dim = -1)

            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(
                    probs, descending = True, dim = -1
                )
                cumulative_probs = torch.cumsum(sorted_probs, dim = -1)
                sorted_mask = cumulative_probs >= top_p
                sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                sorted_mask[..., 0] = False
                sorted_probs = sorted_probs.masked_fill(sorted_mask, 0.0)
                sorted_probs = sorted_probs / sorted_probs.sum(dim = -1, keepdim = True)
                filtered_probs = torch.zeros_like(probs)
                filtered_probs.scatter_(dim = -1, index = sorted_indices, src = sorted_probs)
                probs = filtered_probs

            next_token = torch.multinomial(probs, num_samples = 1)
            tokens = torch.cat([tokens, next_token], dim = -1)
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

    output_ids = tokens[0].tolist()
    return tokenizer.decode(output_ids)