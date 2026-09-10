import torch
import numpy as np
import argparse
import torch.nn.functional as F

from model import TransformerLM, AdamW, get_batch, get_lr_cosine_schedule, gradient_clipping, save_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument("--train-path", type = str, required = True)
parser.add_argument("--val-path", type = str, required = True)
parser.add_argument("--vocab-size", type = int, required = True)
parser.add_argument("--context-length", type = int, required = True)
parser.add_argument("--d-model", type=int, required=True)
parser.add_argument("--num-layers", type=int, required=True)
parser.add_argument("--num-heads", type=int, required=True)
parser.add_argument("--d-ff", type=int, required=True)
parser.add_argument("--rope-theta", type=float, required=True)
parser.add_argument("--batch-size", type=int, required=True)
parser.add_argument("--learning-rate", type=float, required=True)
parser.add_argument("--beta1", type=float, required=True)
parser.add_argument("--beta2", type=float, required=True)
parser.add_argument("--eps", type=float, required=True)
parser.add_argument("--weight-decay", type=float, required=True)
parser.add_argument("--num-iterations", type=int, required=True)
parser.add_argument("--checkpoint-path", type=str, required=True)
parser.add_argument("--checkpoint-interval", type=int, required=True)
parser.add_argument("--log-interval", type=int, required=True)
parser.add_argument("--val-interval", type=int, required=True)
parser.add_argument("--data-dtype", type=str, required=True)
parser.add_argument("--device", type=str, default = "cpu")
parser.add_argument("--min-learning-rate", type=float, required=True)
parser.add_argument("--warmup-iters", type=int, required=True)
parser.add_argument("--cosine-cycle-iters", type=int, required=True)
parser.add_argument("--max-l2-norm", type=float, required=True)

args = parser.parse_args()
train_data = np.memmap(args.train_path, dtype=np.dtype(args.data_dtype), mode="r")
val_data = np.memmap(args.val_path, dtype=np.dtype(args.data_dtype), mode="r")

model = TransformerLM(
    vocab_size = args.vocab_size, 
    context_length = args.context_length,
    d_model = args.d_model,
    num_layers = args.num_layers,
    num_heads = args.num_heads,
    d_ff = args.d_ff,
    rope_theta = args.rope_theta,
)
model = model.to(args.device)
optimizer = AdamW(
    model.parameters(),
    args.learning_rate,
    (args.beta1, args.beta2),
    args.eps,
    args.weight_dacay,
)

for iteration in range(1, args.num_iterations + 1):
    optimizer.zero_grad()
    current_lr = get_lr_cosine_schedule(
        iteration,
        args.learning_rate,
        args.min_learning_rate,
        args.warmup_iters,
        args.cosine_cycle_iters,
    )
    for group in optimizer.param_groups:
        group["lr"] = current_lr

    x, y = get_batch(train_data, args.batch_size, args.context_length, args.device)
    logits = model(x)
    flat_logits = logits.reshape(-1, args.vocab_size)
    flat_targets = y.reshape(-1)
    loss = F.cross_entropy(flat_logits, flat_targets)
    loss.backward()
    gradient_clipping(model.parameters(), args.max_l2_norm)
    optimizer.step()
    if iteration % args.log_interval == 0:
        print(f"iteration {iteration}: train loss = {loss.item():.4f}")

    if iteration % args.val_interval == 0:
        val_x, val_y = get_batch(val_data, args.batch_size, args.context_length, args.device)
        with torch.no_grad():
            val_logits = model(val_x)
            val_flat_logits = val_logits.reshape(-1, args.vocab_size)
            val_flat_targets = val_y.reshape(-1)
            val_loss = F.cross_entropy(val_flat_logits, val_flat_targets)
        print(f"iteration {iteration}: val loss = {val_loss.item():.4f}")

    if iteration % args.checkpoint_interval == 0:
        save_checkpoint(model, optimizer, iteration, args.checkpoint_path)
    
    




