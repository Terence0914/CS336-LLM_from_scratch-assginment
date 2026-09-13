import torch
import numpy as np
import argparse
import csv
from losses import cross_entropy

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
parser.add_argument("--val-batches", type=int, default = 20)
parser.add_argument("--data-dtype", type=str, required=True)
parser.add_argument("--device", type=str, default = "cpu")
parser.add_argument("--min-learning-rate", type=float, required=True)
parser.add_argument("--warmup-iters", type=int, required=True)
parser.add_argument("--cosine-cycle-iters", type=int, required=True)
parser.add_argument("--max-l2-norm", type=float, required=True)
parser.add_argument("--seed", type = int, default = 42)
parser.add_argument("--metrics-path", type = str, default="metrics.csv")

args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
train_data = np.memmap(args.train_path, dtype=np.dtype(args.data_dtype), mode="r")
val_data = np.memmap(args.val_path, dtype=np.dtype(args.data_dtype), mode="r")
with open(args.metrics_path, "w", newline = "") as file:
    writer = csv.writer(file)
    writer.writerow(["iteration", "train_loss", "val_loss"])

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
    args.weight_decay,
)

for iteration in range(1, args.num_iterations + 1):
    optimizer.zero_grad()
    current_lr = get_lr_cosine_schedule(
        iteration - 1,
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
    loss = cross_entropy(flat_logits, flat_targets)
    loss.backward()
    gradient_clipping(model.parameters(), args.max_l2_norm)
    optimizer.step()
    if iteration % args.log_interval == 0:
        print(f"iteration {iteration}: train loss = {loss.item():.4f}")

    if iteration % args.val_interval == 0:
        validation_loss = 0
        for _ in range(args.val_batches):
            val_x, val_y = get_batch(val_data, args.batch_size, args.context_length, args.device)
            with torch.no_grad():
                val_logits = model(val_x)
                val_flat_logits = val_logits.reshape(-1, args.vocab_size)
                val_flat_targets = val_y.reshape(-1)
                val_loss = cross_entropy(val_flat_logits, val_flat_targets)
                validation_loss += val_loss.item()

        validation_loss /= args.val_batches
        print(f"iteration {iteration}: val loss = {validation_loss:.4f}")
        with open(args.metrics_path, "a", newline = "") as file:
            writer = csv.writer(file)
            writer.writerow([iteration, loss.item(), validation_loss])

    if iteration % args.checkpoint_interval == 0:
        save_checkpoint(model, optimizer, iteration, args.checkpoint_path)
    
    




