from __future__ import annotations

import argparse
import cProfile
import io
import os
import pickle
import pstats
import time

try:
    import resource
except ImportError:
    resource = None

from train_bpe import train_bpe


def max_rss_gb() -> float:
    if resource is None:
        return float("nan")
    # Linux reports ru_maxrss in KiB; this is the format used by Colab.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)


parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--vocab-size", type=int, default=10_000)
parser.add_argument("--out", default="tinystories_bpe.pkl")
parser.add_argument("--profile", action="store_true")
args = parser.parse_args()

special_tokens = ["<|endoftext|>"]

start = time.perf_counter()
if args.profile:
    profiler = cProfile.Profile()
    profiler.enable()
    vocab, merges = train_bpe(args.input, args.vocab_size, special_tokens)
    profiler.disable()
else:
    profiler = None
    vocab, merges = train_bpe(args.input, args.vocab_size, special_tokens)
elapsed = time.perf_counter() - start

longest_id, longest_token = max(vocab.items(), key=lambda item: len(item[1]))

with open(args.out, "wb") as f:
    pickle.dump({"vocab": vocab, "merges": merges}, f)

print(f"trained_vocab_size={len(vocab)}")
print(f"num_merges={len(merges)}")
print(f"elapsed_seconds={elapsed:.2f}")
print(f"max_rss_gb={max_rss_gb():.2f}")
print(f"longest_token_id={longest_id}")
print(f"longest_token_bytes={longest_token!r}")
print(f"longest_token_len={len(longest_token)}")
print(f"output={os.path.abspath(args.out)}")

if profiler is not None:
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(20)
    print(stream.getvalue())
