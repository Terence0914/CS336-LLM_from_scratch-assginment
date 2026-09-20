from pathlib import Path
from multiprocessing import Pool

import numpy as np

from tokenizer import Tokenizer
from tokenizer.train_bpe import find_chunk_boundaries

CODE_DIR = Path(__file__).resolve().parent
DATA_DIR = CODE_DIR.parent / "data"

TRAIN_PATH = DATA_DIR / "TinyStoriesV2-GPT4-train.txt"
OUTPUT_PATH = DATA_DIR / "tinystories_train.bin"

VOCAB_PATH = DATA_DIR / "tinystories_vocab.pkl"
MERGES_PATH = DATA_DIR / "tinystories_merges.pkl"

tokenizer = None

def init_worker():
    global tokenizer

    tokenizer = Tokenizer.from_files(
        VOCAB_PATH,
        MERGES_PATH,
        special_tokens=["<|endoftext|>"],
    )


def encode_chunk(job):
    path, start, end = job

    with open(path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8")

    token_ids = tokenizer.encode(text)

    return np.asarray(token_ids, dtype=np.uint16)


def main():
    with open(TRAIN_PATH, "rb") as f:
        boundaries = find_chunk_boundaries(
            f,
            64,
            b"<|endoftext|>",
        )

    jobs = [
        (TRAIN_PATH, start, end)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    print(f"Processing {len(jobs)} chunks...")

    with open(OUTPUT_PATH, "wb") as out:
        with Pool(initializer=init_worker) as pool:
            for i, token_ids in enumerate(
                pool.imap(encode_chunk, jobs),
                start=1,
            ):
                token_ids.tofile(out)
                print(f"Finished chunk {i}/{len(jobs)}")

    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()