from train_bpe import train_bpe


def show_case(name, input_path, vocab_size, special_tokens):
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
    print(f"\n{name}")
    print("merges:", merges)
    for token_id in range(256, len(vocab)):
        print(f"vocab[{token_id}]:", vocab[token_id])


show_case("plain abab", "smoke_abab.txt", 258, [])
show_case("with special token", "smoke_special.txt", 258, ["<|endoftext|>"])
