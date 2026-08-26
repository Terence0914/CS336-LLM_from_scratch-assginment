import os
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
import regex as re

# GPT-2 风格的预分词正则：先把原始文本切成单词、数字、标点、空白等片段，
# 后续的 BPE 合并只会在每个片段内部进行，不会跨越片段边界。
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# BPE 的基础词表：0~255 对应所有可能的单字节，因此任何 UTF-8 文本都能表示。
BYTE_TOKENS = tuple(bytes([i]) for i in range(256))

def find_chunk_boundaries(file, desired_num_chunks, split_special_token):
    """寻找适合并行处理的大文件分块边界。

    初始边界按文件字节数均分，再把中间边界向后移动到 special token 的位置，
    避免把 special token 从中间切断。
    """
    assert isinstance(split_special_token, bytes)

    # 通过移动文件指针得到文件总字节数，然后回到开头。
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    # 先得到近似均匀的候选边界。
    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    # 从每个候选边界开始向后搜索 special token，将其作为安全边界。
    mini_chunk_size = 4096
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)

        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break

            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


def pretokenize_text(chunk):
    """预分词并统计不同字节序列在文本块中出现的次数。"""
    local_sequences = Counter()
    for match in re.finditer(PAT, chunk):
        # 一个 pretoken 先转成 UTF-8，再拆成单字节 token 序列。
        token_bytes = match.group().encode("utf-8")
        seq = tuple(BYTE_TOKENS[b] for b in token_bytes)
        if len(seq) > 0:
            local_sequences[seq] += 1
    return local_sequences


def pretokenize_file_chunk(job):
    """读取一个文件区间，移除 special token 后完成局部预分词统计。"""
    input_path, start, end, special_tokens = job
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    # special token 不参加普通 BPE 训练，因此先把它们当作分隔符移除。
    # 按长度降序可以避免较短 special token 抢先匹配较长 token 的前缀。
    if special_tokens:
        escaped_tokens = [re.escape(token) for token in sorted(special_tokens, key=len, reverse=True)]
        chunks = re.split("|".join(escaped_tokens), chunk)
    else:
        chunks = [chunk]

    local_sequences = Counter()
    for chunk in chunks:
        local_sequences.update(pretokenize_text(chunk))
    return local_sequences


def train_bpe(input_path, vocab_size, special_tokens):
    """从训练语料学习 BPE 词表和按创建顺序排列的合并规则。"""
    # 设置环境变量 BPE_TIMING=1 时输出各阶段耗时，便于性能分析。
    timing = os.environ.get("BPE_TIMING") == "1"
    phase_start = time.perf_counter()

    # 词表首先包含全部 256 个单字节 token，保证不存在无法编码的字符。
    vocab = {i : bytes([i]) for i in range(256)}

    # special token 直接加入词表，不参与后面的 pair 频率竞争和合并。
    next_id = 256
    for token in special_tokens:
        if len(vocab) >= vocab_size:
            break
        vocab[next_id] = token.encode("utf-8")
        next_id += 1
    
    if timing:
        print(f"[timing] vocab_init={time.perf_counter() - phase_start:.2f}s", flush=True)

    phase_start = time.perf_counter()
    sequences = Counter()
    file_size = os.path.getsize(input_path)
    num_chunks = 64

    # 小文件直接作为一个任务；大文件切块后可交给多个进程并行预分词。
    if file_size < 1_000_000:
        boundaries = [0, file_size]
    elif special_tokens:
        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, num_chunks, special_tokens[0].encode("utf-8"))
    else:
        chunk_size = max(1, file_size // num_chunks)
        boundaries = list(range(0, file_size, chunk_size)) + [file_size]

    if timing:
        print(f"[timing] chunk_boundaries={time.perf_counter() - phase_start:.2f}s jobs={max(0, len(boundaries) - 1)}", flush=True)

    phase_start = time.perf_counter()
    # 每个 job 描述一个需要读取的文件区间。
    jobs = [
        (input_path, start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
        if end > start
    ]

    if len(jobs) < 2:
        for job in jobs:
            sequences.update(pretokenize_file_chunk(job))
    else:
        # 每个进程返回局部 Counter，主进程再把统计结果汇总到 sequences。
        with Pool() as pool:
            for job_i, local_sequences in enumerate(pool.imap_unordered(pretokenize_file_chunk, jobs), start=1):
                sequences.update(local_sequences)
                if timing and job_i % 8 == 0:
                    print(f"[timing] pretokenized_jobs={job_i}/{len(jobs)}", flush=True)

    if timing:
        print(f"[timing] pretokenization={time.perf_counter() - phase_start:.2f}s unique_sequences={len(sequences)}", flush=True)

    phase_start = time.perf_counter()
    # merges 按学习顺序保存；列表下标就是 tokenizer 使用的 merge rank。
    merges = []

    # pair_counts：相邻 token pair -> 在整个语料中的加权出现次数。
    # pair_to_sequences：相邻 pair -> 包含该 pair 的序列集合。
    # 第二张索引让每次合并只更新真正受影响的序列，而不必扫描全部语料。
    pair_counts = Counter()
    pair_to_sequences = defaultdict(set)
    for seq, count in sequences.items():
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i + 1])
            pair_counts[pair] += count
            pair_to_sequences[pair].add(seq)

    if timing:
        print(f"[timing] pair_index={time.perf_counter() - phase_start:.2f}s unique_pairs={len(pair_counts)}", flush=True)

    phase_start = time.perf_counter()
    # 每轮产生一个新 token，直到词表达到目标大小或已经没有可合并的 pair。
    while len(vocab) < vocab_size:
        if not pair_counts:
            break

        # 选择出现次数最多的 pair；次数相同时按 pair 的字节值取较大的那个。
        # item 的形式是 (pair, count)，所以比较键依次使用 count 和 pair。
        best_pair = max(pair_counts.items(), key = lambda item: (item[1], item[0]))[0]

        # 只处理包含 best_pair 的序列。list(...) 创建快照，因为循环中会修改索引。
        for seq in list(pair_to_sequences[best_pair]):
            count = sequences[seq]

            # 先撤销旧序列对所有相邻 pair 统计量的贡献。
            for j in range(len(seq) - 1):
                old_pair = (seq[j], seq[j + 1])
                pair_counts[old_pair] -= count
                if pair_counts[old_pair] <= 0:
                    del pair_counts[old_pair]
                pair_to_sequences[old_pair].discard(seq)

            # 从左到右合并 best_pair；一次命中后前进两格，避免 token 重复使用。
            new_seq = []
            i = 0
            while i < len(seq):
                if i < len(seq) - 1 and (seq[i], seq[i + 1]) == best_pair:
                    new_seq.append(seq[i] + seq[i + 1])
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1

            new_seq = tuple(new_seq)

            # 用合并后的序列替换旧序列，并保留它在语料中的出现次数。
            del sequences[seq]
            sequences[new_seq] += count

            # 将新序列产生的相邻 pair 重新加入频率表和反向索引。
            for j in range(len(new_seq) - 1):
                new_pair = (new_seq[j], new_seq[j + 1])
                pair_counts[new_pair] += count
                pair_to_sequences[new_pair].add(new_seq)

        # 记录本轮规则，并把合并结果作为一个新 token 加入词表。
        merges.append(best_pair)
        vocab[next_id] = best_pair[0] + best_pair[1]
        next_id += 1
    if timing:
        print(f"[timing] merge_loop={time.perf_counter() - phase_start:.2f}s merges={len(merges)}", flush=True)
    return vocab, merges
