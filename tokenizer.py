import regex as re
import pickle

# 与 train_bpe 使用相同的 GPT-2 风格预分词正则。
# BPE 只会在一个 pretoken 内部应用，避免随意跨单词或标点边界合并。
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens = None):
        """保存训练结果，并建立编码、解码和 BPE 合并所需的查询表。"""
        # vocab 的方向是 token_id -> token_bytes，主要供 decode 使用。
        self.vocab = vocab
        self.merges = merges
        self.merge_ranks = {}
        self.byte_to_id = {}

        # 建立 vocab 的反向映射：token_bytes -> token_id，主要供 encode 使用。
        for token_id, token_bytes in self.vocab.items():
            self.byte_to_id[token_bytes] = token_id

        # merges 的列表顺序就是优先级：rank 越小，合并优先级越高。
        for rank, merge_pair in enumerate(self.merges):
            self.merge_ranks[merge_pair] = rank

        if special_tokens is None:
            special_tokens = []

        self.special_tokens = special_tokens

        # 优先匹配较长的 special token，防止短 token 抢先匹配长 token 的前缀。
        self.sorted_special_tokens = sorted(self.special_tokens, key = len, reverse = True)
        self.escaped_special_tokens = []
        for token in self.sorted_special_tokens:
            # 转义正则特殊字符，使 special token 按字面含义进行匹配。
            tokens = re.escape(token)
            self.escaped_special_tokens.append(tokens)
        self.special_to_id = {}

        # 捕获组让 re.split 在切分普通文本时保留 special token 本身。
        self.special_alternation = "|".join(self.escaped_special_tokens)
        if self.special_alternation == "":
            self.special_pattern = None
        else:
            self.special_pattern = "(" + self.special_alternation + ")"

        # 已存在于 vocab 的 special token 复用原 ID，否则追加到词表末尾。
        next_id = max(self.vocab.keys()) + 1
        for special_token in self.special_tokens:
            token_bytes = special_token.encode("utf-8")
            if token_bytes in self.byte_to_id:
                token_id = self.byte_to_id[token_bytes]
            else:
                token_id = next_id
                self.vocab[token_id] = token_bytes
                self.byte_to_id[token_bytes] = token_id
                next_id += 1
            self.special_to_id[special_token] = token_id

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """从 pickle 文件读取训练结果，然后构造并返回 Tokenizer。"""
        with open(vocab_filepath, "rb") as vocab_file:
            vocab = pickle.load(vocab_file)
        with open(merges_filepath, "rb") as merges_file:
            merges = pickle.load(merges_file)
        return cls(vocab, merges, special_tokens)


    def decode(self, ids):
        """把 token ID 序列还原为文本。"""
        # 每个 ID 查回对应 bytes，再按原顺序连接起来。
        id_bytes = []
        for token_id in ids:
            id_bytes.append(self.vocab[token_id])
        all_bytes = b"".join(id_bytes)

        # 如果字节序列不是合法 UTF-8，用 Unicode replacement character 替换坏字节。
        text = all_bytes.decode("utf-8", errors = "replace")
        return text

    def _apply_bpe(self, token_bytes):
        """对一个 pretoken 的 UTF-8 bytes 反复应用已经训练好的 BPE 规则。"""
        # 初始状态是单字节 token 序列；所有文本至少都能退化成这种表示。
        tokens = []
        for byte_value in token_bytes:
            tokens.append(bytes([byte_value]))

        # 每轮选择当前可用且 rank 最小的 pair，直到没有规则可以继续应用。
        while True:
            # 枚举当前序列中的所有相邻 token pair。
            pairs = []
            for i in range(len(tokens) - 1):
                left = tokens[i]
                right = tokens[i + 1]
                pair = (left, right)
                pairs.append(pair)

            # 只保留训练阶段真正学到过的合并规则。
            mergeable_pairs = []
            for pair in pairs:
                if pair in self.merge_ranks:
                    mergeable_pairs.append(pair)
            if len(mergeable_pairs) == 0:
                return tokens
            else:
                # rank 越小代表训练时越早产生，因此需要优先执行。
                best_pair = mergeable_pairs[0]
                for current_pair in mergeable_pairs:
                    if self.merge_ranks[current_pair] < self.merge_ranks[best_pair]:
                        best_pair = current_pair

            # 从左到右合并这一轮选中的 best_pair 的所有非重叠出现位置。
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1:
                    cur_pair = (tokens[i], tokens[i + 1])
                    if cur_pair == best_pair:
                        new_tokens.append(tokens[i] + tokens[i + 1])
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

    def encode(self, text):
        """把输入文本编码为 token ID 列表。"""
        ids = []

        # 必须先分离 special token，保证它不会进入普通预分词和 BPE 流程。
        if self.special_pattern is None:
            pieces = [text]
        else:
            pieces = re.split(self.special_pattern, text)

        for piece in pieces:
            if piece in self.special_to_id:
                # special token 永远作为一个整体，直接查询自己的 ID。
                special_id = self.special_to_id[piece]
                ids.append(special_id)
            else:
                # 普通文本：正则预分词 -> UTF-8 bytes -> BPE -> token ID。
                for match in re.finditer(PAT, piece):
                    pretoken = match.group()
                    pretoken_bytes = pretoken.encode("utf-8")
                    bpe_tokens = self._apply_bpe(pretoken_bytes)
                    for bpe_token in bpe_tokens:
                        # BPE 结果一定来自训练词表，因此可以通过反向表找到 ID。
                        token_id = self.byte_to_id[bpe_token]
                        ids.append(token_id)
        return ids

    def encode_iterable(self, iterable):
        """逐块编码字符串迭代器，并逐个产出 ID，避免一次加载整个大文件。"""
        for chunk in iterable:
            chunk_ids = self.encode(chunk)
            for token_id in chunk_ids:
                yield token_id


        
            



