from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import heapq

from .pre_tokenize import pre_tokenize_file


Pair = tuple[bytes, bytes]
Word = tuple[bytes, ...]


@dataclass(frozen=True)
class HeapItem:
    count: int
    pair: Pair

    def __lt__(self, other: "HeapItem") -> bool:
        if self.count != other.count:
            return self.count > other.count
        return self.pair > other.pair


def count_word_pairs(word: Word) -> Counter[Pair]:
    return Counter(zip(word, word[1:]))


def merge_word(word: Word, merged_pair: Pair) -> tuple[Word, bool]:
    merged_token = merged_pair[0] + merged_pair[1]
    new_word: list[bytes] = []
    changed = False
    index = 0

    while index < len(word):
        if index + 1 < len(word) and (word[index], word[index + 1]) == merged_pair:
            new_word.append(merged_token)
            index += 2
            changed = True
        else:
            new_word.append(word[index])
            index += 1

    return tuple(new_word), changed


def build_pair_caches(
    pre_token_counts: dict[Word, int],
) -> tuple[dict[int, Word], dict[int, int], dict[Pair, int], dict[Pair, set[int]], list[HeapItem]]:
    words: dict[int, Word] = {}
    word_counts: dict[int, int] = {}
    pair_counts: dict[Pair, int] = defaultdict(int)
    pair_to_word_ids: dict[Pair, set[int]] = defaultdict(set)

    for word_id, (word, count) in enumerate(pre_token_counts.items()):
        words[word_id] = word
        word_counts[word_id] = count
        for pair, occurrences in count_word_pairs(word).items():
            pair_counts[pair] += occurrences * count
            pair_to_word_ids[pair].add(word_id)

    heap = [HeapItem(count, pair) for pair, count in pair_counts.items() if count > 0]
    heapq.heapify(heap)
    return words, word_counts, dict(pair_counts), pair_to_word_ids, heap


def get_top_pair(pair_counts: dict[Pair, int], heap: list[HeapItem]) -> Pair:
    while heap:
        top = heapq.heappop(heap)
        if pair_counts.get(top.pair, 0) == top.count and top.count > 0:
            return top.pair
    raise ValueError("No mergeable pairs remain.")


def merge_pair(
    merged_pair: Pair,
    words: dict[int, Word],
    word_counts: dict[int, int],
    pair_counts: dict[Pair, int],
    pair_to_word_ids: dict[Pair, set[int]],
    heap: list[HeapItem],
) -> None:
    affected_word_ids = list(pair_to_word_ids.get(merged_pair, ()))

    for word_id in affected_word_ids:
        old_word = words[word_id]
        old_pair_counts = count_word_pairs(old_word)
        if old_pair_counts.get(merged_pair, 0) == 0:
            pair_to_word_ids[merged_pair].discard(word_id)
            continue

        weight = word_counts[word_id]
        for pair, occurrences in old_pair_counts.items():
            pair_counts[pair] -= occurrences * weight
            pair_to_word_ids[pair].discard(word_id)
            if pair_counts[pair] > 0:
                heapq.heappush(heap, HeapItem(pair_counts[pair], pair))

        new_word, changed = merge_word(old_word, merged_pair)
        if not changed:
            for pair, occurrences in old_pair_counts.items():
                pair_counts[pair] += occurrences * weight
                pair_to_word_ids[pair].add(word_id)
                heapq.heappush(heap, HeapItem(pair_counts[pair], pair))
            continue

        words[word_id] = new_word
        for pair, occurrences in count_word_pairs(new_word).items():
            pair_counts[pair] = pair_counts.get(pair, 0) + occurrences * weight
            pair_to_word_ids[pair].add(word_id)
            heapq.heappush(heap, HeapItem(pair_counts[pair], pair))

    pair_to_word_ids.pop(merged_pair, None)


def get_tokenizer(
    data_file_path: str = "data/TinyStoriesV2-GPT4-valid.txt",
    special_tokens: list[str] = ["<|endoftext|>"],
    total_num_tokens: int = 257,
) -> tuple[dict[int, bytes], list[Pair]]:
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    next_token_id = 256

    for token in special_tokens:
        vocab[next_token_id] = token.encode("utf-8")
        next_token_id += 1

    pre_token_counts = pre_tokenize_file(str(data_file_path), special_tokens)
    words, word_counts, pair_counts, pair_to_word_ids, heap = build_pair_caches(pre_token_counts)
    merges: list[Pair] = []

    while next_token_id < total_num_tokens:
        merged_pair = get_top_pair(pair_counts, heap)
        merges.append(merged_pair)
        vocab[next_token_id] = merged_pair[0] + merged_pair[1]
        next_token_id += 1
        merge_pair(merged_pair, words, word_counts, pair_counts, pair_to_word_ids, heap)

    return vocab, merges
