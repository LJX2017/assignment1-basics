import os
from collections import Counter
from multiprocessing import Pool
from typing import BinaryIO

import regex as re


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_tokens: list[str],
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_tokens, list), "Must represent special tokens as a list[str]"

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    if desired_num_chunks <= 1 or file_size == 0:
        return [0, file_size]

    special_token_bytes = [token.encode("utf-8") for token in split_special_tokens]
    if not special_token_bytes:
        return [0, file_size]

    chunk_size = file_size // desired_num_chunks
    if chunk_size == 0:
        return [0, file_size]

    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    for boundary_index in range(1, len(chunk_boundaries) - 1):
        search_position = chunk_boundaries[boundary_index]
        while True:
            file.seek(search_position)
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[boundary_index] = file_size
                break

            found_at = min(
                (position for token in special_token_bytes if (position := mini_chunk.find(token)) != -1),
                default=-1,
            )
            if found_at != -1:
                chunk_boundaries[boundary_index] = search_position + found_at
                break

            search_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


def string_2_bytes_tuple(s: str) -> tuple[bytes, ...]:
    return tuple(bytes([byte]) for byte in s.encode("utf-8"))


def pre_tokenize_chunk(chunk: str, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    token_2_count: dict[tuple[bytes, ...], int] = {}

    split_pattern = None
    if special_tokens:
        escaped_tokens = [re.escape(token) for token in sorted(special_tokens, key=len, reverse=True)]
        split_pattern = "|".join(escaped_tokens)

    chunks = re.split(split_pattern, chunk) if split_pattern else [chunk]
    for doc in chunks:
        for match in re.finditer(PAT, doc):
            word = match.group()
            word_bytes = string_2_bytes_tuple(word)
            token_2_count[word_bytes] = token_2_count.get(word_bytes, 0) + 1
    return token_2_count


def pre_tokenize_file(data_file_path: str, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    args_list: list[tuple[str, list[str]]] = []
    num_processes = min(4, os.cpu_count() or 1)

    with open(data_file_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            args_list.append((chunk, special_tokens))

    if not args_list:
        return {}

    if len(args_list) == 1:
        token_count_list = [pre_tokenize_chunk(*args_list[0])]
    else:
        with Pool(processes=min(num_processes, len(args_list))) as pool:
            token_count_list = pool.starmap(pre_tokenize_chunk, args_list)

    final_count: Counter[tuple[bytes, ...]] = Counter()
    for token_count in token_count_list:
        final_count.update(token_count)
    return dict(final_count)
