from __future__ import annotations

import json
import pickle
from collections.abc import Iterable, Iterator

import regex as re

from .pre_tokenize import PAT


def _gpt2_bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1))
    bs += list(range(ord("\xa1"), ord("\xac") + 1))
    bs += list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    extra = 0
    for byte in range(256):
        if byte not in bs:
            bs.append(byte)
            cs.append(256 + extra)
            extra += 1
    return dict(zip(bs, (chr(value) for value in cs), strict=True))


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self.cache: dict[bytes, tuple[int, ...]] = {}
        self.pretoken_pattern = re.compile(PAT)

        self.special_tokens = list(dict.fromkeys(special_tokens or []))
        next_token_id = max(self.vocab, default=-1) + 1
        vocab_values = set(self.vocab.values())
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in vocab_values:
                self.vocab[next_token_id] = token_bytes
                vocab_values.add(token_bytes)
                next_token_id += 1

        self.bytes2int = {value: key for key, value in self.vocab.items()}
        self.special_token_ids = {
            token: self.bytes2int[token.encode("utf-8")] for token in self.special_tokens
        }
        self.special_pattern = None
        if self.special_tokens:
            escaped_tokens = [re.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True)]
            self.special_pattern = re.compile("|".join(escaped_tokens))

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        vocab = cls._load_vocab(vocab_filepath)
        merges = cls._load_merges(merges_filepath)
        return cls(vocab, merges, special_tokens)

    @staticmethod
    def _load_vocab(vocab_filepath: str) -> dict[int, bytes]:
        with open(vocab_filepath, "rb") as handle:
            try:
                vocab = pickle.load(handle)
            except Exception:
                pass

        if "vocab" not in locals():
            with open(vocab_filepath, encoding="utf-8") as handle:
                raw_vocab = json.load(handle)
                unicode_to_byte = {value: key for key, value in _gpt2_bytes_to_unicode().items()}
            vocab = {
                index: bytes(unicode_to_byte[char] for char in token)
                for token, index in raw_vocab.items()
            }

        return {int(index): value for index, value in vocab.items()}

    @staticmethod
    def _load_merges(merges_filepath: str) -> list[tuple[bytes, bytes]]:
        with open(merges_filepath, "rb") as handle:
            try:
                merges = pickle.load(handle)
                return [(left, right) for left, right in merges]
            except Exception:
                handle.seek(0)
                unicode_to_byte = {value: key for key, value in _gpt2_bytes_to_unicode().items()}
                parsed_merges: list[tuple[bytes, bytes]] = []
                for line in handle.read().decode("utf-8").splitlines():
                    parts = line.rstrip().split(" ")
                    if len(parts) != 2:
                        continue
                    left, right = parts
                    parsed_merges.append(
                        (
                            bytes(unicode_to_byte[char] for char in left),
                            bytes(unicode_to_byte[char] for char in right),
                        )
                    )
                return parsed_merges

    def _encode_pretoken(self, token_bytes: bytes) -> tuple[int, ...]:
        cached = self.cache.get(token_bytes)
        if cached is not None:
            return cached

        parts = [bytes([byte]) for byte in token_bytes]
        while len(parts) > 1:
            best_pair = None
            best_rank = None
            for pair in zip(parts, parts[1:]):
                rank = self.merge_ranks.get(pair)
                if rank is None:
                    continue
                if best_rank is None or rank < best_rank:
                    best_pair = pair
                    best_rank = rank

            if best_pair is None:
                break

            merged = best_pair[0] + best_pair[1]
            new_parts: list[bytes] = []
            index = 0
            while index < len(parts):
                if index + 1 < len(parts) and (parts[index], parts[index + 1]) == best_pair:
                    new_parts.append(merged)
                    index += 2
                else:
                    new_parts.append(parts[index])
                    index += 1
            parts = new_parts

        encoded = tuple(self.bytes2int[part] for part in parts)
        self.cache[token_bytes] = encoded
        return encoded

    def _match_special(self, buffer: str, position: int, partial: bool) -> re.Match[str] | None:
        if self.special_pattern is None:
            return None
        return self.special_pattern.match(buffer, position, partial=partial)

    def _iter_encode_chunks(self, iterable: Iterable[str]) -> Iterator[int]:
        buffer = ""

        for chunk in iterable:
            buffer += chunk
            position = 0
            while position < len(buffer):
                special_match = None
                if self.special_pattern is not None:
                    special_match = self.special_pattern.search(buffer, position, partial=True)

                segment_end = special_match.start() if special_match is not None else len(buffer)
                while position < segment_end:
                    match = self.pretoken_pattern.match(
                        buffer,
                        position,
                        endpos=segment_end,
                        partial=special_match is None,
                    )
                    if match is None or match.partial:
                        break
                    yield from self._encode_pretoken(match.group().encode("utf-8"))
                    position = match.end()

                if position < segment_end:
                    break

                if special_match is None:
                    continue
                if special_match.partial:
                    break

                yield self.special_token_ids[special_match.group()]
                position = special_match.end()

            buffer = buffer[position:]

        position = 0
        while position < len(buffer):
            special_match = None
            if self.special_pattern is not None:
                special_match = self.special_pattern.search(buffer, position)

            segment_end = special_match.start() if special_match is not None else len(buffer)
            while position < segment_end:
                match = self.pretoken_pattern.match(buffer, position, endpos=segment_end)
                if match is None:
                    raise ValueError("Failed to tokenize remaining input.")
                yield from self._encode_pretoken(match.group().encode("utf-8"))
                position = match.end()

            if special_match is None:
                continue

            if position != special_match.start():
                raise ValueError("Failed to align special-token boundary.")

            if special_match is not None:
                yield self.special_token_ids[special_match.group()]
                position = special_match.end()

    def encode(self, text: str) -> list[int]:
        return list(self._iter_encode_chunks([text]))

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        yield from self._iter_encode_chunks(iterable)

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[token_id] for token_id in ids).decode("utf-8", errors="replace")
