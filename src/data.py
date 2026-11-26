import json
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from .config import Config


SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
PAD_ID, UNK_ID, BOS_ID, EOS_ID = range(4)


def tokenize(text: str) -> list[str]:
    text = text.replace("\ufeff", "")
    text = unicodedata.normalize("NFC", text).lower()
    return text.split()


@dataclass
class TranslationRecord:
    row_id: int
    source_tokens: list[str]
    target_tokens: list[str]


class Vocabulary:
    def __init__(self, token_to_id: dict[str, int]):
        self.token_to_id = token_to_id
        self.id_to_token = {index: token for token, index in token_to_id.items()}
        for expected_id, token in enumerate(SPECIAL_TOKENS):
            if self.token_to_id.get(token) != expected_id:
                raise ValueError(f"Token {token} phải có id {expected_id}")

    def __len__(self) -> int:
        return len(self.token_to_id)

    def encode(self, tokens: Sequence[str]) -> list[int]:
        return [self.token_to_id.get(token, UNK_ID) for token in tokens]

    def decode(self, token_ids: Iterable[int], stop_at_eos: bool = True) -> list[str]:
        result = []
        for value in token_ids:
            token_id = int(value)
            if stop_at_eos and token_id == EOS_ID:
                break
            if token_id not in (PAD_ID, BOS_ID):
                result.append(self.id_to_token.get(token_id, "<unk>"))
        return result

    def to_dict(self) -> dict[str, int]:
        return dict(self.token_to_id)


def build_vocabulary(
    sequences: Iterable[Sequence[str]],
    vocab_limit: int,
    min_frequency: int,
) -> Vocabulary:
    frequencies = Counter(token for sequence in sequences for token in sequence)
    ranked = sorted(frequencies, key=lambda token: (-frequencies[token], token))
    normal_limit = vocab_limit - len(SPECIAL_TOKENS)
    selected = [
        token
        for token in ranked
        if frequencies[token] >= min_frequency and token not in SPECIAL_TOKENS
    ][:normal_limit]
    return Vocabulary({token: index for index, token in enumerate(SPECIAL_TOKENS + selected)})


def load_split(file_path, max_words: int) -> tuple[list[TranslationRecord], int]:
    raw_rows = json.loads(file_path.read_text(encoding="utf-8-sig"))
    rows = []
    for row_id, row in enumerate(raw_rows):
        source_tokens = tokenize(row["english"])
        target_tokens = tokenize(row["vietnamese"])
        if (
            1 <= len(source_tokens) <= max_words
            and 1 <= len(target_tokens) <= max_words
        ):
            rows.append(TranslationRecord(row_id, source_tokens, target_tokens))
    return rows, len(raw_rows)


class TranslationDataset(Dataset):
    def __init__(self, rows: list[TranslationRecord]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> TranslationRecord:
        return self.rows[index]


@dataclass
class Batch:
    source: torch.Tensor
    source_lengths: torch.Tensor
    target: torch.Tensor
    references: list[str]
    row_ids: list[int]

    def to(self, device: torch.device):
        return Batch(
            source=self.source.to(device),
            source_lengths=self.source_lengths,
            target=self.target.to(device),
            references=self.references,
            row_ids=self.row_ids,
        )


def create_collate_fn(source_vocab: Vocabulary, target_vocab: Vocabulary):
    def collate(rows: Sequence[TranslationRecord]) -> Batch:
        source_sequences = [
            torch.tensor(source_vocab.encode(row.source_tokens) + [EOS_ID])
            for row in rows
        ]
        target_sequences = [
            torch.tensor([BOS_ID] + target_vocab.encode(row.target_tokens) + [EOS_ID])
            for row in rows
        ]
        return Batch(
            source=pad_sequence(source_sequences, batch_first=True, padding_value=PAD_ID),
            source_lengths=torch.tensor([len(sequence) for sequence in source_sequences]),
            target=pad_sequence(target_sequences, batch_first=True, padding_value=PAD_ID),
            references=[" ".join(row.target_tokens) for row in rows],
            row_ids=[row.row_id for row in rows],
        )

    return collate


@dataclass
class DataBundle:
    train_rows: list[TranslationRecord]
    dev_rows: list[TranslationRecord]
    test_rows: list[TranslationRecord]
    source_vocab: Vocabulary
    target_vocab: Vocabulary
    counts: dict[str, dict[str, int]]

    def create_loader(
        self,
        split: str,
        batch_size: int,
        shuffle: bool = False,
        seed: int = 42,
        num_workers: int = 0,
    ) -> DataLoader:
        split_rows = {
            "train": self.train_rows,
            "dev": self.dev_rows,
            "test": self.test_rows,
        }
        if split not in split_rows:
            raise ValueError("split phải là train, dev hoặc test")
        generator = torch.Generator().manual_seed(seed)
        return DataLoader(
            TranslationDataset(split_rows[split]),
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=create_collate_fn(self.source_vocab, self.target_vocab),
            generator=generator,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )


def prepare_data(config: Config) -> DataBundle:
    loaded = {
        split: load_split(config.data_dir / f"small-{split}.json", config.max_words)
        for split in ("train", "dev", "test")
    }
    train_rows, _ = loaded["train"]
    source_vocab = build_vocabulary(
        (row.source_tokens for row in train_rows),
        config.vocab_limit,
        config.min_frequency,
    )
    target_vocab = build_vocabulary(
        (row.target_tokens for row in train_rows),
        config.vocab_limit,
        config.min_frequency,
    )
    return DataBundle(
        train_rows=loaded["train"][0],
        dev_rows=loaded["dev"][0],
        test_rows=loaded["test"][0],
        source_vocab=source_vocab,
        target_vocab=target_vocab,
        counts={
            split: {"raw": raw_count, "filtered": len(rows)}
            for split, (rows, raw_count) in loaded.items()
        },
    )
