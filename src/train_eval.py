import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
from rouge_score import rouge_scorer
from torch import nn

from .config import Config, seed_everything
from .data import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    Batch,
    DataBundle,
    TranslationRecord,
    Vocabulary,
    tokenize,
)


class VietnameseTokenizer:
    def tokenize(self, text: str) -> list[str]:
        return tokenize(text)


ROUGE_SCORER = rouge_scorer.RougeScorer(
    ["rougeL"],
    use_stemmer=False,
    tokenizer=VietnameseTokenizer(),
)


@dataclass
class EvaluationResult:
    nll: float
    rougeL_precision: float
    rougeL_recall: float
    rougeL_f1: float
    predictions: list[dict]

    def summary(self) -> dict[str, float]:
        return {
            "nll": self.nll,
            "rougeL_precision": self.rougeL_precision,
            "rougeL_recall": self.rougeL_recall,
            "rougeL_f1": self.rougeL_f1,
        }


def rouge_l(reference: str, hypothesis: str):
    return ROUGE_SCORER.score(reference, hypothesis)["rougeL"]


def train_one_epoch(model, data_loader, optimizer, config, device) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, reduction="sum")
    total_loss = 0.0
    total_tokens = 0

    for batch in data_loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(
            batch.source,
            batch.source_lengths,
            batch.target,
            config.teacher_forcing_ratio,
        )
        gold = batch.target[:, 1:]
        loss_sum = criterion(logits.reshape(-1, logits.size(-1)), gold.reshape(-1))
        token_count = int((gold != PAD_ID).sum().item())
        (loss_sum / max(token_count, 1)).backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()
        total_loss += float(loss_sum.detach().item())
        total_tokens += token_count

    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def greedy_decode(model, source, source_lengths, max_steps: int):
    model.eval()
    memory, state = model.encoder(source, source_lengths)
    source_mask = source != PAD_ID
    current_token = torch.full(
        (source.size(0),),
        BOS_ID,
        dtype=torch.long,
        device=source.device,
    )
    feeding = memory.new_zeros(source.size(0), model.hidden_size)
    finished = torch.zeros(source.size(0), dtype=torch.bool, device=source.device)
    generated = []
    attention_by_step = []

    for _ in range(max_steps):
        logits, state, feeding, weights = model.decoder.step(
            current_token,
            state,
            memory,
            source_mask,
            feeding,
        )
        logits[:, PAD_ID] = float("-inf")
        logits[:, BOS_ID] = float("-inf")
        current_token = logits.argmax(dim=-1)
        current_token = torch.where(
            finished,
            torch.full_like(current_token, EOS_ID),
            current_token,
        )
        generated.append(current_token)
        if weights is not None:
            attention_by_step.append(weights)
        finished |= current_token == EOS_ID
        if bool(finished.all()):
            break

    generated_ids = torch.stack(generated, dim=1)
    attention = (
        torch.stack(attention_by_step, dim=1)
        if attention_by_step
        else None
    )
    return generated_ids, attention


def decode_batch(generated_ids: torch.Tensor, target_vocab: Vocabulary) -> list[str]:
    return [
        " ".join(target_vocab.decode(sequence, stop_at_eos=True))
        for sequence in generated_ids.cpu().tolist()
    ]


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: Iterable[Batch],
    target_vocab: Vocabulary,
    device: torch.device,
    max_decode_steps: int,
) -> EvaluationResult:
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, reduction="sum")
    total_nll = 0.0
    total_tokens = 0
    references, hypotheses, row_ids = [], [], []

    for batch in data_loader:
        batch = batch.to(device)
        logits = model(
            batch.source,
            batch.source_lengths,
            batch.target,
            teacher_forcing_ratio=1.0,
        )
        gold = batch.target[:, 1:]
        total_nll += float(
            criterion(logits.reshape(-1, logits.size(-1)), gold.reshape(-1)).item()
        )
        total_tokens += int((gold != PAD_ID).sum().item())

        generated_ids, _ = greedy_decode(
            model,
            batch.source,
            batch.source_lengths,
            max_decode_steps,
        )
        references.extend(batch.references)
        hypotheses.extend(decode_batch(generated_ids, target_vocab))
        row_ids.extend(batch.row_ids)

    scores = [rouge_l(reference, hypothesis) for reference, hypothesis in zip(references, hypotheses)]
    predictions = [
        {
            "row_id": row_id,
            "reference": reference,
            "hypothesis": hypothesis,
            "rougeL_f1": float(score.fmeasure),
        }
        for row_id, reference, hypothesis, score in zip(row_ids, references, hypotheses, scores)
    ]
    count = len(scores)
    return EvaluationResult(
        nll=total_nll / max(total_tokens, 1),
        rougeL_precision=sum(score.precision for score in scores) / count,
        rougeL_recall=sum(score.recall for score in scores) / count,
        rougeL_f1=sum(score.fmeasure for score in scores) / count,
        predictions=predictions,
    )


def _config_to_dict(config: Config) -> dict:
    values = asdict(config)
    return {key: str(value) if isinstance(value, Path) else value for key, value in values.items()}


def save_checkpoint(path, model, config, data, epoch, dev_result) -> None:
    torch.save(
        {
            "model_state": {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            },
            "mode": model.mode,
            "config": _config_to_dict(config),
            "source_vocab": data.source_vocab.to_dict(),
            "target_vocab": data.target_vocab.to_dict(),
            "epoch": epoch,
            "dev_metrics": dev_result.summary(),
        },
        path,
    )


def write_csv(file_path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fit(
    model: nn.Module,
    data: DataBundle,
    config: Config,
    device: torch.device,
    experiment_name: str,
):
    config.create_output_directories()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    train_loader = data.create_loader(
        "train",
        config.batch_size,
        shuffle=True,
        seed=config.seed,
        num_workers=config.num_workers,
    )
    dev_loader = data.create_loader(
        "dev",
        config.batch_size,
        shuffle=False,
        seed=config.seed,
        num_workers=config.num_workers,
    )
    best_key = (-1.0, float("-inf"))
    best_state = None
    best_epoch = 0
    history = []
    checkpoint_path = config.checkpoint_dir / f"{experiment_name}.pt"

    for epoch in range(1, config.max_epochs + 1):
        seed_everything(config.seed + epoch)
        start = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, optimizer, config, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        dev_result = evaluate(
            model,
            dev_loader,
            data.target_vocab,
            device,
            config.max_decode_steps,
        )
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "dev_nll": dev_result.nll,
            "dev_rougeL": dev_result.rougeL_f1,
            "seconds": seconds,
        }
        history.append(record)
        print(
            f"Epoch {epoch:02d}/{config.max_epochs} | "
            f"train_loss={train_loss:.4f} | dev_nll={dev_result.nll:.4f} | "
            f"dev_ROUGE-L={dev_result.rougeL_f1:.4f} | {seconds:.1f}s"
        )

        current_key = (dev_result.rougeL_f1, -dev_result.nll)
        if current_key > best_key:
            best_key = current_key
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            save_checkpoint(checkpoint_path, model, config, data, epoch, dev_result)
            print("  -> Đã lưu checkpoint tốt nhất")

    if best_state is None:
        raise RuntimeError("Không tạo được checkpoint")
    model.load_state_dict(best_state)
    model.to(device).eval()
    write_csv(config.artifact_dir / "tables" / f"{experiment_name}_history.csv", history)
    return history, best_epoch


@torch.no_grad()
def translate_with_attention(
    model: nn.Module,
    row: TranslationRecord,
    source_vocab: Vocabulary,
    target_vocab: Vocabulary,
    device: torch.device,
    max_steps: int,
) -> dict:
    source_ids = source_vocab.encode(row.source_tokens) + [EOS_ID]
    source = torch.tensor([source_ids], dtype=torch.long, device=device)
    lengths = torch.tensor([len(source_ids)], dtype=torch.long)
    generated_ids, attention = greedy_decode(model, source, lengths, max_steps)
    if attention is None:
        raise ValueError("Mô hình này không có attention")

    generated = generated_ids[0].cpu().tolist()
    output_tokens = []
    for token_id in generated:
        output_tokens.append(target_vocab.id_to_token.get(int(token_id), "<unk>"))
        if int(token_id) == EOS_ID:
            break
    source_tokens = [source_vocab.id_to_token.get(int(token_id), "<unk>") for token_id in source_ids]
    return {
        "hypothesis": " ".join(target_vocab.decode(generated)),
        "source_tokens": source_tokens,
        "output_tokens": output_tokens,
        "attention": attention[0, : len(output_tokens), : len(source_tokens)].cpu(),
    }
