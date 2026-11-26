from pathlib import Path

import matplotlib.pyplot as plt
import torch


def plot_training_history(history: list[dict], output_path: Path, title: str) -> None:
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], marker="o")
    axes[0].set(title=f"{title} - Train loss", xlabel="Epoch", ylabel="Cross-entropy")
    axes[1].plot(
        epochs,
        [row["dev_rougeL"] for row in history],
        marker="o",
        color="crimson",
    )
    axes[1].set(title=f"{title} - Dev ROUGE-L F1", xlabel="Epoch", ylabel="ROUGE-L F1")
    axes[1].set_ylim(bottom=0)
    for axis in axes:
        axis.grid(alpha=0.3)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.show()
    plt.close(figure)


def plot_attention_heatmap(
    attention: torch.Tensor,
    source_tokens: list[str],
    output_tokens: list[str],
    title: str,
    output_path: Path,
) -> None:
    matrix = attention.numpy()
    if matrix.shape != (len(output_tokens), len(source_tokens)):
        raise ValueError("Shape attention không khớp nhãn token")

    width = min(max(8.0, 0.42 * len(source_tokens) + 3), 20.0)
    height = min(max(4.0, 0.38 * len(output_tokens) + 2), 18.0)
    figure, axis = plt.subplots(figsize=(width, height))
    image = axis.imshow(matrix, aspect="auto", interpolation="nearest", cmap="Blues", vmin=0.0)
    axis.set_xticks(range(len(source_tokens)), source_tokens, rotation=55, ha="right")
    axis.set_yticks(range(len(output_tokens)), output_tokens)
    axis.set(xlabel="Câu nguồn tiếng Anh", ylabel="Câu dịch tiếng Việt", title=title)
    figure.colorbar(image, ax=axis, label="Attention weight", fraction=0.025, pad=0.02)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.show()
    plt.close(figure)
