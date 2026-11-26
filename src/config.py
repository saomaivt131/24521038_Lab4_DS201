from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import torch


@dataclass
class Config:
    project_root: Path = Path(__file__).resolve().parent.parent

    # Kiến trúc chung cho cả ba bài
    embedding_dim: int = 256
    hidden_size: int = 256
    num_layers: int = 3
    dropout: float = 0.3

    # Huấn luyện
    batch_size: int = 32
    learning_rate: float = 1e-3
    teacher_forcing_ratio: float = 0.5
    max_epochs: int = 10
    gradient_clip: float = 1.0

    # Dữ liệu
    max_words: int = 50
    vocab_limit: int = 10_000
    min_frequency: int = 2
    max_decode_steps: int = 51

    seed: int = 42
    num_workers: int = 0

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def artifact_dir(self) -> Path:
        return self.project_root / "artifacts"

    @property
    def checkpoint_dir(self) -> Path:
        return self.project_root / "checkpoints"

    def create_output_directories(self) -> None:
        for path in (
            self.artifact_dir / "figures",
            self.artifact_dir / "tables",
            self.checkpoint_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

