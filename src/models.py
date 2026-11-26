import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .config import Config
from .data import PAD_ID


MODEL_LABELS = {
    "none": "Không attention",
    "bahdanau": "Bahdanau",
    "luong": "Luong",
}


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, config: Config):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, config.embedding_dim, padding_idx=PAD_ID)
        self.dropout = nn.Dropout(config.dropout)
        self.lstm = nn.LSTM(
            config.embedding_dim,
            config.hidden_size,
            config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )

    def forward(self, source: torch.Tensor, source_lengths: torch.Tensor):
        embedded = self.dropout(self.embedding(source))
        packed = pack_padded_sequence(
            embedded,
            source_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_memory, state = self.lstm(packed)
        memory, _ = pad_packed_sequence(
            packed_memory,
            batch_first=True,
            total_length=source.size(1),
        )
        return memory, state


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.query_layer = nn.Linear(hidden_size, hidden_size, bias=False)
        self.memory_layer = nn.Linear(hidden_size, hidden_size, bias=False)
        self.score_layer = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, query, memory, source_mask):
        energy = torch.tanh(
            self.query_layer(query).unsqueeze(1) + self.memory_layer(memory)
        )
        scores = self.score_layer(energy).squeeze(-1)
        scores = scores.masked_fill(~source_mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), memory).squeeze(1)
        return context, weights


class LuongGeneralAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.memory_layer = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, query, memory, source_mask):
        projected_memory = self.memory_layer(memory)
        scores = torch.bmm(projected_memory, query.unsqueeze(2)).squeeze(2)
        scores = scores.masked_fill(~source_mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), memory).squeeze(1)
        return context, weights


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, config: Config, mode: str):
        super().__init__()
        if mode not in MODEL_LABELS:
            raise ValueError(f"Mode không hợp lệ: {mode}")
        self.mode = mode
        self.embedding = nn.Embedding(vocab_size, config.embedding_dim, padding_idx=PAD_ID)
        self.dropout = nn.Dropout(config.dropout)

        # Khởi tạo attention trước LSTM để giữ cùng thứ tự khởi tạo
        # tham số với phiên bản thí nghiệm ban đầu.
        if mode == "bahdanau":
            self.attention = BahdanauAttention(config.hidden_size)
        elif mode == "luong":
            self.attention = LuongGeneralAttention(config.hidden_size)
        else:
            self.attention = None

        input_size = config.embedding_dim
        if mode != "none":
            input_size += config.hidden_size

        self.lstm = nn.LSTM(
            input_size,
            config.hidden_size,
            config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )

        self.fusion = (
            nn.Linear(2 * config.hidden_size, config.hidden_size)
            if mode != "none"
            else None
        )
        self.output_layer = nn.Linear(config.hidden_size, vocab_size)

    def step(self, token, state, memory, source_mask, feeding):
        embedded = self.dropout(self.embedding(token))
        weights = None

        if self.mode == "bahdanau":
            context, weights = self.attention(state[0][-1], memory, source_mask)
            decoder_input = torch.cat([embedded, context], dim=-1)
        elif self.mode == "luong":
            decoder_input = torch.cat([embedded, feeding], dim=-1)
        else:
            decoder_input = embedded

        output, new_state = self.lstm(decoder_input.unsqueeze(1), state)
        query = output[:, 0, :]

        if self.mode == "luong":
            context, weights = self.attention(query, memory, source_mask)

        if self.mode != "none":
            # Giữ đúng thứ tự ghép của từng kiến trúc đã dùng trong thí nghiệm.
            pair = (
                torch.cat([query, context], dim=-1)
                if self.mode == "bahdanau"
                else torch.cat([context, query], dim=-1)
            )
            feeding = torch.tanh(self.fusion(pair))
            query = feeding

        logits = self.output_layer(self.dropout(query))
        return logits, new_state, feeding, weights


class Seq2Seq(nn.Module):
    def __init__(
        self,
        mode: str,
        source_vocab_size: int,
        target_vocab_size: int,
        config: Config,
    ):
        super().__init__()
        self.mode = mode
        self.hidden_size = config.hidden_size
        self.encoder = Encoder(source_vocab_size, config)
        self.decoder = Decoder(target_vocab_size, config, mode)

    def forward(
        self,
        source: torch.Tensor,
        source_lengths: torch.Tensor,
        target: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
        return_attention: bool = False,
    ):
        memory, state = self.encoder(source, source_lengths)
        source_mask = source != PAD_ID
        current_token = target[:, 0]
        feeding = memory.new_zeros(source.size(0), self.hidden_size)
        logits_by_step = []
        attention_by_step = []

        for time_step in range(1, target.size(1)):
            logits, state, feeding, weights = self.decoder.step(
                current_token,
                state,
                memory,
                source_mask,
                feeding,
            )
            logits_by_step.append(logits)
            if weights is not None:
                attention_by_step.append(weights)

            predicted = logits.argmax(dim=-1).detach()
            if teacher_forcing_ratio >= 1.0:
                current_token = target[:, time_step]
            elif teacher_forcing_ratio <= 0.0:
                current_token = predicted
            else:
                use_teacher = torch.rand(source.size(0), device=source.device) < teacher_forcing_ratio
                current_token = torch.where(use_teacher, target[:, time_step], predicted)

        all_logits = torch.stack(logits_by_step, dim=1)
        if return_attention:
            all_attention = (
                torch.stack(attention_by_step, dim=1)
                if attention_by_step
                else None
            )
            return all_logits, all_attention
        return all_logits


def build_model(
    mode: str,
    source_vocab_size: int,
    target_vocab_size: int,
    config: Config,
) -> Seq2Seq:
    return Seq2Seq(mode, source_vocab_size, target_vocab_size, config)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
