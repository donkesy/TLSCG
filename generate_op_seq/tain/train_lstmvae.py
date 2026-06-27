import argparse
import csv
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from tqdm import tqdm


VULNERABILITY_LABELS = {
    "reentrancy": 0,
    "timestamp": 1,
    "delegatecall": 2,
    "integeroverflow": 3,
}

BASE_PERTURB_MODES = ("local_shuffle", "global_shuffle", "random_swap", "reverse_blocks")


@dataclass
class TrainConfig:
    vocab_size: int = 128
    embedding_size: int = 128
    hidden_size: int = 256
    latent_size: int = 128
    label_embedding_size: int = 32
    num_layers: int = 2
    dropout: float = 0.3
    max_len: int = 500
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-4
    discriminator_lr: float = 1e-4
    kl_weight: float = 0.05
    bigram_weight: float = 0.3
    boundary_weight: float = 0.2
    adv_weight: float = 0.1
    label_center_weight: float = 0.05
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    cls_token_id: int = 3
    data_dir: str = "./dataset/embedding/smart_contract"
    output_dir: str = "./result/vul_ood"
    checkpoint_name: str = "ConditionalLSTMVAE_mix_vulnerabilities_best.pt"
    seed: int = 42
    boundary_samples_per_epoch: int = 1024
    boundary_extrapolation: float = 1.2
    boundary_noise_std: float = 0.15
    generation_max_token_id: int = 56
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ContractSequenceDataset(data.Dataset):
    def __init__(self, sequences, labels, max_len: int = 500):
        self.sequences = torch.as_tensor(sequences, dtype=torch.long)
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.max_len = max_len
        if self.sequences.size(0) != self.labels.size(0):
            raise ValueError("sequences and labels must have the same number of rows")

    def __getitem__(self, index):
        return {
            "input": self.sequences[index, : self.max_len],
            "label": self.labels[index],
        }

    def __len__(self):
        return self.sequences.size(0)


def sequence_nonzero_lengths(sequences: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
    return (sequences != pad_token_id).sum(dim=1)


def sample_target_lengths(seed_sequences: torch.Tensor, batch_size: int, pad_token_id: int = 0) -> torch.Tensor:
    lengths = sequence_nonzero_lengths(seed_sequences, pad_token_id=pad_token_id)
    lengths = lengths[lengths > 0]
    if lengths.numel() == 0:
        return torch.ones(batch_size, dtype=torch.long, device=seed_sequences.device)
    indices = torch.randint(0, lengths.numel(), (batch_size,), device=seed_sequences.device)
    return lengths[indices].long()


def pad_after_target_lengths(
    generated_sequences: torch.Tensor,
    target_lengths: torch.Tensor,
    pad_token_id: int = 0,
    drop_bos: bool = True,
) -> torch.Tensor:
    sequences = generated_sequences[:, 1:] if drop_bos else generated_sequences
    target_lengths = target_lengths.to(sequences.device).clamp(min=1, max=sequences.size(1))
    left_padded = torch.full_like(sequences, pad_token_id)
    for row_index, target_length in enumerate(target_lengths.tolist()):
        left_padded[row_index, -target_length:] = sequences[row_index, :target_length]
    return left_padded


def sequence_distribution_mask(
    sequences: torch.Tensor,
    min_nonzero_len: int,
    max_nonzero_len: int,
    max_token_id: int,
    pad_token_id: int = 0,
) -> torch.Tensor:
    lengths = sequence_nonzero_lengths(sequences, pad_token_id=pad_token_id)
    valid_lengths = (lengths >= min_nonzero_len) & (lengths <= max_nonzero_len)
    valid_tokens = ((sequences >= 0) & (sequences <= max_token_id)).all(dim=1)
    positions = torch.arange(sequences.size(1), device=sequences.device).unsqueeze(0)
    suffix_start = sequences.size(1) - lengths.unsqueeze(1)
    expected_padding = positions < suffix_start
    left_padding = torch.where(expected_padding, sequences == pad_token_id, sequences != pad_token_id).all(dim=1)
    return valid_lengths & valid_tokens & left_padding


class DifferentiableBigramLoss(nn.Module):
    def __init__(self, vocab_size: int, reduction: str = "mean"):
        super().__init__()
        self.vocab_size = vocab_size
        self.loss_fn = nn.MSELoss(reduction=reduction)

    def forward(self, logits, targets, pad_token_id: int = 0):
        target_one_hot = F.one_hot(targets, num_classes=self.vocab_size).float()
        padding_mask = (targets == pad_token_id).unsqueeze(-1)
        target_one_hot = target_one_hot.masked_fill(padding_mask, 0)

        first_token_target = target_one_hot[:, :-1, :]
        second_token_target = target_one_hot[:, 1:, :]
        target_co_occurrence = torch.einsum("btd,btc->dc", first_token_target, second_token_target)
        target_prob = target_co_occurrence / (target_co_occurrence.sum() + 1e-8)

        probs = F.softmax(logits, dim=-1).masked_fill(padding_mask, 0)
        first_token_pred = probs[:, :-1, :]
        second_token_pred = probs[:, 1:, :]
        pred_co_occurrence = torch.einsum("btd,btc->dc", first_token_pred, second_token_pred)
        pred_prob = pred_co_occurrence / (pred_co_occurrence.sum() + 1e-8)

        return self.loss_fn(pred_prob, target_prob)


class ConditionalLSTMVAE(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        embedding_size: int = 128,
        hidden_size: int = 256,
        latent_size: int = 128,
        label_embedding_size: int = 32,
        num_layers: int = 2,
        dropout: float = 0.3,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_labels = num_labels
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.label_embedding_size = label_embedding_size
        self.num_layers = num_layers
        self.pad_token_id = pad_token_id

        recurrent_dropout = dropout if num_layers > 1 else 0.0
        conditional_input_size = embedding_size + label_embedding_size
        latent_condition_size = latent_size + label_embedding_size

        self.token_embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=pad_token_id)
        self.label_embedding = nn.Embedding(num_labels, label_embedding_size)
        self.encoder_lstm = nn.LSTM(
            conditional_input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.fc_mu = nn.Linear(hidden_size, latent_size)
        self.fc_logvar = nn.Linear(hidden_size, latent_size)

        self.latent_to_hidden = nn.Linear(latent_condition_size, hidden_size * num_layers)
        self.latent_to_cell = nn.Linear(latent_condition_size, hidden_size * num_layers)
        self.decoder_lstm = nn.LSTM(
            conditional_input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def _label_context(self, labels: torch.Tensor, seq_len: int) -> torch.Tensor:
        label_emb = self.label_embedding(labels)
        return label_emb.unsqueeze(1).expand(-1, seq_len, -1)

    def encode(self, src: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        token_emb = self.token_embedding(src)
        label_ctx = self._label_context(labels, src.size(1))
        embedded = torch.cat([token_emb, label_ctx], dim=-1)
        _, (hidden, _) = self.encoder_lstm(embedded)
        last_hidden = hidden[-1]
        return self.fc_mu(last_hidden), self.fc_logvar(last_hidden)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, tgt: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        label_emb = self.label_embedding(labels)
        conditioned_z = torch.cat([z, label_emb], dim=-1)
        hidden = self.latent_to_hidden(conditioned_z).view(self.num_layers, z.size(0), self.hidden_size)
        cell = self.latent_to_cell(conditioned_z).view(self.num_layers, z.size(0), self.hidden_size)

        token_emb = self.token_embedding(tgt)
        label_ctx = label_emb.unsqueeze(1).expand(-1, tgt.size(1), -1)
        decoder_input = torch.cat([token_emb, label_ctx], dim=-1)
        output, _ = self.decoder_lstm(decoder_input, (hidden, cell))
        return self.fc_out(output)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, labels: torch.Tensor):
        mu, logvar = self.encode(src, labels)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z, tgt, labels)
        return logits, mu, logvar

    def generate(
        self,
        z: torch.Tensor,
        labels: torch.Tensor,
        max_len: int = 500,
        bos_token_id: int = 1,
        temperature: float = 1.0,
        top_p: float = 0.95,
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()
        batch_size = z.size(0)
        device = z.device
        generated = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)

        label_emb = self.label_embedding(labels)
        conditioned_z = torch.cat([z, label_emb], dim=-1)
        hidden = self.latent_to_hidden(conditioned_z).view(self.num_layers, batch_size, self.hidden_size)
        cell = self.latent_to_cell(conditioned_z).view(self.num_layers, batch_size, self.hidden_size)

        with torch.no_grad():
            current = generated
            for _ in range(max_len - 1):
                token_emb = self.token_embedding(current[:, -1:])
                label_ctx = label_emb.unsqueeze(1)
                decoder_input = torch.cat([token_emb, label_ctx], dim=-1)
                output, (hidden, cell) = self.decoder_lstm(decoder_input, (hidden, cell))
                logits = self.fc_out(output[:, -1, :]) / max(temperature, 1e-6)
                next_token = sample_from_logits(logits, top_p=top_p).unsqueeze(1)
                generated = torch.cat([generated, next_token], dim=1)
                current = generated

        self.train(was_training)
        return generated


class LatentDiscriminator(nn.Module):
    def __init__(self, latent_size: int, hidden_size: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_shifted_batch(
    raw_sequences: torch.Tensor,
    cls_token_id: int = 3,
    bos_token_id: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = raw_sequences.size(0)
    cls = torch.full((batch_size, 1), cls_token_id, dtype=torch.long, device=raw_sequences.device)
    bos = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=raw_sequences.device)
    src = torch.cat([cls, raw_sequences[:, :-1]], dim=1)
    tgt = torch.cat([bos, raw_sequences[:, :-1]], dim=1)
    return src, tgt, raw_sequences


def load_vulnerability_dataset(
    name: str,
    data_dir: str,
    max_len: int,
    limit_per_class: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
    if name == "mix_vulnerabilities":
        selected_names = list(VULNERABILITY_LABELS.keys())
    elif name in VULNERABILITY_LABELS:
        selected_names = [name]
    else:
        selected_names = [name]

    sequences = []
    labels = []
    label_map = {}
    for dataset_name in selected_names:
        path = os.path.join(data_dir, f"{dataset_name}.csv")
        frame = pd.read_csv(path)
        if limit_per_class is not None:
            frame = frame.iloc[:limit_per_class]
        values = frame.values[:, :max_len]
        if dataset_name not in label_map:
            label_map[dataset_name] = len(label_map)
        sequences.append(torch.as_tensor(values, dtype=torch.long))
        labels.append(torch.full((len(values),), label_map[dataset_name], dtype=torch.long))

    return torch.cat(sequences, dim=0), torch.cat(labels, dim=0), label_map


def reconstruction_loss_per_sample(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_token_id: int = 0,
) -> torch.Tensor:
    per_token = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
        ignore_index=pad_token_id,
    ).view(targets.size(0), targets.size(1))
    mask = (targets != pad_token_id).float()
    return (per_token * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def kl_loss_per_sample(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)


def compute_sequence_scores(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    pad_token_id: int = 0,
) -> Dict[str, torch.Tensor]:
    recon_error = reconstruction_loss_per_sample(logits, targets, pad_token_id=pad_token_id)
    kl_error = kl_loss_per_sample(mu, logvar)
    return {
        "recon_error": recon_error,
        "kl_error": kl_error,
        "ood_score": recon_error + kl_error,
    }


def transfer_compatible_weights(source_state_dict: Dict[str, torch.Tensor], target_model: nn.Module):
    """Load only checkpoint tensors whose names and shapes match the target model."""
    target_state = target_model.state_dict()
    compatible_state = {}
    skipped = []

    for name, tensor in source_state_dict.items():
        if name in target_state and target_state[name].shape == tensor.shape:
            compatible_state[name] = tensor
        else:
            skipped.append(name)

    target_state.update(compatible_state)
    target_model.load_state_dict(target_state)
    return {"loaded": sorted(compatible_state.keys()), "skipped": sorted(skipped)}


def load_checkpoint_state(checkpoint_path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def load_trained_lstm_vae(checkpoint_path: str, device: torch.device):
    """Restore a ConditionalLSTMVAE saved by this training script."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(
            "generate-only mode requires a checkpoint saved by train_lstmVAE_ood.py "
            "with model_state_dict, config, and label_map."
        )

    config_values = TrainConfig().__dict__.copy()
    config_values.update(checkpoint.get("config", {}))
    config_values["device"] = str(device)
    config = TrainConfig(**config_values)
    label_map = checkpoint.get("label_map", {"label_0": 0})

    model = ConditionalLSTMVAE(
        vocab_size=config.vocab_size,
        num_labels=len(label_map),
        embedding_size=config.embedding_size,
        hidden_size=config.hidden_size,
        latent_size=config.latent_size,
        label_embedding_size=config.label_embedding_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
        pad_token_id=config.pad_token_id,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config, label_map


@torch.no_grad()
def score_sequences_with_normal_model(
    normal_model: ConditionalLSTMVAE,
    sequences: torch.Tensor,
    cls_token_id: int = 3,
    bos_token_id: int = 1,
    pad_token_id: int = 0,
) -> Dict[str, torch.Tensor]:
    """Score sequences against a VAE trained only on normal contracts."""
    device = next(normal_model.parameters()).device
    normal_model.eval()
    sequences = torch.as_tensor(sequences, dtype=torch.long, device=device)
    normal_labels = torch.zeros(sequences.size(0), dtype=torch.long, device=device)
    src, tgt, real_x = build_shifted_batch(sequences, cls_token_id, bos_token_id)
    logits, mu, logvar = normal_model(src, tgt, normal_labels)
    return compute_sequence_scores(logits, real_x, mu, logvar, pad_token_id=pad_token_id)


def label_center_loss(z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    losses = []
    for label in labels.unique():
        class_z = z[labels == label]
        if class_z.size(0) < 2:
            continue
        center = class_z.mean(dim=0, keepdim=True)
        losses.append(torch.mean((class_z - center).pow(2)))
    if not losses:
        return torch.zeros((), device=z.device)
    return torch.stack(losses).mean()


def sample_boundary_latents(
    vulnerability_z: torch.Tensor,
    labels: torch.Tensor,
    normal_center: Optional[torch.Tensor] = None,
    num_samples: int = 1024,
    extrapolation: float = 1.2,
    noise_std: float = 0.15,
    return_labels: bool = False,
):
    if vulnerability_z.size(0) == 0:
        raise ValueError("vulnerability_z must not be empty")
    device = vulnerability_z.device
    if normal_center is None:
        normal_center = vulnerability_z.mean(dim=0)
    normal_center = normal_center.to(device)

    indices = torch.randint(0, vulnerability_z.size(0), (num_samples,), device=device)
    base = vulnerability_z[indices]
    sampled_labels = labels.to(device)[indices]
    direction = base - normal_center.unsqueeze(0)
    norm = direction.norm(dim=1, keepdim=True).clamp_min(1e-6)
    direction = direction / norm
    radius = norm * extrapolation
    noise = torch.randn_like(base) * noise_std
    samples = base + direction * radius + noise
    if return_labels:
        return samples, sampled_labels
    return samples


def perturb_opcode_sequence(
    sequence: torch.Tensor,
    intensity: float = 1.0,
    mode: str = "local_shuffle",
    window_size: int = 8,
    pad_token_id: int = 0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Build an extreme anomaly by disturbing opcode order while preserving tokens and padding."""
    result = sequence.clone()
    non_pad = sequence[sequence != pad_token_id]
    if non_pad.numel() <= 1:
        return result

    intensity = max(0.0, min(float(intensity), 1.0))
    window_size = max(1, int(window_size))
    active_len = non_pad.numel()

    if mode == "reverse_blocks":
        blocks = [non_pad[index : index + window_size] for index in range(0, active_len, window_size)]
        perturbed = torch.cat(list(reversed(blocks)))
    elif mode == "global_shuffle":
        if intensity <= 0.0:
            perturbed = non_pad
        else:
            perm = torch.randperm(active_len, device=sequence.device, generator=generator)
            shuffled = non_pad[perm]
            change_count = max(1, round(active_len * intensity))
            perturbed = non_pad.clone()
            perturbed[:change_count] = shuffled[:change_count]
    elif mode == "random_swap":
        perturbed = non_pad.clone()
        swap_count = max(1, round(active_len * intensity))
        for _ in range(swap_count):
            pair = torch.randperm(active_len, device=sequence.device, generator=generator)[:2]
            left, right = int(pair[0].item()), int(pair[1].item())
            value = perturbed[left].clone()
            perturbed[left] = perturbed[right]
            perturbed[right] = value
    elif mode == "local_shuffle":
        perturbed_blocks = []
        for block in [non_pad[index : index + window_size] for index in range(0, active_len, window_size)]:
            if block.numel() <= 1 or intensity <= 0.0:
                perturbed_blocks.append(block)
                continue
            perm = torch.randperm(block.numel(), device=sequence.device, generator=generator)
            shuffled = block[perm]
            change_count = max(1, round(block.numel() * intensity))
            mixed = block.clone()
            mixed[:change_count] = shuffled[:change_count]
            perturbed_blocks.append(mixed)
        perturbed = torch.cat(perturbed_blocks)
    else:
        raise ValueError(f"Unknown perturbation mode: {mode}")

    result[:] = pad_token_id
    result[-active_len:] = perturbed
    return result


def expand_perturb_modes(mode: str) -> Tuple[str, ...]:
    if mode == "all":
        return BASE_PERTURB_MODES
    if mode in BASE_PERTURB_MODES:
        return (mode,)
    raise ValueError(f"Unknown perturbation mode: {mode}")


def perturb_opcode_batch(
    sequences: torch.Tensor,
    intensity: float = 1.0,
    mode: str = "local_shuffle",
    window_size: int = 8,
    pad_token_id: int = 0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    modes = expand_perturb_modes(mode)
    return torch.stack(
        [
            perturb_opcode_sequence(
                sequence,
                intensity=intensity,
                mode=modes[index % len(modes)],
                window_size=window_size,
                pad_token_id=pad_token_id,
                generator=generator,
            )
            for index, sequence in enumerate(sequences)
        ],
        dim=0,
    )


def interpolate_latents(
    normal_z: torch.Tensor,
    extreme_z: torch.Tensor,
    alphas: torch.Tensor,
) -> torch.Tensor:
    if normal_z.shape != extreme_z.shape:
        raise ValueError("normal_z and extreme_z must have the same shape")
    alphas = alphas.to(device=normal_z.device, dtype=normal_z.dtype).view(1, -1, 1)
    normal = normal_z.unsqueeze(1)
    extreme = extreme_z.unsqueeze(1)
    return (normal * (1.0 - alphas) + extreme * alphas).reshape(-1, normal_z.size(-1))


def boundary_score_mask(
    scores: torch.Tensor,
    min_score: Optional[float] = None,
    max_score: Optional[float] = None,
) -> torch.Tensor:
    keep = torch.ones(scores.size(0), dtype=torch.bool, device=scores.device)
    if min_score is not None:
        keep &= scores >= min_score
    if max_score is not None:
        keep &= scores <= max_score
    return keep


def build_normal_extreme_sequences(
    normal_sequences: torch.Tensor,
    perturb_mode: str = "local_shuffle",
    perturb_intensity: float = 1.0,
    perturb_window_size: int = 8,
    pad_token_id: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """Build training data from normals plus order-perturbed extremes.

    Returns:
        all_sequences: normal samples followed by all perturbed variants.
        anomaly_sequences: only the perturbed samples (for anomaly-detection training).
        perturb_modes: perturbation mode label per row in anomaly_sequences.
    """
    perturbed_sets = []
    perturb_modes: List[str] = []
    for mode in expand_perturb_modes(perturb_mode):
        perturbed = perturb_opcode_batch(
            normal_sequences,
            intensity=perturb_intensity,
            mode=mode,
            window_size=perturb_window_size,
            pad_token_id=pad_token_id,
        )
        perturbed_sets.append(perturbed)
        perturb_modes.extend([mode] * perturbed.size(0))
    anomaly_sequences = torch.cat(perturbed_sets, dim=0)
    all_sequences = torch.cat([normal_sequences, anomaly_sequences], dim=0)
    return all_sequences, anomaly_sequences, perturb_modes


def save_extreme_anomaly_dataset(
    anomaly_sequences: torch.Tensor,
    output_path: str,
    perturb_modes: Optional[Sequence[str]] = None,
    source_indices: Optional[Sequence[int]] = None,
) -> None:
    """Save order-perturbed anomaly sequences for downstream anomaly-detection training."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    pd.DataFrame(anomaly_sequences.cpu().tolist()).to_csv(output_path, index=False)
    print(f"Saved {anomaly_sequences.size(0)} extreme anomaly samples to {output_path}")

    if perturb_modes is not None or source_indices is not None:
        metadata_path = os.path.splitext(output_path)[0] + "_metadata.csv"
        metadata = {}
        if perturb_modes is not None:
            metadata["perturb_mode"] = list(perturb_modes)
        if source_indices is not None:
            metadata["source_index"] = list(source_indices)
        pd.DataFrame(metadata).to_csv(metadata_path, index=False)
        print(f"Saved anomaly metadata to {metadata_path}")


def export_extreme_anomaly_dataset(
    normal_dataset_name: str = "normal_all",
    output_path: str = "./dataset/embedding/generated_contract/extreme_anomaly_perturbed.csv",
    config: Optional[TrainConfig] = None,
    limit: Optional[int] = None,
    perturb_mode: str = "local_shuffle",
    perturb_intensity: float = 1.0,
    perturb_window_size: int = 8,
) -> Tuple[torch.Tensor, List[str]]:
    """Build and persist order-perturbed anomalies without running VAE training."""
    config = config or TrainConfig()
    set_seed(config.seed)
    normal_sequences = load_single_dataset(normal_dataset_name, config.data_dir, config.max_len, limit=limit)
    _, anomaly_sequences, perturb_modes = build_normal_extreme_sequences(
        normal_sequences,
        perturb_mode=perturb_mode,
        perturb_intensity=perturb_intensity,
        perturb_window_size=perturb_window_size,
        pad_token_id=config.pad_token_id,
    )
    num_normals = normal_sequences.size(0)
    source_indices = [index % num_normals for index in range(anomaly_sequences.size(0))]
    save_extreme_anomaly_dataset(
        anomaly_sequences,
        output_path,
        perturb_modes=perturb_modes,
        source_indices=source_indices,
    )
    return anomaly_sequences, perturb_modes


@torch.no_grad()
def encode_dataset_latents(
    model: ConditionalLSTMVAE,
    loader: data.DataLoader,
    device: torch.device,
    config: TrainConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_mu = []
    all_labels = []
    for batch in loader:
        raw = batch["input"].to(device)
        labels = batch["label"].to(device)
        src, _, _ = build_shifted_batch(raw, config.cls_token_id, config.bos_token_id)
        mu, _ = model.encode(src, labels)
        all_mu.append(mu.cpu())
        all_labels.append(labels.cpu())
    return torch.cat(all_mu, dim=0), torch.cat(all_labels, dim=0)


def sample_from_logits(logits: torch.Tensor, top_p: float = 0.95) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_remove = cumulative_probs > top_p
    sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
    sorted_remove[..., 0] = False
    remove = torch.zeros_like(sorted_remove).scatter(1, sorted_indices, sorted_remove)
    probs = probs.masked_fill(remove, 0.0)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return torch.multinomial(probs, num_samples=1).squeeze(1)


def train(
    dataset_name: str = "mix_vulnerabilities",
    config: Optional[TrainConfig] = None,
    limit_per_class: Optional[int] = None,
    init_checkpoint_path: Optional[str] = None,
) -> ConditionalLSTMVAE:
    config = config or TrainConfig()
    set_seed(config.seed)
    device = torch.device(config.device)
    os.makedirs(config.output_dir, exist_ok=True)

    sequences, labels, label_map = load_vulnerability_dataset(
        dataset_name,
        data_dir=config.data_dir,
        max_len=config.max_len,
        limit_per_class=limit_per_class,
    )
    dataset = ContractSequenceDataset(sequences, labels, max_len=config.max_len)
    loader = data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model = ConditionalLSTMVAE(
        vocab_size=config.vocab_size,
        num_labels=len(label_map),
        embedding_size=config.embedding_size,
        hidden_size=config.hidden_size,
        latent_size=config.latent_size,
        label_embedding_size=config.label_embedding_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
        pad_token_id=config.pad_token_id,
    ).to(device)
    if init_checkpoint_path:
        source_state = load_checkpoint_state(init_checkpoint_path, device)
        report = transfer_compatible_weights(source_state, model)
        print(
            f"Loaded {len(report['loaded'])} compatible tensors from {init_checkpoint_path}; "
            f"skipped {len(report['skipped'])} tensors."
        )
    discriminator = LatentDiscriminator(config.latent_size, config.hidden_size).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    d_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=config.discriminator_lr)
    criterion_bigram = DifferentiableBigramLoss(config.vocab_size).to(device)

    best_loss = float("inf")
    checkpoint_path = os.path.join(config.output_dir, config.checkpoint_name)

    for epoch in range(config.num_epochs):
        model.train()
        discriminator.train()
        epoch_total = 0.0
        progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{config.num_epochs}", unit="batch")

        for batch in progress:
            raw = batch["input"].to(device)
            batch_labels = batch["label"].to(device)
            src, tgt, real_x = build_shifted_batch(raw, config.cls_token_id, config.bos_token_id)

            logits, mu, logvar = model(src, tgt, batch_labels)
            z = model.reparameterize(mu, logvar)

            recon = reconstruction_loss_per_sample(logits, real_x, config.pad_token_id).mean()
            kl = kl_loss_per_sample(mu, logvar).mean()
            bigram = criterion_bigram(logits, real_x, pad_token_id=config.pad_token_id)
            center = label_center_loss(mu, batch_labels)

            with torch.no_grad():
                normal_center = torch.zeros(config.latent_size, device=device)
                boundary_z, _ = sample_boundary_latents(
                    mu.detach(),
                    batch_labels,
                    normal_center=normal_center,
                    num_samples=raw.size(0),
                    extrapolation=config.boundary_extrapolation,
                    noise_std=config.boundary_noise_std,
                    return_labels=True,
                )

            d_optimizer.zero_grad()
            real_logits = discriminator(z.detach())
            boundary_logits = discriminator(boundary_z.detach())
            d_loss = 0.5 * (
                F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
                + F.binary_cross_entropy_with_logits(boundary_logits, torch.zeros_like(boundary_logits))
            )
            d_loss.backward()
            d_optimizer.step()

            adv_logits = discriminator(z)
            adv = F.binary_cross_entropy_with_logits(adv_logits, torch.ones_like(adv_logits))

            # Encourage encoded vulnerability samples to remain away from the normal prior center.
            boundary_margin = F.relu(1.0 - mu.norm(dim=1)).mean()
            loss = (
                recon
                + config.kl_weight * kl
                + config.bigram_weight * bigram
                + config.label_center_weight * center
                + config.boundary_weight * boundary_margin
                + config.adv_weight * adv
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_total += loss.item()
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                recon=f"{recon.item():.4f}",
                kl=f"{kl.item():.4f}",
                d=f"{d_loss.item():.4f}",
            )

        avg_loss = epoch_total / max(len(loader), 1)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config.__dict__,
                    "label_map": label_map,
                    "best_loss": best_loss,
                },
                checkpoint_path,
            )
            print(f"Saved best checkpoint to {checkpoint_path} (loss={best_loss:.4f})")

    return model


def load_single_dataset(
    dataset_name: str,
    data_dir: str,
    max_len: int,
    limit: Optional[int] = None,
) -> torch.Tensor:
    path = os.path.join(data_dir, f"{dataset_name}.csv")
    frame = pd.read_csv(path)
    if limit is not None:
        frame = frame.iloc[:limit]
    return torch.as_tensor(frame.values[:, :max_len], dtype=torch.long)


def load_length_reference_dataset(
    dataset_name: str,
    data_dir: str,
    max_len: int,
    limit: Optional[int] = None,
) -> torch.Tensor:
    if dataset_name == "mix_vulnerabilities" or dataset_name in VULNERABILITY_LABELS:
        sequences, _, _ = load_vulnerability_dataset(
            dataset_name,
            data_dir,
            max_len,
            limit_per_class=limit,
        )
        return sequences
    return load_single_dataset(dataset_name, data_dir, max_len, limit=limit)


def train_normal_vae(
    normal_dataset_name: str = "normal_all",
    config: Optional[TrainConfig] = None,
    limit: Optional[int] = None,
) -> Tuple[ConditionalLSTMVAE, str]:
    """Pretrain a VAE only on normal contracts; keep this model for anomaly scoring."""
    config = config or TrainConfig()
    set_seed(config.seed)
    device = torch.device(config.device)
    os.makedirs(config.output_dir, exist_ok=True)

    sequences = load_single_dataset(normal_dataset_name, config.data_dir, config.max_len, limit=limit)
    labels = torch.zeros(sequences.size(0), dtype=torch.long)
    loader = data.DataLoader(
        ContractSequenceDataset(sequences, labels, config.max_len),
        batch_size=config.batch_size,
        shuffle=True,
    )

    model = ConditionalLSTMVAE(
        vocab_size=config.vocab_size,
        num_labels=1,
        embedding_size=config.embedding_size,
        hidden_size=config.hidden_size,
        latent_size=config.latent_size,
        label_embedding_size=config.label_embedding_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
        pad_token_id=config.pad_token_id,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion_bigram = DifferentiableBigramLoss(config.vocab_size).to(device)
    best_loss = float("inf")
    checkpoint_path = os.path.join(config.output_dir, "NormalLSTMVAE_best.pt")

    for epoch in range(config.num_epochs):
        model.train()
        epoch_total = 0.0
        progress = tqdm(loader, desc=f"Normal pretrain {epoch + 1}/{config.num_epochs}", unit="batch")

        for batch in progress:
            raw = batch["input"].to(device)
            batch_labels = batch["label"].to(device)
            src, tgt, real_x = build_shifted_batch(raw, config.cls_token_id, config.bos_token_id)
            logits, mu, logvar = model(src, tgt, batch_labels)

            recon = reconstruction_loss_per_sample(logits, real_x, config.pad_token_id).mean()
            kl = kl_loss_per_sample(mu, logvar).mean()
            bigram = criterion_bigram(logits, real_x, pad_token_id=config.pad_token_id)
            loss = recon + config.kl_weight * kl + config.bigram_weight * bigram

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_total += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}", recon=f"{recon.item():.4f}", kl=f"{kl.item():.4f}")

        avg_loss = epoch_total / max(len(loader), 1)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config.__dict__,
                    "label_map": {normal_dataset_name: 0},
                    "role": "normal_vae",
                    "best_loss": best_loss,
                },
                checkpoint_path,
            )
            print(f"Saved normal checkpoint to {checkpoint_path} (loss={best_loss:.4f})")

    return model, checkpoint_path


def train_interpolation_vae(
    normal_dataset_name: str = "normal_all",
    config: Optional[TrainConfig] = None,
    limit: Optional[int] = None,
    init_checkpoint_path: Optional[str] = None,
    perturb_mode: str = "local_shuffle",
    perturb_intensity: float = 1.0,
    perturb_window_size: int = 8,
    anomaly_output_path: Optional[str] = None,
) -> Tuple[ConditionalLSTMVAE, str]:
    """Finetune a VAE on normal samples plus order-perturbed extremes for latent interpolation."""
    config = config or TrainConfig()
    set_seed(config.seed)
    device = torch.device(config.device)
    os.makedirs(config.output_dir, exist_ok=True)

    normal_sequences = load_single_dataset(normal_dataset_name, config.data_dir, config.max_len, limit=limit)
    train_sequences, anomaly_sequences, perturb_modes = build_normal_extreme_sequences(
        normal_sequences,
        perturb_mode=perturb_mode,
        perturb_intensity=perturb_intensity,
        perturb_window_size=perturb_window_size,
        pad_token_id=config.pad_token_id,
    )
    if anomaly_output_path:
        num_normals = normal_sequences.size(0)
        source_indices = [index % num_normals for index in range(anomaly_sequences.size(0))]
        save_extreme_anomaly_dataset(
            anomaly_sequences,
            anomaly_output_path,
            perturb_modes=perturb_modes,
            source_indices=source_indices,
        )
    labels = torch.zeros(train_sequences.size(0), dtype=torch.long)
    loader = data.DataLoader(
        ContractSequenceDataset(train_sequences, labels, config.max_len),
        batch_size=config.batch_size,
        shuffle=True,
    )

    model = ConditionalLSTMVAE(
        vocab_size=config.vocab_size,
        num_labels=1,
        embedding_size=config.embedding_size,
        hidden_size=config.hidden_size,
        latent_size=config.latent_size,
        label_embedding_size=config.label_embedding_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
        pad_token_id=config.pad_token_id,
    ).to(device)
    if init_checkpoint_path:
        report = transfer_compatible_weights(load_checkpoint_state(init_checkpoint_path, device), model)
        print(
            f"Initialized interpolation VAE from {init_checkpoint_path}: "
            f"loaded={len(report['loaded'])}, skipped={len(report['skipped'])}"
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion_bigram = DifferentiableBigramLoss(config.vocab_size).to(device)
    best_loss = float("inf")
    checkpoint_path = os.path.join(config.output_dir, "InterpolationLSTMVAE_best.pt")

    for epoch in range(config.num_epochs):
        model.train()
        epoch_total = 0.0
        progress = tqdm(loader, desc=f"Interpolation finetune {epoch + 1}/{config.num_epochs}", unit="batch")

        for batch in progress:
            raw = batch["input"].to(device)
            batch_labels = batch["label"].to(device)
            src, tgt, real_x = build_shifted_batch(raw, config.cls_token_id, config.bos_token_id)
            logits, mu, logvar = model(src, tgt, batch_labels)

            recon = reconstruction_loss_per_sample(logits, real_x, config.pad_token_id).mean()
            kl = kl_loss_per_sample(mu, logvar).mean()
            bigram = criterion_bigram(logits, real_x, pad_token_id=config.pad_token_id)
            loss = recon + config.kl_weight * kl + config.bigram_weight * bigram

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_total += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}", recon=f"{recon.item():.4f}", kl=f"{kl.item():.4f}")

        avg_loss = epoch_total / max(len(loader), 1)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config.__dict__,
                    "label_map": {"normal_extreme_interpolation": 0},
                    "role": "interpolation_vae",
                    "perturb_mode": perturb_mode,
                    "perturb_intensity": perturb_intensity,
                    "perturb_window_size": perturb_window_size,
                    "best_loss": best_loss,
                },
                checkpoint_path,
            )
            print(f"Saved interpolation checkpoint to {checkpoint_path} (loss={best_loss:.4f})")

    return model, checkpoint_path


def filter_sequences_by_normal_score(
    normal_model: ConditionalLSTMVAE,
    sequences: torch.Tensor,
    config: TrainConfig,
    min_ood_score: Optional[float] = None,
    max_ood_score: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    scores = score_sequences_with_normal_model(
        normal_model,
        sequences,
        cls_token_id=config.cls_token_id,
        bos_token_id=config.bos_token_id,
        pad_token_id=config.pad_token_id,
    )
    keep = torch.ones(sequences.size(0), dtype=torch.bool, device=scores["ood_score"].device)
    if min_ood_score is not None:
        keep &= scores["ood_score"] >= min_ood_score
    if max_ood_score is not None:
        keep &= scores["ood_score"] <= max_ood_score
    filtered_scores = {name: value[keep] for name, value in scores.items()}
    return sequences[keep.cpu()], filtered_scores


def collect_seed_sequences(seed_loader: data.DataLoader, device: torch.device) -> torch.Tensor:
    sequences = []
    for batch in seed_loader:
        sequences.append(batch["input"].to(device))
    return torch.cat(sequences, dim=0)


@torch.no_grad()
def generate_interpolated_boundary_dataset(
    generation_model: ConditionalLSTMVAE,
    seed_loader: data.DataLoader,
    output_path: str,
    config: TrainConfig,
    num_samples: int = 5000,
    interpolation_steps: int = 5,
    min_alpha: float = 0.25,
    max_alpha: float = 0.85,
    perturb_mode: str = "local_shuffle",
    perturb_intensity: float = 1.0,
    perturb_window_size: int = 8,
    min_normal_ood_score: Optional[float] = None,
    max_normal_ood_score: Optional[float] = None,
    temperature: float = 1.2,
    top_p: float = 0.95,
    normal_score_model: Optional[ConditionalLSTMVAE] = None,
    length_reference_sequences: Optional[torch.Tensor] = None,
) -> None:
    """Generate boundary samples by interpolating normal and order-perturbed latent codes."""
    device = next(generation_model.parameters()).device
    score_model = normal_score_model or generation_model
    generation_model.eval()
    score_model.eval()

    seed_sequences = collect_seed_sequences(seed_loader, device)
    length_reference_sequences = (
        seed_sequences
        if length_reference_sequences is None
        else torch.as_tensor(length_reference_sequences, dtype=torch.long, device=device)[:, : config.max_len]
    )
    reference_lengths = sequence_nonzero_lengths(length_reference_sequences, config.pad_token_id)
    valid_reference_lengths = reference_lengths[reference_lengths > 0]
    if valid_reference_lengths.numel() == 0:
        raise ValueError("length reference must contain at least one non-empty sequence")
    min_reference_len = int(valid_reference_lengths.min().item())
    max_reference_len = int(valid_reference_lengths.max().item())

    interpolation_steps = max(1, int(interpolation_steps))
    alphas = torch.linspace(float(min_alpha), float(max_alpha), steps=interpolation_steps, device=device)
    accepted_rows = []
    score_rows = []
    attempts = 0
    max_attempts = max(num_samples * 20, config.batch_size)
    seed_iterator = iter(seed_loader)

    while len(accepted_rows) < num_samples and attempts < max_attempts:
        try:
            batch = next(seed_iterator)
        except StopIteration:
            seed_iterator = iter(seed_loader)
            batch = next(seed_iterator)

        raw = batch["input"].to(device)
        batch_labels = torch.zeros(raw.size(0), dtype=torch.long, device=device)
        extreme = perturb_opcode_batch(
            raw,
            intensity=perturb_intensity,
            mode=perturb_mode,
            window_size=perturb_window_size,
            pad_token_id=config.pad_token_id,
        )

        normal_src, _, _ = build_shifted_batch(raw, config.cls_token_id, config.bos_token_id)
        extreme_src, _, _ = build_shifted_batch(extreme, config.cls_token_id, config.bos_token_id)
        normal_mu, _ = generation_model.encode(normal_src, batch_labels)
        extreme_mu, _ = generation_model.encode(extreme_src, batch_labels)
        z = interpolate_latents(normal_mu, extreme_mu, alphas)
        generated_labels = torch.zeros(z.size(0), dtype=torch.long, device=device)

        samples = generation_model.generate(
            z,
            generated_labels,
            max_len=config.max_len + 1,
            bos_token_id=config.bos_token_id,
            temperature=temperature,
            top_p=top_p,
        )
        target_lengths = sample_target_lengths(
            length_reference_sequences,
            z.size(0),
            config.pad_token_id,
        )
        samples = pad_after_target_lengths(
            samples,
            target_lengths,
            pad_token_id=config.pad_token_id,
            drop_bos=True,
        ).cpu()
        distribution_keep = sequence_distribution_mask(
            samples,
            min_nonzero_len=min_reference_len,
            max_nonzero_len=max_reference_len,
            max_token_id=config.generation_max_token_id,
            pad_token_id=config.pad_token_id,
        )
        samples = samples[distribution_keep]
        attempts += z.size(0)
        if samples.numel() == 0:
            continue

        scores = score_sequences_with_normal_model(
            score_model,
            samples,
            cls_token_id=config.cls_token_id,
            bos_token_id=config.bos_token_id,
            pad_token_id=config.pad_token_id,
        )
        score_keep = boundary_score_mask(
            scores["ood_score"],
            min_score=min_normal_ood_score,
            max_score=max_normal_ood_score,
        ).cpu()
        kept = samples[score_keep]
        kept_scores = scores["ood_score"].detach().cpu()[score_keep]
        accepted_rows.extend(kept.tolist())
        score_rows.extend(kept_scores.tolist())

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    pd.DataFrame(accepted_rows[:num_samples]).to_csv(output_path, index=False)
    score_path = os.path.splitext(output_path)[0] + "_normal_scores.csv"
    pd.DataFrame({"normal_ood_score": score_rows[:num_samples]}).to_csv(score_path, index=False)
    print(f"Saved {min(len(accepted_rows), num_samples)} interpolated boundary samples to {output_path}")


def generate_filtered_boundary_dataset(
    vulnerability_model: ConditionalLSTMVAE,
    normal_model: ConditionalLSTMVAE,
    seed_loader: data.DataLoader,
    output_path: str,
    config: TrainConfig,
    num_samples: int = 5000,
    min_normal_ood_score: Optional[float] = None,
    max_normal_ood_score: Optional[float] = None,
    temperature: float = 1.2,
    top_p: float = 0.95,
) -> None:
    """Generate vulnerability-boundary samples and keep those that look abnormal to Normal-VAE."""
    device = next(vulnerability_model.parameters()).device
    latent_mu, latent_labels = encode_dataset_latents(vulnerability_model, seed_loader, device, config)
    seed_sequences = collect_seed_sequences(seed_loader, device)
    seed_lengths = sequence_nonzero_lengths(seed_sequences, config.pad_token_id)
    min_seed_len = int(seed_lengths.min().item())
    max_seed_len = int(seed_lengths.max().item())
    accepted_rows = []
    score_rows = []
    attempts = 0
    max_attempts = max(num_samples * 20, config.batch_size)

    while len(accepted_rows) < num_samples and attempts < max_attempts:
        batch_size = min(config.batch_size, num_samples - len(accepted_rows))
        z, label_indices = sample_boundary_latents(
            latent_mu.to(device),
            latent_labels.to(device),
            normal_center=torch.zeros(config.latent_size, device=device),
            num_samples=batch_size,
            extrapolation=config.boundary_extrapolation,
            noise_std=config.boundary_noise_std,
            return_labels=True,
        )
        samples = vulnerability_model.generate(
            z,
            label_indices,
            max_len=config.max_len,
            bos_token_id=config.bos_token_id,
            temperature=temperature,
            top_p=top_p,
        )
        target_lengths = sample_target_lengths(seed_sequences, batch_size, config.pad_token_id)
        samples = pad_after_target_lengths(
            samples,
            target_lengths,
            pad_token_id=config.pad_token_id,
            drop_bos=True,
        ).cpu()
        distribution_keep = sequence_distribution_mask(
            samples,
            min_nonzero_len=min_seed_len,
            max_nonzero_len=max_seed_len,
            max_token_id=config.generation_max_token_id,
            pad_token_id=config.pad_token_id,
        )
        samples = samples[distribution_keep]
        if samples.numel() == 0:
            attempts += batch_size
            continue
        kept, scores = filter_sequences_by_normal_score(
            normal_model,
            samples,
            config,
            min_ood_score=min_normal_ood_score,
            max_ood_score=max_normal_ood_score,
        )
        accepted_rows.extend(kept.tolist())
        score_rows.extend(scores["ood_score"].detach().cpu().tolist())
        attempts += batch_size

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    pd.DataFrame(accepted_rows[:num_samples]).to_csv(output_path, index=False)
    score_path = os.path.splitext(output_path)[0] + "_normal_scores.csv"
    pd.DataFrame({"normal_ood_score": score_rows[:num_samples]}).to_csv(score_path, index=False)
    print(f"Saved {min(len(accepted_rows), num_samples)} filtered boundary samples to {output_path}")


def run_two_stage_pipeline(
    normal_dataset_name: str = "normal_all",
    vulnerability_dataset_name: str = "mix_vulnerabilities",
    config: Optional[TrainConfig] = None,
    normal_epochs: Optional[int] = None,
    vulnerability_epochs: Optional[int] = None,
    normal_limit: Optional[int] = None,
    vulnerability_limit_per_class: Optional[int] = None,
    generated_output_path: Optional[str] = None,
    num_generated: int = 5000,
    min_normal_ood_score: Optional[float] = None,
    max_normal_ood_score: Optional[float] = None,
) -> Tuple[ConditionalLSTMVAE, ConditionalLSTMVAE]:
    """Normal-VAE defines normality; vulnerability-CVAE learns known vulnerability directions."""
    config = config or TrainConfig()
    normal_config = clone_config(config, num_epochs=normal_epochs) if normal_epochs is not None else config
    vulnerability_config = (
        clone_config(config, num_epochs=vulnerability_epochs) if vulnerability_epochs is not None else config
    )
    normal_model, normal_checkpoint = train_normal_vae(normal_dataset_name, normal_config, limit=normal_limit)
    vulnerability_model = train(
        vulnerability_dataset_name,
        config=vulnerability_config,
        limit_per_class=vulnerability_limit_per_class,
        init_checkpoint_path=normal_checkpoint,
    )

    if generated_output_path:
        sequences, labels, _ = load_vulnerability_dataset(
            vulnerability_dataset_name,
            vulnerability_config.data_dir,
            vulnerability_config.max_len,
            limit_per_class=vulnerability_limit_per_class,
        )
        loader = data.DataLoader(
            ContractSequenceDataset(sequences, labels, vulnerability_config.max_len),
            batch_size=vulnerability_config.batch_size,
            shuffle=True,
        )
        generate_filtered_boundary_dataset(
            vulnerability_model,
            normal_model,
            loader,
            generated_output_path,
            vulnerability_config,
            num_samples=num_generated,
            min_normal_ood_score=min_normal_ood_score,
            max_normal_ood_score=max_normal_ood_score,
        )

    return normal_model, vulnerability_model


def clone_config(config: TrainConfig, **updates) -> TrainConfig:
    values = config.__dict__.copy()
    values.update({key: value for key, value in updates.items() if value is not None})
    return TrainConfig(**values)


@torch.no_grad()
def export_latent_features(
    model: ConditionalLSTMVAE,
    loader: data.DataLoader,
    output_path: str,
    config: TrainConfig,
) -> None:
    device = next(model.parameters()).device
    model.eval()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        header = ["label", "recon_error", "kl_error", "ood_score"]
        header.extend(f"z_{i}" for i in range(config.latent_size))
        writer.writerow(header)

        for batch in loader:
            raw = batch["input"].to(device)
            labels = batch["label"].to(device)
            src, tgt, real_x = build_shifted_batch(raw, config.cls_token_id, config.bos_token_id)
            logits, mu, logvar = model(src, tgt, labels)
            scores = compute_sequence_scores(logits, real_x, mu, logvar, config.pad_token_id)
            for row_idx in range(raw.size(0)):
                row = [
                    labels[row_idx].item(),
                    scores["recon_error"][row_idx].item(),
                    scores["kl_error"][row_idx].item(),
                    scores["ood_score"][row_idx].item(),
                ]
                row.extend(mu[row_idx].detach().cpu().tolist())
                writer.writerow(row)


def generate_boundary_dataset(
    model: ConditionalLSTMVAE,
    seed_loader: data.DataLoader,
    output_path: str,
    config: TrainConfig,
    num_samples: int = 5000,
    temperature: float = 1.2,
    top_p: float = 0.95,
) -> None:
    device = next(model.parameters()).device
    latent_mu, latent_labels = encode_dataset_latents(model, seed_loader, device, config)
    seed_sequences = collect_seed_sequences(seed_loader, device)
    seed_lengths = sequence_nonzero_lengths(seed_sequences, config.pad_token_id)
    min_seed_len = int(seed_lengths.min().item())
    max_seed_len = int(seed_lengths.max().item())
    generated_rows = []
    model.eval()

    attempts = 0
    max_attempts = max(num_samples * 20, config.batch_size)
    while len(generated_rows) < num_samples and attempts < max_attempts:
        remaining = num_samples - len(generated_rows)
        batch_size = min(config.batch_size, remaining)
        z = sample_boundary_latents(
            latent_mu.to(device),
            latent_labels.to(device),
            normal_center=torch.zeros(config.latent_size, device=device),
            num_samples=batch_size,
            extrapolation=config.boundary_extrapolation,
            noise_std=config.boundary_noise_std,
        )
        label_indices = torch.randint(0, model.num_labels, (batch_size,), device=device)
        samples = model.generate(
            z,
            label_indices,
            max_len=config.max_len,
            bos_token_id=config.bos_token_id,
            temperature=temperature,
            top_p=top_p,
        )
        target_lengths = sample_target_lengths(seed_sequences, batch_size, config.pad_token_id)
        samples = pad_after_target_lengths(
            samples,
            target_lengths,
            pad_token_id=config.pad_token_id,
            drop_bos=True,
        ).cpu()
        distribution_keep = sequence_distribution_mask(
            samples,
            min_nonzero_len=min_seed_len,
            max_nonzero_len=max_seed_len,
            max_token_id=config.generation_max_token_id,
            pad_token_id=config.pad_token_id,
        )
        generated_rows.extend(samples[distribution_keep].tolist())
        attempts += batch_size

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    pd.DataFrame(generated_rows[:num_samples]).to_csv(output_path, index=False)
    print(f"Saved {min(len(generated_rows), num_samples)} boundary anomaly samples to {output_path}")


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="Train a conditional LSTM-VAE for OOD vulnerability features.")
    parser.add_argument("--dataset", default="mix_vulnerabilities")
    parser.add_argument("--two-stage", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--interpolate-anomaly", action="store_true")
    parser.add_argument("--train-interpolation-vae", action="store_true")
    parser.add_argument("--normal-dataset", default="normal_all")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--normal-epochs", type=int, default=None)
    parser.add_argument("--finetune-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit-per-class", type=int, default=None)
    parser.add_argument("--normal-limit", type=int, default=None)
    parser.add_argument("--output-dir", default="./result/lstm_ood_vae")
    parser.add_argument("--generate-boundary-csv", default=None)
    parser.add_argument(
        "--save-extreme-anomaly-csv",
        default=None,
        help="Save order-perturbed anomaly sequences built by build_normal_extreme_sequences.",
    )
    parser.add_argument(
        "--export-extreme-anomaly-only",
        action="store_true",
        help="Only build and save order-perturbed anomalies; skip VAE training.",
    )
    parser.add_argument("--num-generated", type=int, default=5000)
    parser.add_argument("--min-normal-ood-score", type=float, default=None)
    parser.add_argument("--max-normal-ood-score", type=float, default=None)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--normal-checkpoint", default=None)
    parser.add_argument("--vulnerability-checkpoint", default=None)
    parser.add_argument("--interpolation-checkpoint", default=None)
    parser.add_argument("--length-reference-dataset", default=None)
    parser.add_argument("--length-reference-limit", type=int, default=None)
    parser.add_argument("--interpolation-steps", type=int, default=5)
    parser.add_argument("--min-alpha", type=float, default=0.25)
    parser.add_argument("--max-alpha", type=float, default=0.85)
    parser.add_argument(
        "--perturb-mode",
        default="all",
        choices=["all", *BASE_PERTURB_MODES],
    )
    parser.add_argument("--perturb-intensity", type=float, default=1.0)
    parser.add_argument("--perturb-window-size", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    config = TrainConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )
    if args.export_extreme_anomaly_only:
        if not args.save_extreme_anomaly_csv:
            raise ValueError("--export-extreme-anomaly-only requires --save-extreme-anomaly-csv")
        export_extreme_anomaly_dataset(
            normal_dataset_name=args.normal_dataset,
            output_path=args.save_extreme_anomaly_csv,
            config=config,
            limit=args.normal_limit,
            perturb_mode=args.perturb_mode,
            perturb_intensity=args.perturb_intensity,
            perturb_window_size=args.perturb_window_size,
        )
        return

    if args.train_interpolation_vae:
        interpolation_model, interpolation_checkpoint = train_interpolation_vae(
            normal_dataset_name=args.normal_dataset,
            config=config,
            limit=args.normal_limit,
            init_checkpoint_path=args.normal_checkpoint,
            perturb_mode=args.perturb_mode,
            perturb_intensity=args.perturb_intensity,
            perturb_window_size=args.perturb_window_size,
            anomaly_output_path=args.save_extreme_anomaly_csv,
        )
        if args.generate_boundary_csv:
            device = torch.device(config.device)
            normal_score_model = interpolation_model
            if args.normal_checkpoint:
                normal_score_model, _, _ = load_trained_lstm_vae(args.normal_checkpoint, device)
            sequences = load_single_dataset(
                args.normal_dataset,
                config.data_dir,
                config.max_len,
                limit=args.normal_limit,
            )
            labels = torch.zeros(sequences.size(0), dtype=torch.long)
            loader = data.DataLoader(
                ContractSequenceDataset(sequences, labels, config.max_len),
                batch_size=config.batch_size,
                shuffle=True,
            )
            length_reference_sequences = None
            if args.length_reference_dataset:
                length_reference_sequences = load_length_reference_dataset(
                    args.length_reference_dataset,
                    config.data_dir,
                    config.max_len,
                    limit=args.length_reference_limit,
                )
            generate_interpolated_boundary_dataset(
                interpolation_model,
                loader,
                args.generate_boundary_csv,
                config,
                num_samples=args.num_generated,
                interpolation_steps=args.interpolation_steps,
                min_alpha=args.min_alpha,
                max_alpha=args.max_alpha,
                perturb_mode=args.perturb_mode,
                perturb_intensity=args.perturb_intensity,
                perturb_window_size=args.perturb_window_size,
                min_normal_ood_score=args.min_normal_ood_score,
                max_normal_ood_score=args.max_normal_ood_score,
                normal_score_model=normal_score_model,
                length_reference_sequences=length_reference_sequences,
            )
        else:
            print(f"Interpolation VAE checkpoint: {interpolation_checkpoint}")
        return

    if args.interpolate_anomaly:
        if not args.normal_checkpoint:
            raise ValueError("--interpolate-anomaly requires --normal-checkpoint")
        if not args.generate_boundary_csv:
            raise ValueError("--interpolate-anomaly requires --generate-boundary-csv")

        device = torch.device(config.device)
        normal_model, loaded_config, _ = load_trained_lstm_vae(args.normal_checkpoint, device)
        generation_model = normal_model
        if args.interpolation_checkpoint:
            generation_model, loaded_config, _ = load_trained_lstm_vae(args.interpolation_checkpoint, device)
        loaded_config.batch_size = args.batch_size
        loaded_config.output_dir = args.output_dir

        sequences = load_single_dataset(
            args.normal_dataset,
            loaded_config.data_dir,
            loaded_config.max_len,
            limit=args.normal_limit,
        )
        labels = torch.zeros(sequences.size(0), dtype=torch.long)
        loader = data.DataLoader(
            ContractSequenceDataset(sequences, labels, loaded_config.max_len),
            batch_size=loaded_config.batch_size,
            shuffle=True,
        )
        length_reference_sequences = None
        if args.length_reference_dataset:
            length_reference_sequences = load_length_reference_dataset(
                args.length_reference_dataset,
                loaded_config.data_dir,
                loaded_config.max_len,
                limit=args.length_reference_limit,
            )
        generate_interpolated_boundary_dataset(
            generation_model,
            loader,
            args.generate_boundary_csv,
            loaded_config,
            num_samples=args.num_generated,
            interpolation_steps=args.interpolation_steps,
            min_alpha=args.min_alpha,
            max_alpha=args.max_alpha,
            perturb_mode=args.perturb_mode,
            perturb_intensity=args.perturb_intensity,
            perturb_window_size=args.perturb_window_size,
            min_normal_ood_score=args.min_normal_ood_score,
            max_normal_ood_score=args.max_normal_ood_score,
            normal_score_model=normal_model,
            length_reference_sequences=length_reference_sequences,
        )
        return

    if args.generate_only:
        if not args.vulnerability_checkpoint:
            raise ValueError("--generate-only requires --vulnerability-checkpoint")
        if not args.generate_boundary_csv:
            raise ValueError("--generate-only requires --generate-boundary-csv")

        device = torch.device(config.device)
        vulnerability_model, loaded_config, _ = load_trained_lstm_vae(args.vulnerability_checkpoint, device)
        loaded_config.batch_size = args.batch_size
        loaded_config.output_dir = args.output_dir

        sequences, labels, _ = load_vulnerability_dataset(
            args.dataset,
            loaded_config.data_dir,
            loaded_config.max_len,
            limit_per_class=args.limit_per_class,
        )
        loader = data.DataLoader(
            ContractSequenceDataset(sequences, labels, loaded_config.max_len),
            batch_size=loaded_config.batch_size,
            shuffle=True,
        )

        if args.normal_checkpoint:
            normal_model, _, _ = load_trained_lstm_vae(args.normal_checkpoint, device)
            generate_filtered_boundary_dataset(
                vulnerability_model,
                normal_model,
                loader,
                args.generate_boundary_csv,
                loaded_config,
                num_samples=args.num_generated,
                min_normal_ood_score=args.min_normal_ood_score,
                max_normal_ood_score=args.max_normal_ood_score,
            )
        else:
            generate_boundary_dataset(
                vulnerability_model,
                loader,
                args.generate_boundary_csv,
                loaded_config,
                num_samples=args.num_generated,
            )
        return

    if args.two_stage:
        run_two_stage_pipeline(
            normal_dataset_name=args.normal_dataset,
            vulnerability_dataset_name=args.dataset,
            config=config,
            normal_epochs=args.normal_epochs,
            vulnerability_epochs=args.finetune_epochs,
            normal_limit=args.normal_limit,
            vulnerability_limit_per_class=args.limit_per_class,
            generated_output_path=args.generate_boundary_csv,
            num_generated=args.num_generated,
            min_normal_ood_score=args.min_normal_ood_score,
            max_normal_ood_score=args.max_normal_ood_score,
        )
        return

    model = train(
        args.dataset,
        config=config,
        limit_per_class=args.limit_per_class,
        init_checkpoint_path=args.init_checkpoint,
    )
    if args.generate_boundary_csv:
        sequences, labels, _ = load_vulnerability_dataset(
            args.dataset,
            config.data_dir,
            config.max_len,
            limit_per_class=args.limit_per_class,
        )
        loader = data.DataLoader(
            ContractSequenceDataset(sequences, labels, config.max_len),
            batch_size=config.batch_size,
            shuffle=True,
        )
        generate_boundary_dataset(
            model,
            loader,
            args.generate_boundary_csv,
            config,
            num_samples=args.num_generated,
        )


if __name__ == "__main__":
    main()


# """
# python generated_smart_contracts\train\train_lstmVAE_ood.py --dataset mix_vulnerabilities --epochs 150 --batch-size 32 --generate-boundary-csv dataset\embedding\generated_contract\generated_ood_vulnerabilities.csv --num-generated 5000

# python generated_smart_contracts\train\train_lstmVAE_ood.py --two-stage --normal-dataset normal_all --dataset mix_vulnerabilities --normal-epochs 100 --finetune-epochs 80 --batch-size 128 --output-dir result\lstm_ood_vae_two_stage --generate-boundary-csv dataset\embedding\generated_contract\generated_two_stage_ood.csv --num-generated 5000 --min-normal-ood-score 5.0

# python generated_smart_contracts\train\train_lstmVAE_ood.py --generate-only --dataset mix_vulnerabilities --vulnerability-checkpoint result\lstm_ood_vae_two_stage\ConditionalLSTMVAE_mix_vulnerabilities_best.pt --normal-checkpoint result\lstm_ood_vae_two_stage\NormalLSTMVAE_best.pt --generate-boundary-csv dataset\embedding\generated_contract\generated_two_stage_ood.csv --num-generated 5000 --batch-size 128

# python generated_smart_contracts\train\train_lstmVAE_ood.py --train-interpolation-vae --normal-checkpoint result\lstm_ood_vae_two_stage\NormalLSTMVAE_best.pt --normal-dataset normal_all --epochs 50 --batch-size 128 --output-dir result\lstm_ood_vae_interpolation --perturb-mode all --perturb-window-size 16 --save-extreme-anomaly-csv dataset\embedding\generated_contract\extreme_anomaly_perturbed.csv

# python generated_smart_contracts\train\train_lstmVAE_ood.py --export-extreme-anomaly-only --normal-dataset normal_all --perturb-mode all --perturb-window-size 16 --save-extreme-anomaly-csv dataset\embedding\generated_contract\extreme_anomaly_perturbed.csv

# python generated_smart_contracts\train\train_lstmVAE_ood.py --interpolate-anomaly --normal-checkpoint result\lstm_ood_vae_two_stage\NormalLSTMVAE_best.pt --interpolation-checkpoint result\lstm_ood_vae_interpolation\InterpolationLSTMVAE_best.pt --normal-dataset normal_all --length-reference-dataset mix_vulnerabilities --generate-boundary-csv dataset\embedding\generated_contract\generated_interpolated_ood.csv --num-generated 5000 --batch-size 128 --interpolation-steps 5 --min-alpha 0.35 --max-alpha 0.85 --perturb-mode all --perturb-window-size 16 --min-normal-ood-score 0.3 --max-normal-ood-score 2.5

# python generated_smart_contracts\train\train_lstmVAE_ood.py `
#   --export-extreme-anomaly-only `
#   --normal-dataset normal_all `
#   --perturb-mode all --perturb-window-size 16 `
#   --save-extreme-anomaly-csv dataset\embedding\generated_contract\extreme_anomaly_perturbed.csv
# """
