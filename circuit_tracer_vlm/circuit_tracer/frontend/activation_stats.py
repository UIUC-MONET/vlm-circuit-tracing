from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
from safetensors import safe_open
import torch
from transformers import AutoTokenizer


DEFAULT_ACTIVATION_STATS_REPO = "Jingcheng/gemma3-4b-it-plt-activations"


def cantor_unpair(value: int) -> tuple[int, int]:
    if value < 0:
        raise ValueError("feature index must be non-negative")
    diagonal = (math.isqrt(8 * value + 1) - 1) // 2
    diagonal_start = diagonal * (diagonal + 1) // 2
    feature = value - diagonal_start
    return diagonal - feature, feature


@dataclass(frozen=True)
class _LayerStats:
    feature_ids: torch.Tensor
    slot_offsets: torch.Tensor
    top_values: torch.Tensor
    input_ids: torch.Tensor
    token_positions: torch.Tensor
    firing_counts: torch.Tensor


class ActivationStatsStore:
    """Lazy reader for the public activation-stat artifact used by the frontend."""

    def __init__(
        self,
        repo_id: str = DEFAULT_ACTIVATION_STATS_REPO,
        *,
        revision: str = "main",
        token: str | None = None,
        context_radius: int = 16,
    ) -> None:
        if "@" in repo_id:
            repo_id, revision = repo_id.rsplit("@", 1)
        self.repo_id = repo_id
        self.revision = revision
        self.token = token
        self.context_radius = context_radius
        self._manifest: dict[str, Any] | None = None
        self._layers: dict[int, _LayerStats] = {}
        self._input_tables: dict[int, Any] = {}
        self._tokenizer = None

    def _download(self, filename: str) -> Path:
        return Path(
            hf_hub_download(
                self.repo_id,
                filename,
                revision=self.revision,
                token=self.token,
            )
        )

    @property
    def manifest(self) -> dict[str, Any]:
        if self._manifest is None:
            self._manifest = json.loads(self._download("manifest.json").read_text())
            if self._manifest.get("schema") != "circuit-tracer-activation-stats-v1":
                raise ValueError("Unsupported activation-stat artifact schema")
        return self._manifest

    def _layer(self, layer: int) -> _LayerStats:
        cached = self._layers.get(layer)
        if cached is not None:
            return cached
        entry = next(
            (item for item in self.manifest["layers"] if int(item["layer"]) == layer),
            None,
        )
        if entry is None:
            raise KeyError(f"Activation statistics do not contain layer {layer}")
        path = self._download(entry["filename"])
        with safe_open(path, framework="pt", device="cpu") as source:
            feature_ids = source.get_tensor("feature_ids").to(torch.int64)
            slot_counts = source.get_tensor("slot_counts").to(torch.int64)
            cached = _LayerStats(
                feature_ids=feature_ids,
                slot_offsets=torch.cat(
                    (torch.zeros(1, dtype=torch.int64), slot_counts.cumsum(0))
                ),
                top_values=source.get_tensor("top_values"),
                input_ids=source.get_tensor("input_ids").to(torch.int64),
                token_positions=source.get_tensor("token_positions").to(torch.int64),
                firing_counts=source.get_tensor("firing_counts").to(torch.int64),
            )
        if feature_ids.numel() > 1 and not bool(torch.all(feature_ids[1:] > feature_ids[:-1])):
            raise ValueError(f"Layer {layer} feature IDs are not strictly sorted")
        self._layers[layer] = cached
        return cached

    def _input(self, input_id: int) -> dict[str, Any]:
        entries = self.manifest["inputs"]
        starts = [int(item["start"]) for item in entries]
        file_index = bisect_right(starts, input_id) - 1
        if file_index < 0 or input_id >= int(entries[file_index]["end"]):
            raise KeyError(f"Input {input_id} is outside the activation artifact")
        table = self._input_tables.get(file_index)
        if table is None:
            table = pq.read_table(self._download(entries[file_index]["filename"]))
            self._input_tables[file_index] = table
            while len(self._input_tables) > 4:
                self._input_tables.pop(next(iter(self._input_tables)))
        row = input_id - int(entries[file_index]["start"])
        return {name: table[name][row].as_py() for name in table.column_names}

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.manifest["model_name"],
                revision=self.manifest.get("model_revision"),
                token=self.token,
            )
        return self._tokenizer

    def _render_example(
        self, input_id: int, token_position: int, activation: float
    ) -> dict[str, Any]:
        record = self._input(input_id)
        token_ids = record["token_ids"]
        token_mask = record["token_mask"]
        if not 0 <= token_position < len(token_ids) or not token_mask[token_position]:
            raise ValueError(f"Invalid retained position {token_position} for input {input_id}")
        start = max(0, token_position - self.context_radius)
        end = min(len(token_ids), token_position + self.context_radius + 1)
        positions = [position for position in range(start, end) if token_mask[position]]
        selected = positions.index(token_position)
        tokens = [
            self.tokenizer.decode(
                [token_ids[position]],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            for position in positions
        ]
        activations = [0.0] * len(tokens)
        activations[selected] = activation
        return {
            "activation": activation,
            "tokens_acts_list": activations,
            "train_token_ind": selected,
            "is_repeated_datapoint": False,
            "tokens": tokens,
            "image_references": record["image_references"] or [],
            "example_id": record["example_id"],
            "metadata": json.loads(record["metadata_json"]),
        }

    def feature(self, frontend_index: int) -> dict[str, Any]:
        layer, feature_id = cantor_unpair(frontend_index)
        stats = self._layer(layer)
        row = int(torch.searchsorted(stats.feature_ids, feature_id))
        if row >= stats.feature_ids.numel() or int(stats.feature_ids[row]) != feature_id:
            return self._feature_payload(frontend_index, feature_id, layer, [], 0)
        start, end = int(stats.slot_offsets[row]), int(stats.slot_offsets[row + 1])
        examples = [
            self._render_example(
                int(stats.input_ids[index]),
                int(stats.token_positions[index]),
                float(stats.top_values[index]),
            )
            for index in range(start, end)
        ]
        return self._feature_payload(
            frontend_index,
            feature_id,
            layer,
            examples,
            int(stats.firing_counts[row]),
        )

    def _feature_payload(
        self,
        frontend_index: int,
        feature_id: int,
        layer: int,
        examples: list[dict[str, Any]],
        firing_count: int,
    ) -> dict[str, Any]:
        maximum = max((float(item["activation"]) for item in examples), default=0.0)
        total_tokens = int(self.manifest.get("processor_tokens", 0))
        return {
            "featureIndex": frontend_index,
            "index": feature_id,
            "layer": layer,
            "scan": f"{self.repo_id}@{self.revision}",
            "transcoder_id": self.manifest.get("transcoder", {}).get("fingerprint", ""),
            "top_logits": [],
            "bottom_logits": [],
            "act_min": 0.0,
            "act_max": maximum if maximum > 0 else 1.0,
            "activation_frequency": firing_count / total_tokens if total_tokens else 0.0,
            "firing_count": firing_count,
            "quantile_values": [],
            "histogram": [],
            "examples_quantiles": [
                {
                    "quantile_name": f"Top activations · max {maximum:.4g}",
                    "examples": examples,
                }
            ],
            "isDead": not examples,
        }
