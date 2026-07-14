from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_IMAGE_PROMPT = """<start_of_turn>user
<start_of_image>{text}<end_of_turn>
<start_of_turn>model
"""


@dataclass
class DatasetSourceConfig:
    """One dataset stream used for PLT training."""

    path: str
    split: str = "train"
    weight: int = 1
    name: str | None = None
    text_column: str | None = "text"
    image_column: str | None = "image"
    prompt_template: str = DEFAULT_IMAGE_PROMPT
    max_length: int = 1024
    packing: bool = False
    map_batch_size: int = 64
    padding: bool | str = "max_length"
    truncation: bool = True
    apply_prompt_without_image: bool = False
    trust_remote_code: bool = True

    def validate(self) -> None:
        if not self.path:
            raise ValueError("Dataset source path must not be empty.")
        if self.weight <= 0:
            raise ValueError(f"Dataset source weight must be positive, got {self.weight}.")
        if self.max_length <= 0:
            raise ValueError(f"Dataset max_length must be positive, got {self.max_length}.")
        if self.map_batch_size <= 0:
            raise ValueError(
                f"Dataset map_batch_size must be positive, got {self.map_batch_size}."
            )
        if self.padding not in (True, False, "longest", "max_length", "do_not_pad"):
            raise ValueError(f"Unsupported padding value: {self.padding!r}.")
        if self.packing and self.text_column is None:
            raise ValueError("Packed text datasets require a text_column.")
        if self.packing and self.image_column is not None:
            raise ValueError("Packed text datasets cannot have an image_column.")


def load_dataset_config(path: str | Path) -> list[DatasetSourceConfig]:
    """Load and validate YAML containing a non-empty sources list."""

    config_path = Path(path)
    with config_path.open() as file:
        payload = yaml.safe_load(file)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("sources"), list):
        raise ValueError(f"{config_path} must contain a top-level sources list.")
    if not payload["sources"]:
        raise ValueError(f"{config_path} contains no dataset sources.")

    sources = []
    for index, value in enumerate(payload["sources"]):
        if not isinstance(value, Mapping):
            raise ValueError(f"Dataset source {index} must be a mapping.")
        try:
            source = DatasetSourceConfig(**dict(value))
        except TypeError as exc:
            raise ValueError(f"Invalid dataset source {index}: {exc}") from exc
        source.validate()
        sources.append(source)
    return sources


def load_and_prepare_datasets(
    sources: Sequence[DatasetSourceConfig],
    processor,
    *,
    hf_token: str | None = None,
):
    """Load each source independently and convert it to model-ready tensors."""

    try:
        from datasets import DatasetDict, load_dataset, load_from_disk
    except ImportError as exc:
        raise ImportError("Install training dependencies with pip install -e .[train].") from exc

    prepared = []
    for source in sources:
        source.validate()
        dataset_path = Path(source.path)
        if dataset_path.exists():
            dataset = load_from_disk(str(dataset_path))
        else:
            dataset = load_dataset(
                source.path,
                name=source.name,
                split=source.split,
                token=hf_token,
                trust_remote_code=source.trust_remote_code,
            )
        if isinstance(dataset, DatasetDict):
            dataset = dataset[source.split]
        if "input_ids" not in dataset.column_names:
            dataset = prepare_dataset(dataset, processor, source)
        else:
            dataset = dataset.with_format("torch")
        prepared.append(dataset)
    return prepared


def prepare_dataset(dataset, processor, source: DatasetSourceConfig):
    """Tokenize one image, conversation, or packed-text dataset source."""

    if source.packing:
        return prepare_packed_text_dataset(dataset, processor, source)

    columns = list(dataset.column_names)
    has_images = source.image_column is not None and source.image_column in columns
    if source.text_column is None and not has_images:
        raise ValueError(
            f"Dataset {source.path!r} contains neither configured text nor image data."
        )
    if source.text_column is not None and source.text_column not in columns:
        raise ValueError(
            f"Dataset {source.path!r} does not contain text column "
            f"{source.text_column!r}; available columns: {columns}."
        )

    def tokenize(batch):
        batch_size = len(next(iter(batch.values())))
        values = (
            batch[source.text_column]
            if source.text_column is not None
            else [""] * batch_size
        )
        template = (
            source.prompt_template
            if has_images or source.apply_prompt_without_image
            else "{text}"
        )
        texts = [format_prompt(value, template) for value in values]
        kwargs = {
            "text": texts,
            "padding": source.padding,
            "max_length": source.max_length,
            "truncation": source.truncation,
            "return_tensors": "pt",
        }
        if has_images:
            kwargs["images"] = [
                image.convert("RGB") for image in batch[source.image_column]
            ]
        return processor(**kwargs)

    tokenized = dataset.map(
        tokenize,
        batched=True,
        batch_size=source.map_batch_size,
        remove_columns=columns,
    )
    return tokenized.with_format("torch")


def prepare_packed_text_dataset(dataset, processor, source: DatasetSourceConfig):
    """Pack documents into full-length token blocks separated by EOS tokens."""

    tokenizer = getattr(processor, "tokenizer", processor)
    separator = tokenizer.eos_token or "<|endoftext|>"
    columns = list(dataset.column_names)
    if source.text_column not in columns:
        raise ValueError(
            f"Dataset {source.path!r} does not contain text column "
            f"{source.text_column!r}; available columns: {columns}."
        )

    def tokenize(batch):
        texts = [str(value) for value in batch[source.text_column]]
        joined = separator.join([""] + texts)
        input_ids = tokenizer(joined, add_special_tokens=False)["input_ids"]
        block_count = len(input_ids) // source.max_length
        blocks = [
            input_ids[index * source.max_length : (index + 1) * source.max_length]
            for index in range(block_count)
        ]
        return {
            "input_ids": blocks,
            "attention_mask": [[1] * source.max_length for _ in blocks],
        }

    tokenized = dataset.map(
        tokenize,
        batched=True,
        batch_size=source.map_batch_size,
        remove_columns=columns,
    )
    return tokenized.with_format("torch")


def format_prompt(value: Any, template: str) -> str:
    """Format plain text or user/assistant conversation records."""

    fields = {"text": "", "user": "", "assistant": "", "prompt": ""}
    if isinstance(value, Mapping):
        fields.update({key: str(item) for key, item in value.items()})
        fields["text"] = str(
            value.get("text") or value.get("user") or value.get("prompt") or value
        )
        fields["user"] = str(value.get("user") or value.get("prompt") or fields["text"])
        fields["assistant"] = str(value.get("assistant") or "")
    else:
        fields["text"] = str(value)
        fields["user"] = str(value)
    try:
        return template.format(**fields)
    except KeyError as exc:
        raise ValueError(f"Unknown prompt template field: {exc.args[0]!r}.") from exc


class WeightedBatchIterator:
    """Cycle dataloaders forever according to an exact integer-weight schedule."""

    def __init__(self, dataloaders: Sequence[Iterable], weights: Sequence[int]):
        if not dataloaders:
            raise ValueError("At least one dataloader is required.")
        if len(dataloaders) != len(weights):
            raise ValueError("Each dataloader must have one weight.")
        if any(weight <= 0 for weight in weights):
            raise ValueError("Dataloader weights must all be positive.")

        self.dataloaders = list(dataloaders)
        self.schedule = [
            index for index, weight in enumerate(weights) for _ in range(weight)
        ]
        self._iterators = [iter(dataloader) for dataloader in self.dataloaders]
        self._position = 0

    def __iter__(self):
        return self

    def __next__(self):
        source_index = self.schedule[self._position]
        self._position = (self._position + 1) % len(self.schedule)
        try:
            return next(self._iterators[source_index])
        except StopIteration:
            self._iterators[source_index] = iter(self.dataloaders[source_index])
            try:
                return next(self._iterators[source_index])
            except StopIteration as exc:
                raise RuntimeError(f"Dataset source {source_index} is empty.") from exc
