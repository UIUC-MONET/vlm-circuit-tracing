from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import inspect

import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import save_file
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, PreTrainedModel

logger = logging.getLogger(__name__)


@dataclass
class PLTConfig:
    """Configuration for per-layer transcoder training."""

    model_name: str = "google/gemma-3-4b-it"
    dataset: str | None = None
    split: str = "train"
    save_dir: str = "checkpoints/plt"
    layers: list[int] | None = None
    layer_stride: int = 1
    batch_size: int = 1
    grad_acc_steps: int = 1
    max_steps: int = 1000
    save_every: int = 1000
    lr: float = 5e-4
    expansion_factor: int = 32
    num_features: int | None = None
    top_k: int = 48
    skip_connection: bool = False
    max_length: int = 1024
    text_column: str = "text"
    image_column: str | None = "image"
    prompt_template: str = (
        "<start_of_turn>user\n<start_of_image>{text}<end_of_turn>\n"
        "<start_of_turn>model\n"
    )
    dtype: str = "bfloat16"
    device: str | None = None
    trust_remote_code: bool = True
    revision: str | None = None
    hf_token: str | None = None
    num_workers: int = 0
    log_every: int = 10

    @property
    def torch_dtype(self) -> torch.dtype:
        mapping = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
        }
        try:
            return mapping[self.dtype]
        except KeyError as exc:
            raise ValueError(f"Unsupported dtype: {self.dtype}") from exc


class TrainablePLT(nn.Module):
    """Trainable top-k per-layer transcoder exported as a ReLU PLT plus top-k config."""

    def __init__(
        self,
        d_model: int,
        num_features: int,
        top_k: int,
        *,
        skip_connection: bool = False,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_features = num_features
        self.top_k = top_k

        self.W_enc = nn.Parameter(
            torch.empty(num_features, d_model, device=device, dtype=dtype)
        )
        self.b_enc = nn.Parameter(torch.zeros(num_features, device=device, dtype=dtype))
        self.W_dec = nn.Parameter(
            torch.zeros(num_features, d_model, device=device, dtype=dtype)
        )
        self.b_dec = nn.Parameter(torch.zeros(d_model, device=device, dtype=dtype))
        self.W_skip = (
            nn.Parameter(torch.zeros(d_model, d_model, device=device, dtype=dtype))
            if skip_connection
            else None
        )
        nn.init.kaiming_uniform_(self.W_enc, a=5**0.5)

    @property
    def dtype(self) -> torch.dtype:
        return self.W_enc.dtype

    @torch.no_grad()
    def initialize_biases(self, inputs: Tensor, targets: Tensor) -> None:
        target_mean = targets.mean(dim=0).to(self.dtype)
        input_mean = inputs.mean(dim=0).to(self.dtype)
        self.b_dec.copy_(target_mean)
        self.b_enc.copy_(-(input_mean @ self.W_enc.T))

    def encode(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        pre_acts = F.linear(inputs.to(self.dtype), self.W_enc, self.b_enc)
        acts = F.relu(pre_acts)
        k = min(self.top_k, acts.shape[-1])
        return torch.topk(acts, k, dim=-1, sorted=False)

    def decode(self, values: Tensor, indices: Tensor, inputs: Tensor) -> Tensor:
        decoder_vectors = self.W_dec[indices]
        reconstruction = (decoder_vectors * values.unsqueeze(-1).to(self.dtype)).sum(dim=1)
        reconstruction = reconstruction + self.b_dec
        if self.W_skip is not None:
            reconstruction = reconstruction + inputs.to(self.dtype) @ self.W_skip.T
        return reconstruction

    def forward(self, inputs: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
        values, indices = self.encode(inputs)
        reconstruction = self.decode(values, indices, inputs)
        residual = targets.to(reconstruction.dtype) - reconstruction
        denom = (targets - targets.mean(dim=0, keepdim=True)).pow(2).sum().clamp_min(1e-8)
        fvu = residual.float().pow(2).sum() / denom.float()
        return fvu, reconstruction

    def export_state(self) -> dict[str, Tensor]:
        state = {
            "W_enc": self.W_enc.detach().cpu(),
            "W_dec": self.W_dec.detach().cpu(),
            "b_enc": self.b_enc.detach().cpu(),
            "b_dec": self.b_dec.detach().cpu(),
        }
        if self.W_skip is not None:
            state["W_skip"] = self.W_skip.detach().cpu()
        return state


class PLTTrainer:
    """Train per-layer transcoders from live Hugging Face model activations."""

    def __init__(
        self,
        cfg: PLTConfig,
        dataset,
        model: PreTrainedModel,
        processor=None,
    ) -> None:
        self.cfg = cfg
        self.dataset = dataset
        self.model = model
        self.processor = processor
        self.device = torch.device(
            cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dtype = cfg.torch_dtype

        self.model.to(self.device)
        self.model.requires_grad_(False)
        self.model.eval()
        tokenizer = getattr(processor, "tokenizer", None)
        self.pad_token_id = getattr(tokenizer, "pad_token_id", 0) or 0

        self.layers_path, self.layer_modules = find_layer_modules(model)
        layer_ids = cfg.layers
        if layer_ids is None:
            layer_ids = list(range(len(self.layer_modules)))
        self.layers = layer_ids[:: cfg.layer_stride]
        if not self.layers:
            raise ValueError("No layers selected for PLT training.")

        self.hookpoints = [f"{self.layers_path}.{layer}.mlp" for layer in self.layers]
        self.modules = [self.model.get_submodule(name) for name in self.hookpoints]

        widths = infer_module_widths(self.model, self.modules)
        self.plts = nn.ModuleDict()
        for layer, width in zip(self.layers, widths, strict=True):
            num_features = cfg.num_features or width * cfg.expansion_factor
            self.plts[str(layer)] = TrainablePLT(
                width,
                num_features,
                cfg.top_k,
                skip_connection=cfg.skip_connection,
                device=self.device,
                dtype=self.dtype,
            )

        self.optimizer = torch.optim.AdamW(self.plts.parameters(), lr=cfg.lr)
        self.global_step = 0
        self._initialized = {str(layer): False for layer in self.layers}
        self._tokens_mask: Tensor | None = None
        self._losses: dict[str, float] = {}

    def fit(self) -> None:
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            collate_fn=lambda batch: collate_batch(batch, self.pad_token_id),
        )
        data_iter = iter(dataloader)
        pbar = tqdm(total=self.cfg.max_steps, desc="Training PLTs")

        while self.global_step < self.cfg.max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            batch = move_batch_to_device(batch, self.device)
            self._tokens_mask = make_token_mask(batch)
            self._losses = {}

            handles = [
                module.register_forward_hook(self._make_hook(layer, module))
                for layer, module in zip(self.layers, self.modules, strict=True)
            ]
            try:
                self.model(**filter_model_inputs(self.model, batch))
            finally:
                for handle in handles:
                    handle.remove()

            if (self.global_step + 1) % self.cfg.grad_acc_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            self.global_step += 1
            if self.cfg.log_every and self.global_step % self.cfg.log_every == 0:
                loss_str = ", ".join(
                    f"layer {layer}: {loss:.4f}" for layer, loss in sorted(self._losses.items())
                )
                logger.info("step %s: %s", self.global_step, loss_str)
            if self.cfg.save_every and self.global_step % self.cfg.save_every == 0:
                self.save(Path(self.cfg.save_dir))
            pbar.update(1)

        pbar.close()
        self.save(Path(self.cfg.save_dir))

    def _make_hook(self, layer: int, module: nn.Module):
        del module

        def hook(_module: nn.Module, inputs: tuple[Any, ...], output: Any):
            if self._tokens_mask is None:
                raise RuntimeError("Token mask was not set before forward pass.")
            source = unpack_tensor(inputs[0])
            target = unpack_tensor(output)
            if source.ndim != 3 or target.ndim != 3:
                raise ValueError(
                    f"Expected layer {layer} MLP activations with shape [batch, pos, d_model], "
                    f"got {source.shape} and {target.shape}."
                )

            mask = self._tokens_mask
            if mask.shape != source.shape[:2]:
                mask = torch.ones(source.shape[:2], device=source.device, dtype=torch.bool)

            flat_source = source.detach().reshape(-1, source.shape[-1])[mask.reshape(-1)]
            flat_target = target.detach().reshape(-1, target.shape[-1])[mask.reshape(-1)]
            if flat_source.numel() == 0:
                return None

            plt = self.plts[str(layer)]
            if not self._initialized[str(layer)]:
                plt.initialize_biases(flat_source, flat_target)
                self._initialized[str(layer)] = True

            loss, _ = plt(flat_source, flat_target)
            (loss / self.cfg.grad_acc_steps).backward()
            self._losses[str(layer)] = float(loss.detach().cpu())
            return None

        return hook

    def save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for layer, plt in self.plts.items():
            save_file(plt.export_state(), output_dir / f"layer_{layer}.safetensors")

        config = {
            "model_kind": "transcoder_set",
            "model_name": self.cfg.model_name,
            "feature_input_hook": "mlp.hook_in",
            "feature_output_hook": "mlp.hook_out",
            "top_k": self.cfg.top_k,
            "layers": self.layers,
            "num_model_layers": len(self.layer_modules),
            "transcoders": [f"layer_{layer}.safetensors" for layer in self.layers],
            "training": asdict(self.cfg),
        }
        with open(output_dir / "config.yaml", "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)


def train_plt_from_hf(cfg: PLTConfig) -> PLTTrainer:
    try:
        from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
    except ImportError as exc:
        raise ImportError("Install training dependencies with `pip install -e .[train]`.") from exc

    processor = AutoProcessor.from_pretrained(
        cfg.model_name,
        revision=cfg.revision,
        token=cfg.hf_token,
        trust_remote_code=cfg.trust_remote_code,
    )
    model = AutoModel.from_pretrained(
        cfg.model_name,
        revision=cfg.revision,
        token=cfg.hf_token,
        trust_remote_code=cfg.trust_remote_code,
        torch_dtype=cfg.torch_dtype,
    )

    if cfg.dataset is None:
        raise ValueError("A dataset path or Hugging Face dataset name is required.")
    dataset_path = Path(cfg.dataset)
    if dataset_path.exists():
        dataset = load_from_disk(str(dataset_path))
    else:
        dataset = load_dataset(cfg.dataset, split=cfg.split, trust_remote_code=True)
    if isinstance(dataset, DatasetDict):
        dataset = dataset[cfg.split]
    if not isinstance(dataset, Dataset):
        dataset = dataset.with_format("torch")

    if "input_ids" not in dataset.column_names:
        dataset = prepare_dataset(dataset, processor, cfg)
    else:
        dataset = dataset.with_format("torch")

    trainer = PLTTrainer(cfg, dataset, model, processor)
    trainer.fit()
    return trainer


def prepare_dataset(dataset, processor, cfg: PLTConfig):
    remove_columns = list(dataset.column_names)
    has_images = cfg.image_column is not None and cfg.image_column in dataset.column_names

    def tokenize(batch):
        texts = batch[cfg.text_column]
        texts = [format_prompt(text, cfg.prompt_template if has_images else "{text}") for text in texts]
        kwargs = {
            "text": texts,
            "padding": "max_length",
            "max_length": cfg.max_length,
            "truncation": True,
            "return_tensors": "pt",
        }
        if has_images:
            kwargs["images"] = [image.convert("RGB") for image in batch[cfg.image_column]]
        return processor(**kwargs)

    tokenized = dataset.map(
        tokenize,
        batched=True,
        batch_size=cfg.batch_size,
        remove_columns=remove_columns,
    )
    return tokenized.with_format("torch")


def format_prompt(value: Any, template: str) -> str:
    if isinstance(value, dict):
        text = value.get("user") or value.get("prompt") or value.get("text") or str(value)
    else:
        text = str(value)
    return template.format(text=text)


def find_layer_modules(model: PreTrainedModel) -> tuple[str, nn.ModuleList]:
    layer_count = getattr(getattr(model.config, "text_config", None), "num_hidden_layers", None)
    if layer_count is None:
        layer_count = getattr(model.config, "num_hidden_layers", None)

    candidates = []
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and (layer_count is None or len(module) == layer_count):
            if len(module) and hasattr(module[0], "mlp"):
                candidates.append((name, module))
    if not candidates:
        raise ValueError("Could not find transformer layers with MLP modules.")
    candidates.sort(key=lambda item: len(item[0]))
    return candidates[0]


def infer_module_widths(model: PreTrainedModel, modules: list[nn.Module]) -> list[int]:
    hidden_size = getattr(getattr(model.config, "text_config", None), "hidden_size", None)
    if hidden_size is None:
        hidden_size = getattr(model.config, "hidden_size", None)
    if hidden_size is not None:
        return [int(hidden_size)] * len(modules)

    widths = []
    for module in modules:
        linears = [child for child in module.modules() if isinstance(child, nn.Linear)]
        if not linears:
            raise ValueError(
                "Could not infer MLP width from model config or linear submodules."
            )
        widths.append(linears[-1].out_features)
    return widths


def filter_model_inputs(model: PreTrainedModel, batch: dict[str, Any]) -> dict[str, Any]:
    try:
        params = inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return batch
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return batch
    return {key: value for key, value in batch.items() if key in params}


def unpack_tensor(value: Any) -> Tensor:
    if isinstance(value, Tensor):
        return value
    if isinstance(value, tuple):
        return unpack_tensor(value[0])
    raise TypeError(f"Expected tensor or tuple of tensors, got {type(value).__name__}")


def move_batch_to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {key: move_batch_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, list):
        return [move_batch_to_device(value, device) for value in batch]
    return batch


def collate_batch(batch: list[dict[str, Any]], pad_token_id: int) -> dict[str, Any]:
    keys = batch[0].keys()
    collated = {}
    for key in keys:
        values = [item[key] for item in batch]
        if not all(isinstance(value, Tensor) for value in values):
            collated[key] = values
            continue

        tensors = [value for value in values]
        if all(tensor.shape == tensors[0].shape for tensor in tensors):
            collated[key] = torch.stack(tensors)
            continue

        padding_value = 0
        if key == "input_ids":
            padding_value = pad_token_id
        collated[key] = pad_sequence(tensors, batch_first=True, padding_value=padding_value)
    return collated


def make_token_mask(batch: dict[str, Any]) -> Tensor:
    if "attention_mask" in batch:
        return batch["attention_mask"].bool()
    if "input_ids" in batch:
        return torch.ones_like(batch["input_ids"], dtype=torch.bool)
    raise ValueError("Batch must include input_ids or attention_mask.")
