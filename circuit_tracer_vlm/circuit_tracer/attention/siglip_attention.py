from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoProcessor

AttentionMethod = Literal["rollout", "similarity"]
PoolReduce = Literal["max", "mean"]
RenderMode = Literal["overlay", "grayscale"]


class SiglipAttentionMapper:
    """Compute image-token attention maps for Gemma-style VLMs with a SigLIP tower.

    Gemma 3 represents an image as a 16x16 grid of language-model image tokens.
    The embedded SigLIP tower usually exposes a finer patch grid, such as 64x64.
    This class computes maps from each pooled image token back onto that finer grid and
    renders them at the original image resolution.
    """

    def __init__(
        self,
        model_id: str = "google/gemma-3-4b-pt",
        device: str | None = None,
        dtype: torch.dtype | None = None,
        last_k_layers: int = 4,
        head_keep_frac: float = 0.30,
        add_identity: bool = True,
        block: int = 4,
        pool_reduce: PoolReduce = "max",
        clip_top_pct: float = 5.0,
        cache_maps: bool = True,
    ) -> None:
        if pool_reduce not in ("max", "mean"):
            raise ValueError("pool_reduce must be 'max' or 'mean'")
        if last_k_layers < 1:
            raise ValueError("last_k_layers must be >= 1")
        if block < 1:
            raise ValueError("block must be >= 1")

        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.last_k_layers = last_k_layers
        self.head_keep_frac = head_keep_frac
        self.add_identity = add_identity
        self.block = block
        self.pool_reduce = pool_reduce
        self.clip_top_pct = float(clip_top_pct)
        self.cache_maps = cache_maps

        self.processor = AutoProcessor.from_pretrained(model_id)
        config = AutoConfig.from_pretrained(model_id)
        for obj in (config, getattr(config, "vision_config", None)):
            if obj is None:
                continue
            if hasattr(obj, "output_attentions"):
                obj.output_attentions = True
            if hasattr(obj, "attn_implementation"):
                obj.attn_implementation = "eager"

        model_kwargs = {"config": config}
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        self.model = AutoModel.from_pretrained(model_id, **model_kwargs).to(self.device).eval()

        self.siglip, self.siglip_cfg = self._find_siglip(self.model)
        if self.siglip_cfg is not None:
            if hasattr(self.siglip_cfg, "output_attentions"):
                self.siglip_cfg.output_attentions = True
            if hasattr(self.siglip_cfg, "attn_implementation"):
                self.siglip_cfg.attn_implementation = "eager"

        self.img_pil: Image.Image | None = None
        self.width: int | None = None
        self.height: int | None = None
        self.hidden: torch.Tensor | None = None
        self.attn_layers: tuple[torch.Tensor, ...] | None = None
        self.num_tokens: int | None = None
        self.has_cls: bool | None = None
        self.grid_h: int | None = None
        self.grid_w: int | None = None
        self.patch_slice: slice | None = None
        self.start_tok: int | None = None
        self.end_tok: int | None = None
        self.num_patches: int | None = None
        self.rowsets: list[torch.Tensor] | None = None
        self._pooled_rollout: torch.Tensor | None = None
        self._pooled_similarity: torch.Tensor | None = None
        self._rollout_maps: list[Image.Image] | None = None
        self._similarity_maps: list[Image.Image] | None = None

    def set_image(self, image: str | Path | Image.Image) -> None:
        """Run the vision tower once for an image and reset per-image caches."""
        self._pooled_rollout = None
        self._pooled_similarity = None
        self._rollout_maps = None
        self._similarity_maps = None

        if isinstance(image, Image.Image):
            self.img_pil = image.convert("RGB")
        else:
            self.img_pil = Image.open(image).convert("RGB")
        self.width, self.height = self.img_pil.size

        with torch.no_grad():
            batch = self.processor(
                images=self.img_pil,
                text=["<start_of_image>"],
                return_tensors="pt",
            )
            pixel_values = batch["pixel_values"].to(
                device=self.device, dtype=next(self.siglip.parameters()).dtype
            )
            vision_out = self.siglip(
                pixel_values=pixel_values,
                output_attentions=True,
                return_dict=True,
            )

        if getattr(vision_out, "attentions", None) is None:
            raise RuntimeError(
                "Vision attentions were not returned. The model may need eager attention support."
            )

        self.hidden = vision_out.last_hidden_state
        self.attn_layers = tuple(vision_out.attentions)
        _, self.num_tokens, _ = self.hidden.shape

        (
            self.has_cls,
            self.grid_h,
            self.grid_w,
            self.patch_slice,
            self.start_tok,
            self.end_tok,
        ) = self._infer_grid(self.num_tokens, pixel_values, self.siglip_cfg)
        self.num_patches = self.grid_h * self.grid_w

        if (self.grid_h % self.block) or (self.grid_w % self.block):
            raise RuntimeError(
                f"Vision grid {self.grid_h}x{self.grid_w} must be divisible by block={self.block}."
            )

        self.rowsets = self._make_rowsets(
            self.grid_h, self.grid_w, self.block, self.hidden.device
        )

    def get_rollout_maps(self, indices: Sequence[int] | None = None) -> list[Image.Image]:
        """Return grayscale attention-rollout maps, one per pooled image token."""
        self._assert_prepared()
        pooled = self._compute_pooled_rollout()
        return self._get_grayscale_maps(pooled, "rollout", indices)

    def get_raw_similarity_maps(self, indices: Sequence[int] | None = None) -> list[Image.Image]:
        """Return grayscale raw hidden-state cosine-similarity maps."""
        self._assert_prepared()
        pooled = self._compute_pooled_similarity()
        return self._get_grayscale_maps(pooled, "similarity", indices)

    def get_maps(
        self, method: AttentionMethod = "rollout", indices: Sequence[int] | None = None
    ) -> list[Image.Image]:
        if method == "rollout":
            return self.get_rollout_maps(indices)
        if method == "similarity":
            return self.get_raw_similarity_maps(indices)
        raise ValueError("method must be 'rollout' or 'similarity'")

    def overlay(
        self,
        maps: Sequence[Image.Image],
        base_image: Image.Image | None = None,
        opacity: float = 0.8,
    ) -> list[Image.Image]:
        """Overlay grayscale maps on the current image using a red-yellow heatmap."""
        self._assert_prepared()
        base = (base_image or self.img_pil).convert("RGB")
        base_np = np.asarray(base).astype(np.float32) / 255.0
        height, width = base_np.shape[:2]

        out: list[Image.Image] = []
        for map_img in maps:
            heat = np.asarray(
                map_img.resize((width, height), Image.BILINEAR), dtype=np.float32
            ) / 255.0
            colored = np.zeros((height, width, 3), dtype=np.float32)
            colored[..., 0] = heat
            colored[..., 1] = np.sqrt(heat) * 0.85
            colored[..., 2] = heat * heat * 0.15
            blended = (1.0 - opacity) * base_np + opacity * colored
            out.append(Image.fromarray((blended.clip(0, 1) * 255).astype(np.uint8)))
        return out

    def save_maps(
        self,
        image: str | Path | Image.Image,
        output_dir: str | Path,
        method: AttentionMethod = "rollout",
        render: RenderMode = "overlay",
        indices: Sequence[int] | None = None,
        opacity: float = 0.8,
        image_format: Literal["jpg", "png"] = "jpg",
        overwrite: bool = False,
    ) -> list[Path]:
        """Compute and save maps as numbered images.

        The default filenames are `0.jpg` through `255.jpg`, which matches the
        local graph frontend's `/emb_image/<ctx_idx>` route.
        """
        self.set_image(image)
        resolved_indices = self._resolve_indices(indices)
        maps = self.get_maps(method, resolved_indices)
        if render == "overlay":
            images = self.overlay(maps, opacity=opacity)
        elif render == "grayscale":
            images = [map_img.convert("RGB") for map_img in maps]
        else:
            raise ValueError("render must be 'overlay' or 'grayscale'")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        suffix = "jpg" if image_format == "jpg" else "png"

        written: list[Path] = []
        for token_idx, img in zip(resolved_indices, images, strict=True):
            path = output_path / f"{token_idx}.{suffix}"
            if path.exists() and not overwrite:
                raise FileExistsError(f"{path} already exists; pass overwrite=True to replace it")
            save_kwargs = {"quality": 95} if suffix == "jpg" else {}
            img.save(path, **save_kwargs)
            written.append(path)

        metadata = {
            "model_id": self.model_id,
            "method": method,
            "render": render,
            "num_maps": len(written),
            "indices": resolved_indices,
            "image_format": suffix,
            "vision_grid": [self.grid_h, self.grid_w],
            "pooled_grid": [self.grid_h // self.block, self.grid_w // self.block],
            "block": self.block,
            "last_k_layers": self.last_k_layers,
            "head_keep_frac": self.head_keep_frac,
            "pool_reduce": self.pool_reduce,
            "clip_top_pct": self.clip_top_pct,
        }
        metadata_path = output_path / "metadata.json"
        if metadata_path.exists() and not overwrite:
            raise FileExistsError(
                f"{metadata_path} already exists; pass overwrite=True to replace it"
            )
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        return written

    @staticmethod
    def _find_siglip(model) -> tuple[torch.nn.Module, object | None]:
        for attr in ("vision_model", "vision_tower", "vision_encoder", "visual"):
            module = getattr(model, attr, None)
            if module is not None and module.__class__.__name__.lower().startswith(
                "siglipvisionmodel"
            ):
                return module, getattr(module, "config", None)
        for _, module in model.named_modules():
            if module.__class__.__name__.lower().startswith("siglipvisionmodel"):
                return module, getattr(module, "config", None)
        raise RuntimeError("SigLIP vision tower not found inside the model.")

    @staticmethod
    def _infer_grid(token_count, pixel_values, cfg) -> tuple[bool, int, int, slice, int, int]:
        root = int(round(token_count**0.5))
        if root * root == token_count:
            return False, root, root, slice(0, token_count), 0, token_count

        patch_count = token_count - 1
        root = int(round(patch_count**0.5))
        if root * root == patch_count:
            return True, root, root, slice(1, token_count), 1, token_count

        _, _, height_px, width_px = pixel_values.shape
        patch = getattr(cfg, "patch_size", None)
        if not patch or (height_px % patch) or (width_px % patch):
            raise RuntimeError("Cannot infer SigLIP token grid from token count and patch size.")
        grid_h, grid_w = height_px // patch, width_px // patch
        has_cls = token_count == grid_h * grid_w + 1
        return (
            has_cls,
            grid_h,
            grid_w,
            slice(1, token_count) if has_cls else slice(0, token_count),
            1 if has_cls else 0,
            token_count,
        )

    @staticmethod
    def _make_rowsets(grid_h: int, grid_w: int, block: int, device) -> list[torch.Tensor]:
        rowsets = []
        for pooled_row in range(grid_h // block):
            for pooled_col in range(grid_w // block):
                indices = []
                row_start = pooled_row * block
                col_start = pooled_col * block
                for row_offset in range(block):
                    for col_offset in range(block):
                        row = row_start + row_offset
                        col = col_start + col_offset
                        indices.append(row * grid_w + col)
                rowsets.append(torch.tensor(indices, dtype=torch.long, device=device))
        return rowsets

    def _pool_rows(self, matrix: torch.Tensor) -> torch.Tensor:
        rows = []
        for indices in self.rowsets:
            block_rows = matrix[indices]
            if self.pool_reduce == "max":
                rows.append(block_rows.max(dim=0, keepdim=True).values)
            else:
                rows.append(block_rows.mean(dim=0, keepdim=True))
        return torch.cat(rows, dim=0)

    def _compute_pooled_rollout(self) -> torch.Tensor:
        if self._pooled_rollout is not None:
            return self._pooled_rollout

        selected_layers = self.attn_layers[-self.last_k_layers :]
        rollout_layers = []
        for attentions in selected_layers:
            layer_attn = attentions[0]
            probs = layer_attn / (layer_attn.sum(-1, keepdim=True) + 1e-6)
            entropy = -(probs * (probs + 1e-6).log()).sum(-1)
            head_scores = entropy.mean(dim=-1)
            num_heads = layer_attn.shape[0]
            keep_count = max(1, int(round(num_heads * self.head_keep_frac)))
            keep_indices = torch.topk(-head_scores, keep_count).indices
            layer_attn = layer_attn[keep_indices].mean(dim=0)
            if self.add_identity:
                identity = torch.eye(
                    layer_attn.size(-1), device=layer_attn.device, dtype=layer_attn.dtype
                )
                layer_attn = layer_attn + identity
            layer_attn = layer_attn / (layer_attn.sum(-1, keepdim=True) + 1e-6)
            rollout_layers.append(layer_attn)

        rollout = torch.eye(self.num_tokens, device=self.hidden.device, dtype=self.hidden.dtype)
        for layer_attn in rollout_layers:
            rollout = layer_attn @ rollout

        patch_rollout = rollout[self.start_tok : self.end_tok, self.patch_slice]
        self._pooled_rollout = self._pool_rows(patch_rollout)
        return self._pooled_rollout

    def _compute_pooled_similarity(self) -> torch.Tensor:
        if self._pooled_similarity is not None:
            return self._pooled_similarity
        hidden = F.normalize(self.hidden[0], dim=-1)
        similarity = hidden @ hidden.T
        patch_similarity = similarity[self.start_tok : self.end_tok, self.patch_slice]
        self._pooled_similarity = self._pool_rows(patch_similarity)
        return self._pooled_similarity

    def _get_grayscale_maps(
        self, pooled: torch.Tensor, cache_name: str, indices: Sequence[int] | None
    ) -> list[Image.Image]:
        cache_attr = "_rollout_maps" if cache_name == "rollout" else "_similarity_maps"
        maps = getattr(self, cache_attr)
        if maps is None:
            maps = self._render_all_grayscale(pooled)
            if self.cache_maps:
                setattr(self, cache_attr, maps)

        resolved_indices = self._resolve_indices(indices)
        return [maps[index] for index in resolved_indices]

    def _render_all_grayscale(
        self, pooled: torch.Tensor, chunk_size: int = 64
    ) -> list[Image.Image]:
        map_count = pooled.size(0)
        maps = pooled.reshape(map_count, self.grid_h, self.grid_w)

        if self.clip_top_pct > 0:
            quantile = 1.0 - (self.clip_top_pct / 100.0)
            thresholds = torch.quantile(
                maps.reshape(map_count, -1).float(), quantile, dim=1, keepdim=True
            ).to(maps.dtype)
            maps = torch.minimum(maps, thresholds.view(map_count, 1, 1))

        min_values = maps.amin(dim=(1, 2), keepdim=True)
        max_values = maps.amax(dim=(1, 2), keepdim=True)
        maps = (maps - min_values) / (max_values - min_values).clamp_min(1e-6)

        rendered: list[Image.Image] = []
        for start in range(0, map_count, chunk_size):
            end = min(start + chunk_size, map_count)
            upsampled = F.interpolate(
                maps[start:end].unsqueeze(1),
                size=(self.height, self.width),
                mode="bilinear",
                align_corners=False,
            )
            arrays = (upsampled.squeeze(1).clamp(0, 1) * 255.0).byte().cpu().numpy()
            rendered.extend(Image.fromarray(arrays[index]) for index in range(arrays.shape[0]))
        return rendered

    def _resolve_indices(self, indices: Sequence[int] | None) -> list[int]:
        map_count = (self.grid_h // self.block) * (self.grid_w // self.block)
        if indices is None:
            return list(range(map_count))

        resolved = []
        for index in indices:
            index = int(index)
            if not 0 <= index < map_count:
                raise IndexError(f"Index {index} out of range [0, {map_count - 1}]")
            resolved.append(index)
        return resolved

    def _assert_prepared(self) -> None:
        if self.img_pil is None or self.hidden is None:
            raise RuntimeError("No image has been set. Call set_image(image) first.")


def save_attention_maps(
    image: str | Path | Image.Image,
    output_dir: str | Path,
    model_id: str = "google/gemma-3-4b-pt",
    method: AttentionMethod = "rollout",
    render: RenderMode = "overlay",
    indices: Sequence[int] | None = None,
    device: str | None = None,
    dtype: torch.dtype | None = None,
    overwrite: bool = False,
    **mapper_kwargs,
) -> list[Path]:
    mapper = SiglipAttentionMapper(
        model_id=model_id, device=device, dtype=dtype, **mapper_kwargs
    )
    return mapper.save_maps(
        image=image,
        output_dir=output_dir,
        method=method,
        render=render,
        indices=indices,
        overwrite=overwrite,
    )
