import json

import pyarrow as pa
import pyarrow.parquet as pq
from safetensors.torch import save_file
import torch

from circuit_tracer.frontend.activation_stats import ActivationStatsStore, cantor_unpair


class _Tokenizer:
    def decode(self, token_ids, **_kwargs):
        return f"<{token_ids[0]}>"


def test_cantor_unpair_uses_layer_then_feature():
    for layer, feature in ((0, 0), (2, 7), (33, 163_839)):
        paired = (layer + feature) * (layer + feature + 1) // 2 + feature
        assert cantor_unpair(paired) == (layer, feature)


def test_feature_joins_safetensors_to_parquet(tmp_path):
    (tmp_path / "stats").mkdir()
    (tmp_path / "inputs").mkdir()
    save_file(
        {
            "feature_ids": torch.tensor([3], dtype=torch.int32),
            "slot_counts": torch.tensor([1], dtype=torch.int32),
            "top_values": torch.tensor([2.5], dtype=torch.bfloat16),
            "input_ids": torch.tensor([0], dtype=torch.int32),
            "token_positions": torch.tensor([1], dtype=torch.int32),
            "firing_counts": torch.tensor([4], dtype=torch.int64),
        },
        tmp_path / "stats/layer_02.safetensors",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "input_id": 0,
                    "example_id": "example-0",
                    "text": "formatted prompt",
                    "image_references": ["hf-dataset://image"],
                    "metadata_json": '{"source":"test"}',
                    "token_ids": [10, 11, 12],
                    "token_mask": [True, True, True],
                    "token_type_ids": [0, 1, 0],
                }
            ]
        ),
        tmp_path / "inputs/part-00000.parquet",
    )
    manifest = {
        "schema": "circuit-tracer-activation-stats-v1",
        "model_name": "unused",
        "processor_tokens": 8,
        "transcoder": {"fingerprint": "run"},
        "layers": [
            {
                "layer": 2,
                "filename": "stats/layer_02.safetensors",
            }
        ],
        "inputs": [
            {
                "filename": "inputs/part-00000.parquet",
                "start": 0,
                "end": 1,
            }
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    store = ActivationStatsStore("test/repo")
    store._download = lambda filename: tmp_path / filename
    store._tokenizer = _Tokenizer()
    paired = (2 + 3) * (2 + 3 + 1) // 2 + 3
    feature = store.feature(paired)

    assert feature["layer"] == 2
    assert feature["index"] == 3
    assert feature["firing_count"] == 4
    assert feature["activation_frequency"] == 0.5
    example = feature["examples_quantiles"][0]["examples"][0]
    assert example["tokens"] == ["<10>", "<11>", "<12>"]
    assert example["tokens_acts_list"] == [0.0, 2.5, 0.0]
    assert example["image_references"] == ["hf-dataset://image"]
