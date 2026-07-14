from pathlib import Path

import pytest
import yaml

from circuit_tracer.training.data import (
    DatasetSourceConfig,
    WeightedBatchIterator,
    format_prompt,
    load_dataset_config,
    prepare_packed_text_dataset,
)


def test_weighted_batch_iterator_uses_exact_schedule_and_restarts_sources():
    iterator = WeightedBatchIterator(
        [[{"source": "image"}], [{"source": "text"}], [{"source": "cauldron"}]],
        [2, 2, 1],
    )

    observed = [next(iterator)["source"] for _ in range(10)]

    assert observed == [
        "image",
        "image",
        "text",
        "text",
        "cauldron",
        "image",
        "image",
        "text",
        "text",
        "cauldron",
    ]


def test_format_prompt_preserves_conversation_answer():
    prompt = format_prompt(
        {"user": "What is shown?", "assistant": "A red bird."},
        "user={user}; assistant={assistant}; text={text}",
    )

    assert prompt == "user=What is shown?; assistant=A red bird.; text=What is shown?"


def test_load_dataset_config(tmp_path: Path):
    config_path = tmp_path / "datasets.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "sources": [
                    {
                        "path": "images",
                        "weight": 2,
                        "text_column": None,
                        "image_column": "image",
                    },
                    {
                        "path": "text",
                        "weight": 1,
                        "image_column": None,
                        "packing": True,
                    },
                ]
            }
        )
    )

    sources = load_dataset_config(config_path)

    assert [source.path for source in sources] == ["images", "text"]
    assert [source.weight for source in sources] == [2, 1]
    assert sources[1].packing is True


def test_load_dataset_config_rejects_non_positive_weight(tmp_path: Path):
    config_path = tmp_path / "datasets.yaml"
    config_path.write_text(
        yaml.safe_dump({"sources": [{"path": "images", "weight": 0}]})
    )

    with pytest.raises(ValueError, match="weight must be positive"):
        load_dataset_config(config_path)


def test_prepare_packed_text_dataset_emits_only_full_blocks():
    datasets = pytest.importorskip("datasets")

    class Tokenizer:
        eos_token = "|"

        def __call__(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            return {"input_ids": list(range(len(text)))}

    dataset = datasets.Dataset.from_dict({"text": ["abc", "def"]})
    source = DatasetSourceConfig(
        path="text",
        text_column="text",
        image_column=None,
        packing=True,
        max_length=4,
        map_batch_size=2,
    )

    prepared = prepare_packed_text_dataset(dataset, Tokenizer(), source)

    assert len(prepared) == 2
    assert all(len(input_ids) == 4 for input_ids in prepared["input_ids"])
