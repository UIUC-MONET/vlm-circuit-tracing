# Circuit Tracing in Vision-Language Models

[![arXiv](https://img.shields.io/badge/arXiv-2602.20330-b31b1b.svg)](https://arxiv.org/abs/2602.20330)

Official repository for "Circuit Tracing in Vision-Language Models: Understanding the Internal Mechanisms of Multimodal Thinking", accepted by CVPR 2026 Findings.

We are in the process of uploading implementations and results. All code and models will be available before CVPR 2026.

Todo:
- ~~Scripts for computing attribution graphs~~
- Scripts for training per-layer transcoders
- ~~Transcoder weights for Gemma3-4B-IT~~
- ~~Scripts for visualizing circuits~~
- ~~Scripts for computing attention graphs~~

## Current Scope

This release supports VLM circuit tracing for Gemma3-4B-IT with released transcoders. It can:

- compute attribution graphs from an image and prompt,
- convert attribution graphs into frontend JSON,
- serve and annotate circuits in a local browser UI,
- generate image-token attention maps for the circuit viewer.

Training per-layer transcoders is not included in this release yet.

## Installation

We recommend Python 3.10+ and a recent CUDA-enabled PyTorch install. Our experiments used Torch 2.7.1.

Install the bundled TransformerLens fork first, then the VLM circuit tracer:

```bash
git clone <repo-url>
cd vlm-circuit-tracing
pip install -e third_party/TransformerLens
pip install -e circuit_tracer_vlm
```

The bundled TransformerLens fork is required because VLM tracing uses `HookedVLTransformer`, which is not available in upstream `transformer-lens`.

## Transcoder Weights

The official Gemma3-4B-IT transcoder weights are available at:

https://huggingface.co/tianhux2/gemma3-4b-it-plt

## Attribution Graphs

To compute an attribution graph and write graph files for the local viewer:

```bash
circuit-tracer attribute \
  --model google/gemma-3-4b-it \
  --prompt "<start_of_image> Your text prompt here" \
  --transcoder_set tianhux2/gemma3-4b-it-plt \
  --slug demo \
  --image your_image.png \
  --graph_file_dir ./your_graph \
  --batch_size 4 \
  --dtype bfloat16
```

This writes `./your_graph`, which contains the JSON files needed by the local circuit viewer. We ran this workflow on a single H100 GPU; reduce `--batch_size`, `--max_feature_nodes`, or use `--offload cpu` if memory is tight.

## Visualizing Circuits

Start the local viewer for an existing graph directory:

```bash
circuit-tracer start-server --graph_file_dir ./your_graph --port 8041
```

Then open `http://localhost:8041` in a browser. If the server is running on a remote machine, forward the port first.

You can also compute attribution and launch the viewer in one command:

```bash
circuit-tracer attribute \
  --model google/gemma-3-4b-it \
  --prompt "<start_of_image> Your text prompt here" \
  --transcoder_set tianhux2/gemma3-4b-it-plt \
  --slug demo \
  --image your_image.png \
  --graph_file_dir ./your_graph \
  --batch_size 4 \
  --dtype bfloat16 \
  --server
```

The open-source viewer stores annotations in the local graph JSON files only. It does not require or use a database.

## Attention Maps

The local graph frontend can display one image-token map for each visual token. To generate these maps for an existing graph directory:

```bash
circuit-tracer attention-maps \
  --image your_image.png \
  --graph_file_dir ./your_graph \
  --model google/gemma-3-4b-pt \
  --method rollout \
  --render overlay \
  --dtype bfloat16 \
  --overwrite
```

This writes numbered images to `./your_graph/attention_maps/`. The frontend serves them through `/emb_image/<index>` when inspecting image-token nodes. Use `--method similarity` to render raw hidden-state cosine-similarity maps instead of attention rollout.

## Issues

If you encounter issues, questions, or ambiguities regarding our paper, please contact us or create an issue.
