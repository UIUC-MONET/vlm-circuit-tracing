# Circuit Tracing in Vision–Language Models

[![arXiv](https://img.shields.io/badge/arXiv-2602.20330-b31b1b.svg)](https://arxiv.org/abs/2602.20330)

Official repository for "Circuit Tracing in Vision–Language Models: Understanding the Internal Mechanisms of Multimodal Thinking", accepted by CVPR 2026 Findings.

We are in the process of uploading our implementations and results. All code and models will be available before CVPR 2026.

Todo:
- ~~Scripts for computing attribution graphs~~
- Scripts for training per-layer transcoders
- ~~Transcoder weights for Gemma3-4B-IT~~
- Scripts for visualizing circuits
- ~~Scripts for computing attention graphs~~

---

### State of circuit tracing in VLMs

Currently, our code supports the entire circuit tracing workflow for VLMs. Our code and experiments were conducted using Gemma3-4B-IT, and we believe the current framework can be extended to all models in the Gemma3 family.

Our future release will support many advance features that will make circuit level analysis of VLMs easy. It will support other popular VLM models (e.g. Qwen, LLaVa), built using nnsight and dictionary_learning, as well as new methods and optimizations.

## Instructions

Currently, you may need to configure different environments for different stages of the pipeline. We thus combine instructions for installation and running of each module.

We recommend python 3.10+ and torch 2.7.1.


### Transcoder Weights

The official transcoder weights for Gemma3-4B-IT is here:
https://huggingface.co/tianhux2/gemma3-4b-it-plt

### Attribution Graph

```
cd circuit_tracer_vlm

pip install -e .

pip uninstall transformer-lens
```

The reason you need to remove transformer-lens is so the code can use the custom fork, in ~/third_party/TransformerLens/transformer_lens

To find the attribution graph of a prompt, use the following command.

```
circuit-tracer attribute \
  --prompt "<start_of_image> Your text prompt here" \
  --transcoder_set tianhux2/gemma3-4b-it-plt \
  --slug demo \
  --image your_image.png \
  --graph_file_dir ./your_graph \
  --batch_size 4 --dtype bfloat16
```

We ran this script with a single H100 GPU, you may need to adjust the code and your configs for memory reduction. It will create a folder, specified in graph_file_dir, that will contain all the information you need to find circuits on this graph.


### Attention Maps

The local graph frontend can display one image-token map for each visual token. To generate these maps for an existing graph directory, run:

```
circuit-tracer attention-maps \
  --image your_image.png \
  --graph_file_dir ./your_graph \
  --model google/gemma-3-4b-pt \
  --method rollout \
  --render overlay \
  --dtype bfloat16 \
  --overwrite
```

This writes numbered images to `./your_graph/attention_maps/`. The frontend serves them through the existing `/emb_image/<index>` route when inspecting image-token nodes. Use `--method similarity` to render raw hidden-state cosine-similarity maps instead of attention rollout.


### Issues

If you encounter issues, questions or ambiguities regarding our paper, please contact us or create an issue. Thanks.