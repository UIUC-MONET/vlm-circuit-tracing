# circuit-tracer

`circuit-tracer` computes and visualizes attribution graphs for vision-language models using pre-trained per-layer transcoders. This package is the VLM adaptation of the attribution graph workflow introduced by Ameisen et al. (2025) and Lindsey et al. (2025).

The current public release targets Gemma3-4B-IT with the released Gemma3 transcoders. It includes:

- attribution graph computation for image plus text prompts,
- graph pruning and JSON export for the browser viewer,
- a local circuit visualization server,
- image-token attention map generation for the viewer,
- per-layer transcoder training and export.

## Installation

Install from the repository root:

```bash
pip install -e third_party/TransformerLens
pip install -e circuit_tracer_vlm
```

The first command installs the bundled TransformerLens fork as `transformer-lens==2.16.0+vlm.0`. This fork provides `HookedVLTransformer`, which the VLM circuit tracer needs and upstream TransformerLens does not provide.

Install training extras when training PLTs:

```bash
pip install -e "circuit_tracer_vlm[train]"
```

## Transcoder Weights

The released Gemma3-4B-IT transcoders are available at:

https://huggingface.co/tianhux2/gemma3-4b-it-plt

Use this repository as the `--transcoder_set` argument in the CLI.

Local transcoder directories are also supported when they contain `config.yaml`
and `layer_*.safetensors`.

## End-To-End Workflow

Compute an attribution graph, write graph JSON files, and launch the local viewer:

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

The command writes graph files into `./your_graph` and serves them locally. If you are running on a remote machine, forward the selected port before opening the browser.

## Attribution Only

Save the raw attribution graph without creating frontend files:

```bash
circuit-tracer attribute \
  --model google/gemma-3-4b-it \
  --prompt "<start_of_image> Your text prompt here" \
  --transcoder_set tianhux2/gemma3-4b-it-plt \
  --image your_image.png \
  --graph_output_path graph.pt \
  --batch_size 4 \
  --dtype bfloat16
```

## Create Viewer Files From A Graph

The `attribute` command creates viewer files directly when `--slug` and `--graph_file_dir` are supplied. Programmatic users can also call:

```python
from circuit_tracer.utils.create_graph_files import create_graph_files

create_graph_files(
    graph_or_path="graph.pt",
    slug="demo",
    output_path="./your_graph",
    scan="tianhux2/gemma3-4b-it-plt",
)
```

## Visualize Existing Circuits

Serve an existing graph directory:

```bash
circuit-tracer start-server --graph_file_dir ./your_graph --port 8041
```

Then open `http://localhost:8041`. The open-source server stores annotations in the local graph JSON files; it does not use a database.

Activation examples load by default from
`Jingcheng/gemma3-4b-it-plt-activations`. The server lazily joins each requested
Safetensors feature to its Parquet input records. Open the standalone browser at
`/feature-view.html?layer=<layer>&featureId=<feature>`, or inspect a feature node
inside a circuit. Override the source with
`start-server --activation_stats org/repo@revision`. Multimodal records contain
image references rather than image bytes.

## Attention Maps

Generate image-token attention maps for the local viewer:

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

This saves `0.jpg` through `255.jpg` in `./your_graph/attention_maps`, matching Gemma-style 16x16 pooled image tokens. Use `--method similarity` for raw hidden-state similarity maps, or `--indices 0,5,17` to save only selected token maps.

## Training PLTs

Train per-layer transcoders from a Hugging Face dataset or a local `datasets`
directory:

```bash
circuit-tracer train-plt /path/to/or/hf-dataset \
  --model google/gemma-3-4b-it \
  --batch_size 1 \
  --max_steps 10000 \
  --save_dir ./gemma3-plt
```

By default the trainer expects `image` and `text` columns and uses the package's
Gemma3 image prompt template. Use `--no_image` for text-only datasets, or set
`--image_column`, `--text_column`, and `--prompt_template` for custom schemas.
The output directory can be used as `--transcoder_set ./gemma3-plt`.
Use `--layers` only for quick debug runs; attribution requires a PLT for every
model layer.

## CLI Reference

Important `attribute` arguments:

- `--model`: VLM model name or path. For Gemma3-4B-IT, use `google/gemma-3-4b-it`.
- `--image`: Input image path.
- `--prompt`: Prompt text. Include `<start_of_image>` where the image tokens should appear.
- `--transcoder_set`: Hugging Face repo or local path containing transcoders.
- `--graph_file_dir`: Directory for frontend JSON files.
- `--slug`: Name for the graph in the frontend.
- `--graph_output_path`: Optional raw `.pt` graph output.
- `--dtype`: `float32`, `float16`, or `bfloat16`.
- `--batch_size`: Backward-pass batch size. Lower it if memory is tight.
- `--max_feature_nodes`: Maximum number of feature nodes retained before graph pruning.
- `--offload`: Optional memory offload mode, such as `cpu`.
- `--server`: Start the local viewer after graph creation.

Important visualization arguments:

- `circuit-tracer start-server --graph_file_dir ./your_graph --port 8041`
- `circuit-tracer attention-maps --image your_image.png --graph_file_dir ./your_graph`
- `circuit-tracer train-plt /path/to/or/hf-dataset --model google/gemma-3-4b-it`

## Graph Interaction

In the browser viewer:

- Click a node to inspect it.
- Ctrl-click or Command-click a node to pin or unpin it.
- Use the edit controls in the side panel to annotate nodes.
- Hold `G` and click nodes to group them into a supernode.

Image-token nodes show generated attention maps when `attention_maps/` is present in the graph directory.

## Reproducing the Mixed-Data Recipe

For the released run's three-stream data preparation, copy
configs/gemma3_4b_it_data.example.yaml and pass it with --dataset_config. The
trainer prepares each source independently and uses deterministic weighted
round-robin batch selection. The example reproduces the 2:2:1 schedule for
image-only, packed SmolLM text, and Cauldron image-conversation batches without
embedding private filesystem paths in the code.
