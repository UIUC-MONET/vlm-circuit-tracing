# Attribution Graph Frontend Assets

This directory contains the browser frontend assets used by `circuit-tracer start-server`.

The frontend is a bundled snapshot of the attribution graph UI associated with:

- [On the Biology of a Large Language Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)
- [Circuit Tracing: Revealing Computational Graphs in Language Models](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)

For this VLM package, do not run these assets directly with a separate JavaScript dev server. Use the Python server instead:

```bash
circuit-tracer start-server --graph_file_dir ./your_graph --port 8041
```

The server provides graph JSON files, local annotation persistence, and image-token attention maps from the selected graph directory.
