# Contributing to circuit-tracer

Thank you for your interest in contributing. This repository is under active development, and we maintain it on a best-effort basis.

## Scope

The current public package supports VLM circuit tracing with released Gemma3-4B-IT transcoders. Contributions are most useful when they improve:

- VLM attribution graph computation,
- graph export and local visualization,
- image-token attention maps,
- packaging, documentation, tests, and reliability.

Training per-layer transcoders is not included in this release yet.

## Development Install

Install the bundled TransformerLens fork first, then install the VLM circuit tracer with dev dependencies:

```bash
pip install -e ../third_party/TransformerLens
pip install -e ".[dev]"
```

If you are installing from the repository root instead:

```bash
pip install -e third_party/TransformerLens
pip install -e "circuit_tracer_vlm[dev]"
```

The bundled TransformerLens fork is required because VLM tracing uses `HookedVLTransformer`.

## Testing

Run the test suite before submitting a pull request:

```bash
pytest
```

Some tests or workflows may require model/transcoder downloads and GPU memory. For changes that touch attribution, graph export, or attention maps, please note what hardware and model weights you used for validation.

## Linting And Type Checking

Use the project tools before submitting:

```bash
ruff check
ruff format
pyright
```

## Pull Requests

When opening a pull request:

- describe the VLM workflow affected by the change,
- include reproduction or validation steps,
- mention any model weights, images, prompts, or GPU assumptions,
- avoid adding cluster-specific paths, credentials, or private service dependencies.

## API Stability

This library is under active development. Breaking changes are possible as the VLM tracing pipeline becomes more complete.
