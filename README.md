# NanoQuant

NanoQuant is a C++ model-weight compression playground focused on local inference, edge devices, and Apple Silicon class machines with unified memory. The project is intentionally small and public: no payments, no SaaS dashboard, no private model dumps, and no CUDA-only assumptions.

The current build demonstrates the core primitives that matter for a real compression stack:

- grouped symmetric int4 quantization
- row-wise one-bit centroid quantization
- bit/nibble packing
- 2:4 structured sparsity analysis
- deterministic demo tensors for reproducible output

This is not claiming to be a finished LLM runtime yet. It is a clean C++ foundation that can grow toward GGUF/llama.cpp-style integration, Metal kernels, and real checkpoint conversion.

## Quick Demo

```bash
git clone https://github.com/swayam8624/Nanoquant.git
cd Nanoquant
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/nanoquant demo --rows 1024 --cols 1024
```

Run tests:

```bash
ctest --test-dir build --output-on-failure
```

Inspect memory estimates:

```bash
./build/nanoquant inspect --rows 4096 --cols 4096
```

List public compression modes:

```bash
./build/nanoquant levels
```

Plan a Hugging Face to Ollama compression run:

```bash
./build/nanoquant hf-pipeline \
  --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --ollama-name tinyllama-nq \
  --reference-ollama-name tinyllama-reference \
  --prompt "Explain quantization in one sentence."
```

The command above is a dry run. Add `--execute` only after the plan points at working external tools:

- `huggingface-cli` or `git`
- `python3`
- a local `llama.cpp` checkout with `convert_hf_to_gguf.py`
- `llama-quantize`
- `ollama`

Push to an Ollama registry namespace when the model name and account are configured:

```bash
./build/nanoquant hf-pipeline \
  --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --ollama-name yourname/tinyllama-nq \
  --reference-ollama-name tinyllama-reference \
  --ollama-push \
  --execute
```

## Example Output

The exact timing varies by machine, but a demo prints:

```text
NanoQuant demo tensor: 1024x1024 fp32 weights (4194304 bytes)

codec: onebit-per-row
  original_bytes:   4194304
  compressed_bytes: 139264
  ratio:            30.12x
  size_reduction:   96.68%

codec: int4-symmetric
  original_bytes:   4194304
  compressed_bytes: 655360
  ratio:            6.40x
  size_reduction:   84.38%
```

## Why C++

Python is still useful for research scripts and bindings, but the compression core should be close to the memory layout:

- packed bits and nibbles are easier to audit in C++
- tensor storage can be made predictable
- Metal, Accelerate, and mmap/GGUF integration fit naturally
- the demo avoids installing PyTorch just to prove byte-level compression works

## Apple Silicon Notes

The target development machine has 32 GB unified memory and a Metal-capable GPU. NanoQuant therefore avoids CUDA barriers in the public core:

- CPU code is the portable baseline.
- Metal should be added as an optional backend, not a required dependency.
- 2:4 sparsity is reported as a storage/structure property unless a backend actually accelerates it.
- Runtime memory must include weights, temporary buffers, tokenizer/model metadata, and KV cache; a 32 GB machine cannot be treated like a 32 GB model-only budget.

See [docs/DESIGN.md](docs/DESIGN.md) for the implementation roadmap.

## Project Layout

```text
include/nanoquant/   Public C++ headers
src/                 Compression implementation
apps/                CLI entry point
tests/               Dependency-free test executable
docs/                Design notes and roadmap
```

See [docs/HF_OLLAMA_WORKFLOW.md](docs/HF_OLLAMA_WORKFLOW.md) for the end-to-end model workflow.

## Roadmap

- Add file-backed tensor loading for simple binary matrices.
- Add GGUF metadata inspection without depending on a full runtime.
- Add optional Metal kernels for dequantization and matvec.
- Add a converter path that can prove compression on a small open model.
- Add Python bindings only after the C++ ABI is stable enough to be worth binding.

## License

MIT
