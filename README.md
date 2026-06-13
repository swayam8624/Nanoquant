# NanoQuant

NanoQuant is a C++ model-weight compression playground focused on local inference, edge devices, and Apple Silicon class machines with unified memory. The project is intentionally small and public: no payments, no SaaS dashboard, no private model dumps, and no CUDA-only assumptions.

The current build demonstrates the core primitives that matter for a real compression stack:

- grouped symmetric int4 quantization
- row-wise one-bit centroid quantization
- bit/nibble packing
- 2:4 structured sparsity analysis
- file-backed fp32 tensor loading with mmap
- GGUF metadata and tensor-directory inspection without a model runtime
- portable CPU matvec/dequantization fallback plus optional Metal kernel source/discovery
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

Create and inspect a file-backed tensor:

```bash
./build/nanoquant tensor-save --path artifacts/demo.tensor --rows 8 --cols 8 --seed 7
./build/nanoquant tensor-inspect --path artifacts/demo.tensor
./build/nanoquant tensor-demo --path artifacts/demo.tensor
```

Inspect a GGUF header without loading model weights:

```bash
./build/nanoquant gguf-inspect --path path/to/model.gguf --metadata-limit 20 --tensor-limit 20
```

Check whether the optional Metal backend is compiled and visible:

```bash
./build/nanoquant metal-info
```

Serialize an int4 tensor and benchmark CPU vs Metal execution:

```bash
./build/nanoquant int4-save --path artifacts/demo.int4 --rows 256 --cols 256 --group-size 32
./build/nanoquant int4-inspect --path artifacts/demo.int4
./build/nanoquant int4-demo --path artifacts/demo.int4
./build/nanoquant benchmark --rows 1024 --cols 1024 --iterations 20 --csv artifacts/bench.csv
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

Use the small-model proof preset for a lighter first demo:

```bash
./build/nanoquant prove-small-model --ollama-name smollm2-nq
```

That is also a dry run by default. Add `--execute` only when `llama.cpp`, `ollama`, and the Hugging Face download toolchain are installed.

Compare a reference and compressed Ollama model with a prompt set:

```bash
./build/nanoquant evaluate-prompts \
  --reference smollm2-reference \
  --compressed smollm2-nq \
  --prompt-file examples/prompts.txt \
  --output artifacts/prompt-eval.md
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

Python is exposed only through the stable C ABI in `include/nanoquant/c_api.h`. The small `bindings/python/nanoquant_ctypes.py` helper is intentionally thin and does not bind C++ internals.

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

- Done: file-backed tensor loading for simple binary matrices.
- Done: GGUF metadata inspection without depending on a full runtime.
- Done: optional Metal kernel execution for dequantization and matvec, with CPU fallback.
- Done: converter path that can prove compression on a small open model through `prove-small-model`.
- Done: stable C ABI first; Python access is a thin ctypes layer over that ABI.
- Done: prompt-set evaluator for comparing reference and compressed Ollama outputs.

Next useful work:

- Add packed int4 mmap loading, not just serialized readback.
- Add Metal buffer reuse for cleaner benchmark numbers on repeated runs.
- Add semantic/embedding-based evaluation next to lexical overlap.

## License

MIT
