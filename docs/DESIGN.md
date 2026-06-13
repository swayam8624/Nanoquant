# NanoQuant Design Notes

## Goal

NanoQuant should be a public, cloneable C++ project that demonstrates serious systems work around model compression. The goal is not to ship a web product. The goal is to make the compression path inspectable, reproducible, and useful to open-source contributors.

## Current Scope

The current public version implements byte-level compression primitives and lightweight model-artifact inspection:

- `OneBitTensor`: row-wise positive/negative centroids plus packed sign bits.
- `Int4Tensor`: symmetric grouped int4 quantization with packed nibbles.
- `StructuredSparsityReport`: deterministic 2:4 sparsity analysis.
- `BinaryTensor`: a simple file-backed fp32 matrix format with mmap loading.
- `GgufInfo`: metadata and tensor-directory inspection for GGUF files without loading weights.
- `BackendInfo`: CPU fallback operations and optional Metal backend discovery/kernel execution.
- CLI demos that report size reduction and reconstruction error.

The implementation is intentionally dependency-free for the core build. A reviewer can build it with CMake and a C++20 compiler without installing PyTorch, CUDA, npm, Docker, databases, or cloud SDKs. On Apple platforms, Metal support is optional and compiled only when available.

## Non-Goals

- No payment system.
- No login or credit accounting.
- No hosted dashboard.
- No checked-in model weights.
- No CUDA-only path.
- No claims that sparse storage implies sparse runtime speedup without backend evidence.

## Apple Silicon Direction

The main target machine has 32 GB unified memory. That changes the design:

- Use CPU as the correctness baseline.
- Keep memory movement explicit; unified memory helps, but it does not remove bandwidth pressure.
- Keep Metal optional. The repo now dispatches Metal kernels for int4 dequantization and dense matvec while preserving CPU tests as the correctness baseline.
- Prefer streaming and mmap-friendly formats over loading every artifact into temporary vectors.
- Track KV-cache memory separately from compressed weight storage.

## Backend Strategy

The core should stay backend-neutral:

```text
packed weights -> backend adapter -> dequantize/matvec kernels
```

Adapters:

- Portable CPU: correctness, tests, and fallback.
- Metal: optional Apple GPU kernels for packed dequantization and matrix-vector work.
- Accelerate: future dense CPU math where useful.
- CUDA: optional external adapter only; never required for the public demo.

## Implemented Roadmap

1. `tensor-save`, `tensor-inspect`, and `tensor-demo` prove file-backed matrix loading.
2. `gguf-inspect` reads GGUF metadata and tensor descriptors without a runtime.
3. `int4-save`, `int4-inspect`, and `int4-demo` prove serialized packed-int4 tensors.
4. `benchmark` measures CPU and Metal dequantization plus matvec paths.
5. `evaluate-prompts` compares reference and compressed Ollama models over a prompt set.
6. `prove-small-model` presets the Hugging Face to GGUF to Ollama workflow for `HuggingFaceTB/SmolLM2-135M-Instruct`.
7. `include/nanoquant/c_api.h` is the ABI boundary; Python uses `bindings/python/nanoquant_ctypes.py` over that ABI.

## Next Implementation Steps

1. Add mmap support for serialized int4 tensors.
2. Reuse Metal buffers across benchmark iterations to isolate kernel time from allocation time.
3. Add benchmarks that separate compression time from dequantization and matvec time.
4. Add semantic or judge-based comparison reports next to lexical overlap.
5. Add pyproject packaging only after the C ABI has more real use.
