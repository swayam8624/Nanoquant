# NanoQuant Design Notes

## Goal

NanoQuant should be a public, cloneable C++ project that demonstrates serious systems work around model compression. The goal is not to ship a web product. The goal is to make the compression path inspectable, reproducible, and useful to open-source contributors.

## Current Scope

The first public version implements byte-level compression primitives:

- `OneBitTensor`: row-wise positive/negative centroids plus packed sign bits.
- `Int4Tensor`: symmetric grouped int4 quantization with packed nibbles.
- `StructuredSparsityReport`: deterministic 2:4 sparsity analysis.
- CLI demos that report size reduction and reconstruction error.

The implementation is intentionally dependency-free. A reviewer can build it with CMake and a C++20 compiler without installing PyTorch, CUDA, npm, Docker, databases, or cloud SDKs.

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
- Add Metal as an optional backend for dequantization and matvec once the packed formats stabilize.
- Prefer streaming and mmap-friendly formats over loading every artifact into temporary vectors.
- Track KV-cache memory separately from compressed weight storage.

## Backend Strategy

The core should stay backend-neutral:

```text
packed weights -> backend adapter -> dequantize/matvec kernels
```

Potential adapters:

- Portable CPU: correctness, tests, and fallback.
- Accelerate: dense CPU math where useful.
- Metal: Apple GPU kernels for packed dequantization and matrix-vector work.
- CUDA: optional external adapter only; never required for the public demo.

## Next Implementation Steps

1. Add a small binary matrix format for demo inputs and outputs.
2. Add per-channel int4 and one-bit metadata serialization.
3. Add benchmarks that separate compression time, dequantization time, and matvec time.
4. Add a small open-model conversion experiment with documented limits.
5. Add Metal kernels once the CPU output is covered by tests.
