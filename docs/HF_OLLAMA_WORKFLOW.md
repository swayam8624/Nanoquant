# Hugging Face to Ollama Workflow

NanoQuant now includes a C++ orchestration command for the practical model path:

```text
Hugging Face model -> GGUF -> quantized GGUF -> Ollama model -> comparison report
```

The command is intentionally a coordinator. It does not pretend that arbitrary Hugging Face checkpoint conversion is solved inside this repository. Conversion and GGUF quantization are delegated to the mature `llama.cpp` toolchain, while NanoQuant owns the workflow, reporting, comparison guardrail, and future compression kernels.

## Dry Run

```bash
./build/nanoquant hf-pipeline \
  --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --ollama-name tinyllama-nq \
  --reference-ollama-name tinyllama-reference \
  --prompt "Explain quantization in one sentence."
```

Dry run prints every command and writes no model weights. Use it before every real run.

## Small-Model Proof Preset

For a first public demo, use the preset command:

```bash
./build/nanoquant prove-small-model --ollama-name smollm2-nq
```

The preset chooses `HuggingFaceTB/SmolLM2-135M-Instruct`, `Q4_K_M`, a generated f16 reference model, and a local artifact directory under `artifacts/smollm2-proof`. It is still a dry run unless `--execute` is present.

## Execute

Set `LLAMA_CPP_DIR` or pass `--llama-cpp`:

```bash
export LLAMA_CPP_DIR="$HOME/src/llama.cpp"

./build/nanoquant hf-pipeline \
  --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --ollama-name tinyllama-nq \
  --reference-ollama-name tinyllama-reference \
  --quant Q4_K_M \
  --prompt "Explain quantization in one sentence." \
  --execute
```

Expected external steps:

1. Download the Hugging Face snapshot with `huggingface-cli` or `git`.
2. Convert the model to f16 GGUF using `llama.cpp/convert_hf_to_gguf.py`.
3. Quantize the GGUF with `llama-quantize`.
4. Create an Ollama model with a generated Modelfile.
5. Compare the compressed output against a reference model. Use `--reference-ollama-name` to create an f16 reference from the converted GGUF, or `--base-ollama-name` to compare against an existing Ollama model.
6. Write a Markdown report under `artifacts/.../reports/`.

## Push and Pull

Use an Ollama namespace model name if you intend to push:

```bash
./build/nanoquant hf-pipeline \
  --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --ollama-name yourname/tinyllama-nq \
  --ollama-push \
  --execute
```

Another machine can then run:

```bash
ollama pull yourname/tinyllama-nq
ollama run yourname/tinyllama-nq
```

## Memory Model

This workflow is suitable for a 32 GB unified-memory Apple Silicon machine if you choose the model size carefully.

Important constraints:

- Downloading from Hugging Face is file-based.
- GGUF conversion may still need enough RAM to inspect and rewrite tensors.
- Quantization is file-oriented but still uses working buffers.
- Ollama runtime memory is not just compressed weights; KV cache and context length matter.
- CUDA is not required. If `llama.cpp` has Metal enabled, Ollama/llama.cpp can use Apple GPU paths.

For public demos, start with small open models before trying 7B class models.

## Degradation and Fine-Tuning

When `--reference-ollama-name` or `--base-ollama-name` is provided, NanoQuant runs the same prompt against the reference and compressed Ollama models and computes a cheap lexical guardrail:

- low token overlap
- extremely short compressed output
- extremely long compressed output

If the guardrail flags likely degradation, the report includes a distillation recommendation:

1. Build a prompt set for the target use case.
2. Run the original model as a teacher and save `{prompt, teacher_answer}` pairs.
3. Train a LoRA adapter against those teacher answers.
4. Merge/export to GGUF.
5. Re-quantize and rerun comparison.

This is the right direction for “fine tune for needs with intelligence of the original” without pretending that one generic prompt can evaluate a compressed model.
