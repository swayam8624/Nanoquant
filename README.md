---
license: mit
datasets:
  - mikasenghaas/wikitext-2
language:
  - en
metrics:
  - bleu
  - rouge
  - perplexity
  - accuracy
base_model:
  - openai-community/gpt2
tags:
  - Quantized
  - Pruned
  - Small
  - Nano
  - SBC
pipeline_tag: text-generation
---

# Model Card: Pruned & Quantized GPT-2 Fine-Tuned on WikiText-2

## Model Summary

This model is a pruned and quantized version of the GPT-2 architecture, fine-tuned on the WikiText-2 dataset. The pruning and quantization techniques reduce the model's size and computational requirements, making it suitable for deployment in resource-constrained environments, such as edge devices or applications with limited computational power.

## Model Details

### Developed by

- **Developer:** [SynSci]
- **Contact:** [swayam.singal@gmail.com]

### Model Description

- **Architecture:** GPT-2 (Generative Pre-trained Transformer 2)
- **Model Type:** Transformer-based language model
- **Base Model:** [openai-community/gpt2](https://huggingface.co/openai-community/gpt2)
- **Language:** English
- **License:** MIT
- **Fine-tuned on:** [mikasenghaas/wikitext-2](https://huggingface.co/datasets/mikasenghaas/wikitext-2)
- **Modifications:**
  - **Pruning:** Redundant weights removed to decrease model size and inference time.
  - **Quantization:** Weights quantized to 8-bit integers to reduce memory footprint and improve efficiency.

### Direct Use

- Text generation
- Language modeling
- Autocomplete suggestions
- Educational purposes in NLP and model optimization techniques

### Downstream Use

- Integration into applications requiring efficient language models
- Deployment on devices with limited computational resources

### Out-of-Scope Use

- Generation of misleading or harmful content
- Applications requiring understanding of languages other than English
- Tasks demanding high-precision language understanding beyond the model's capabilities

## Bias, Risks, and Limitations

### Biases

The model inherits biases present in the GPT-2 architecture and the WikiText-2 dataset, which consists of Wikipedia articles. These biases may include underrepresentation of certain topics or perspectives.

### Risks

- Potential generation of biased or inappropriate content
- Misinterpretation of generated text as factual information

### Limitations

- Reduced performance compared to the full-sized GPT-2 model due to pruning and quantization
- Limited to English language understanding and generation
- Not suitable for tasks requiring real-time processing of large-scale data

### Recommendations

Users should:

- Implement content filtering mechanisms to prevent the generation of inappropriate content.
- Avoid using the model for critical applications without thorough evaluation.
- Be aware of the model's limitations in understanding nuanced language and context.

## How to Get Started with the Model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("swayamsingal/NanoQuant")
model = AutoModelForCausalLM.from_pretrained("swayamsingal/NanoQuant")
input_text = "Once upon a time"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Training Details

### Training Data

- **Dataset:** [mikasenghaas/wikitext-2](https://huggingface.co/datasets/mikasenghaas/wikitext-2)
- **Description:** A collection of over 100 million tokens extracted from verified Good and Featured articles on Wikipedia. The dataset is available under the Creative Commons Attribution-ShareAlike License.

### Training Procedure

- **Preprocessing:** Standard tokenization and formatting compatible with GPT-2 requirements.
- **Training Regime:** Fine-tuning performed using mixed-precision training to balance performance and resource utilization.
- **Pruning:** Applied magnitude-based pruning to remove weights below a certain threshold.
- **Quantization:** Post-training dynamic quantization to 8-bit integers for weights.

### Hyperparameters

- **Learning Rate:** 5e-5
- **Batch Size:** 32
- **Epochs:** 3
- **Optimizer:** AdamW
- **Weight Decay:** 0.01

### Speeds, Sizes, Times

- **Original Model Size:** ~500 MB
- **Pruned & Quantized Model Size:** ~6 MB
- **Training Time:** Approximately 2 hours on a single MPS chip

## Evaluation

### Testing Data

- **Dataset:** [mikasenghaas/wikitext-2](https://huggingface.co/datasets/mikasenghaas/wikitext-2)
- **Split:** Validation set used for evaluation

### Metrics

- **Perplexity:** 155.43
- **BLEU Score:** 0.0498
- **ROUGE-1 Score:** 0.1836
- **Accuracy:** 93.2%

### Results Summary

The pruned and quantized model achieves competitive performance on the WikiText-2 validation set, with a significant reduction in model size and inference time compared to the original GPT-2 model.

## Model Examination

While specific interpretability analyses were not conducted, the model's architecture remains consistent with GPT-2, and standard transformer interpretability techniques can be applied.

## Environmental Impact

- **Hardware Type:** Macbook MPS [🙂‍↕️can't afford a good cuda gpu]
- **Training Duration:** 2 hours
- **Energy Consumption:** Approximately 0.5 kWh
- **Carbon Emitted:** Estimated 0.2 kg CO₂

## Technical Specifications

### Model Architecture and Objective

- **Architecture:** Transformer decoder with 12 layers, 12 attention heads, and a hidden size of 768.
- **Objective:** Causal language modeling (predicting the next token in a sequence).

### Compute Infrastructure

- **Hardware:** Single NVIDIA V100 GPU
- **Software:** PyTorch, Transformers library by Hugging Face

## Citation

If you use this model, please cite:

```bibtex
@misc{your2025prunedgpt2,
  title={Pruned and Quantized GPT-2 Fine-Tuned on WikiText-2},
  author={swayamsingal},
  year={2025},
  howpublished={\url{https://huggingface.co/your-username/pruned-quantized-gpt2-wikitext2}},
}
```

## Glossary

- **Pruning:** The process of removing weights from a neural network to reduce its size and computational requirements.
- **Quantization:** The process of reducing the precision of the weights in a neural network, typically to 8-bit integers, to decrease model size and increase inference speed.
