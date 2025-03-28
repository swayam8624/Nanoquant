from optimum.onnxruntime import ORTQuantizer, QuantizationConfig

model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
quantizer = ORTQuantizer.from_pretrained(model_id)
# Define quantization configuration
q_config = QuantizationConfig(approach="static", per_channel=True)
# Calibrate and quantize the model
quantized_model_path = quantizer.quantize(
    save_dir="./models/quantized_model",
    quantization_config=q_config,
    calibration_data=your_calibration_data  # Should be an iterable of input dicts
)
print("Quantized model saved at:", quantized_model_path)
