#include "nanoquant/backend.hpp"
#include "nanoquant/binary_tensor.hpp"
#include "nanoquant/gguf.hpp"
#include "nanoquant/quantization.hpp"
#include "nanoquant/tensor.hpp"
#include "nanoquant/workflow.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

struct DemoOptions {
    std::size_t rows = 512;
    std::size_t cols = 512;
    std::uint64_t seed = 42;
    std::size_t group_size = 32;
};

struct TensorFileOptions {
    std::filesystem::path path;
    std::size_t rows = 16;
    std::size_t cols = 16;
    std::uint64_t seed = 42;
    std::size_t group_size = 32;
};

struct GgufOptions {
    std::filesystem::path path;
    std::size_t preview_limit = 80;
    std::size_t metadata_limit = 24;
    std::size_t tensor_limit = 24;
};

std::string next_value(int& index, int argc, char** argv, std::string_view flag) {
    if (index + 1 >= argc) {
        throw std::invalid_argument(std::string(flag) + " requires a value");
    }
    ++index;
    return argv[index];
}

DemoOptions parse_demo_options(int argc, char** argv, int start_index) {
    DemoOptions options;
    for (int index = start_index; index < argc; ++index) {
        const std::string_view arg(argv[index]);
        if (arg == "--rows") {
            options.rows = std::stoull(next_value(index, argc, argv, arg));
        } else if (arg == "--cols") {
            options.cols = std::stoull(next_value(index, argc, argv, arg));
        } else if (arg == "--seed") {
            options.seed = std::stoull(next_value(index, argc, argv, arg));
        } else if (arg == "--group-size") {
            options.group_size = std::stoull(next_value(index, argc, argv, arg));
        } else {
            throw std::invalid_argument("unknown option: " + std::string(arg));
        }
    }
    if (options.rows == 0U || options.cols == 0U) {
        throw std::invalid_argument("rows and cols must be non-zero");
    }
    return options;
}

TensorFileOptions parse_tensor_file_options(int argc, char** argv, int start_index, bool require_shape) {
    TensorFileOptions options;
    for (int index = start_index; index < argc; ++index) {
        const std::string_view arg(argv[index]);
        if (arg == "--path") {
            options.path = next_value(index, argc, argv, arg);
        } else if (arg == "--rows") {
            options.rows = std::stoull(next_value(index, argc, argv, arg));
        } else if (arg == "--cols") {
            options.cols = std::stoull(next_value(index, argc, argv, arg));
        } else if (arg == "--seed") {
            options.seed = std::stoull(next_value(index, argc, argv, arg));
        } else if (arg == "--group-size") {
            options.group_size = std::stoull(next_value(index, argc, argv, arg));
        } else {
            throw std::invalid_argument("unknown tensor option: " + std::string(arg));
        }
    }
    if (options.path.empty()) {
        throw std::invalid_argument("--path is required");
    }
    if (require_shape && (options.rows == 0U || options.cols == 0U)) {
        throw std::invalid_argument("rows and cols must be non-zero");
    }
    return options;
}

GgufOptions parse_gguf_options(int argc, char** argv, int start_index) {
    GgufOptions options;
    for (int index = start_index; index < argc; ++index) {
        const std::string_view arg(argv[index]);
        if (arg == "--path") {
            options.path = next_value(index, argc, argv, arg);
        } else if (arg == "--preview-limit") {
            options.preview_limit = std::stoull(next_value(index, argc, argv, arg));
        } else if (arg == "--metadata-limit") {
            options.metadata_limit = std::stoull(next_value(index, argc, argv, arg));
        } else if (arg == "--tensor-limit") {
            options.tensor_limit = std::stoull(next_value(index, argc, argv, arg));
        } else {
            throw std::invalid_argument("unknown GGUF option: " + std::string(arg));
        }
    }
    if (options.path.empty()) {
        throw std::invalid_argument("--path is required");
    }
    return options;
}

void print_help() {
    std::cout
        << "NanoQuant C++ demo CLI\n\n"
        << "Commands:\n"
        << "  nanoquant demo [--rows N] [--cols N] [--seed N] [--group-size N]\n"
        << "  nanoquant levels\n"
        << "  nanoquant inspect [--rows N] [--cols N]\n"
        << "  nanoquant tensor-save --path PATH [--rows N] [--cols N] [--seed N]\n"
        << "  nanoquant tensor-inspect --path PATH\n"
        << "  nanoquant tensor-demo --path PATH [--group-size N]\n"
        << "  nanoquant gguf-inspect --path MODEL.gguf [--metadata-limit N] [--tensor-limit N]\n"
        << "  nanoquant metal-info\n"
        << "  nanoquant prove-small-model [hf-pipeline options]\n"
        << "  nanoquant hf-pipeline --model-id ID --ollama-name NAME [options]\n\n"
        << "HF pipeline options:\n"
        << "  --output-dir PATH          Artifact directory, default artifacts/nanoquant-model\n"
        << "  --quant QTYPE              llama.cpp quant type, default Q4_K_M\n"
        << "  --llama-cpp PATH           llama.cpp checkout, or set LLAMA_CPP_DIR\n"
        << "  --base-ollama-name NAME    Existing teacher/base model for output comparison\n"
        << "  --reference-ollama-name N  Create an f16 reference Ollama model from converted GGUF\n"
        << "  --prompt TEXT              Prompt used for smoke test and comparison\n"
        << "  --ollama-push              Run `ollama push NAME` after create\n"
        << "  --execute                  Run external tools; omitted means dry-run plan only\n\n"
        << "Examples:\n"
        << "  nanoquant demo --rows 1024 --cols 1024\n"
        << "  nanoquant inspect --rows 4096 --cols 4096\n"
        << "  nanoquant tensor-save --path artifacts/demo.tensor --rows 8 --cols 8\n"
        << "  nanoquant tensor-demo --path artifacts/demo.tensor\n"
        << "  nanoquant gguf-inspect --path artifacts/model.gguf\n"
        << "  nanoquant prove-small-model --ollama-name smollm2-nq\n"
        << "  nanoquant hf-pipeline --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 --ollama-name tinyllama-nq\n";
}

void print_report(const nanoquant::CodecReport& report) {
    const double saved = 100.0 * (1.0 - static_cast<double>(report.compressed_bytes) /
                                           static_cast<double>(report.original_bytes));
    std::cout << "codec: " << report.name << '\n'
              << "  original_bytes:   " << report.original_bytes << '\n'
              << "  compressed_bytes: " << report.compressed_bytes << '\n'
              << "  ratio:            " << std::fixed << std::setprecision(2) << report.compression_ratio << "x\n"
              << "  size_reduction:   " << std::fixed << std::setprecision(2) << saved << "%\n"
              << "  rmse:             " << std::fixed << std::setprecision(6) << report.error.rmse << '\n'
              << "  mean_abs_error:   " << std::fixed << std::setprecision(6) << report.error.mean_absolute_error << '\n'
              << "  max_abs_error:    " << std::fixed << std::setprecision(6) << report.error.max_absolute_error << '\n';
}

int run_demo(int argc, char** argv, int start_index) {
    const DemoOptions options = parse_demo_options(argc, argv, start_index);

    const auto begin = std::chrono::steady_clock::now();
    const nanoquant::Tensor weights = nanoquant::make_deterministic_weights(options.rows, options.cols, options.seed);
    const auto generated = std::chrono::steady_clock::now();
    const nanoquant::CodecReport onebit = nanoquant::report_onebit(weights);
    const nanoquant::CodecReport int4 = nanoquant::report_int4(weights, options.group_size);
    const nanoquant::StructuredSparsityReport sparsity = nanoquant::analyze_2_to_4_sparsity(weights);
    const auto end = std::chrono::steady_clock::now();

    const auto gen_ms = std::chrono::duration_cast<std::chrono::milliseconds>(generated - begin).count();
    const auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

    std::cout << "NanoQuant demo tensor: " << options.rows << "x" << options.cols
              << " fp32 weights (" << weights.bytes() << " bytes)\n"
              << "seed: " << options.seed << "\n\n";
    print_report(onebit);
    std::cout << '\n';
    print_report(int4);
    std::cout << "\n2:4 structured sparsity analysis:\n"
              << "  groups:        " << sparsity.groups << '\n'
              << "  kept_values:   " << sparsity.kept_values << '\n'
              << "  dropped_values: " << sparsity.dropped_values << '\n'
              << "  sparsity:      " << std::fixed << std::setprecision(2) << sparsity.sparsity * 100.0 << "%\n\n"
              << "timing:\n"
              << "  generate_ms:   " << gen_ms << '\n'
              << "  total_ms:      " << total_ms << '\n';

    return EXIT_SUCCESS;
}

int run_levels() {
    std::cout
        << "NanoQuant public compression levels\n"
        << "  int4-symmetric: production-friendly 4-bit grouped weight codec\n"
        << "  onebit-per-row: research codec using row-wise positive/negative centroids\n"
        << "  2:4-analysis:   structured sparsity analyzer; no CUDA dependency or speed claim\n\n"
        << "Roadmap\n"
        << "  Metal kernels should target MPS/Metal buffers on Apple unified memory.\n"
        << "  CUDA-specific sparse kernels are optional adapters, not a core requirement.\n";
    return EXIT_SUCCESS;
}

int run_inspect(int argc, char** argv, int start_index) {
    const DemoOptions options = parse_demo_options(argc, argv, start_index);
    const std::size_t fp32 = options.rows * options.cols * sizeof(float);
    const std::size_t onebit_estimate = ((options.rows * options.cols + 7U) / 8U) + options.rows * 2U * sizeof(float);
    const std::size_t int4_estimate = ((options.rows * options.cols + 1U) / 2U) +
                                      ((options.rows * options.cols + options.group_size - 1U) / options.group_size) *
                                          sizeof(float);

    constexpr double gib = 1024.0 * 1024.0 * 1024.0;
    std::cout << "shape: " << options.rows << "x" << options.cols << '\n'
              << "fp32_estimate:   " << fp32 << " bytes (" << std::fixed << std::setprecision(3) << fp32 / gib
              << " GiB)\n"
              << "onebit_estimate: " << onebit_estimate << " bytes (" << onebit_estimate / gib << " GiB)\n"
              << "int4_estimate:   " << int4_estimate << " bytes (" << int4_estimate / gib << " GiB)\n"
              << "note: estimates cover one dense weight matrix, not full model runtime/KV cache.\n";
    return EXIT_SUCCESS;
}

int run_tensor_save(int argc, char** argv, int start_index) {
    const TensorFileOptions options = parse_tensor_file_options(argc, argv, start_index, true);
    const nanoquant::Tensor tensor = nanoquant::make_deterministic_weights(options.rows, options.cols, options.seed);
    nanoquant::save_binary_tensor(options.path, tensor);
    std::cout << "wrote tensor: " << options.path << '\n'
              << "shape: " << tensor.rows() << "x" << tensor.cols() << '\n'
              << "data_bytes: " << tensor.bytes() << '\n';
    return EXIT_SUCCESS;
}

int run_tensor_inspect(int argc, char** argv, int start_index) {
    const TensorFileOptions options = parse_tensor_file_options(argc, argv, start_index, false);
    const nanoquant::BinaryTensorInfo info = nanoquant::inspect_binary_tensor(options.path);
    std::cout << "path: " << options.path << '\n'
              << "version: " << info.version << '\n'
              << "shape: " << info.rows << "x" << info.cols << '\n'
              << "data_offset: " << info.data_offset << '\n'
              << "data_bytes: " << info.data_bytes << '\n';
    return EXIT_SUCCESS;
}

int run_tensor_demo(int argc, char** argv, int start_index) {
    const TensorFileOptions options = parse_tensor_file_options(argc, argv, start_index, false);
    const nanoquant::MappedTensor mapped(options.path);
    const nanoquant::Tensor tensor = mapped.materialize();
    std::cout << "mapped tensor: " << options.path << '\n'
              << "shape: " << mapped.rows() << "x" << mapped.cols() << '\n'
              << "mapped_values: " << mapped.elements() << "\n\n";
    print_report(nanoquant::report_int4(tensor, options.group_size));
    return EXIT_SUCCESS;
}

int run_gguf_inspect(int argc, char** argv, int start_index) {
    const GgufOptions options = parse_gguf_options(argc, argv, start_index);
    const nanoquant::GgufInfo info = nanoquant::inspect_gguf(options.path, options.preview_limit);

    std::cout << "path: " << options.path << '\n'
              << "version: " << info.version << '\n'
              << "metadata_count: " << info.metadata_count << '\n'
              << "tensor_count: " << info.tensor_count << "\n\n";

    const std::size_t metadata_count = std::min(options.metadata_limit, info.metadata.size());
    std::cout << "metadata preview:\n";
    for (std::size_t index = 0; index < metadata_count; ++index) {
        const auto& entry = info.metadata[index];
        std::cout << "  " << entry.key << " (" << entry.type << "): " << entry.value_preview << '\n';
    }
    if (info.metadata.size() > metadata_count) {
        std::cout << "  ... " << (info.metadata.size() - metadata_count) << " more\n";
    }

    const std::size_t tensor_count = std::min(options.tensor_limit, info.tensors.size());
    std::cout << "\ntensor preview:\n";
    for (std::size_t index = 0; index < tensor_count; ++index) {
        const auto& tensor = info.tensors[index];
        std::cout << "  " << tensor.name << " [";
        for (std::size_t dim = 0; dim < tensor.dimensions.size(); ++dim) {
            if (dim > 0U) {
                std::cout << "x";
            }
            std::cout << tensor.dimensions[dim];
        }
        std::cout << "] " << nanoquant::gguf_type_name(tensor.type) << " offset=" << tensor.offset << '\n';
    }
    if (info.tensors.size() > tensor_count) {
        std::cout << "  ... " << (info.tensors.size() - tensor_count) << " more\n";
    }
    return EXIT_SUCCESS;
}

int run_metal_info() {
    const auto cpu = nanoquant::cpu_backend_info();
    const auto metal = nanoquant::metal_backend_info();
    std::cout << "cpu: available=" << (cpu.available ? "yes" : "no")
              << " accelerated=" << (cpu.accelerated ? "yes" : "no") << '\n'
              << "metal: available=" << (metal.available ? "yes" : "no")
              << " accelerated=" << (metal.accelerated ? "yes" : "no")
              << " name=\"" << metal.name << "\"\n";
    const std::string kernels = nanoquant::metal_kernel_source();
    if (!kernels.empty()) {
        std::cout << "metal kernels: dequant_int4_to_f32, matvec_f32\n";
    }
    return EXIT_SUCCESS;
}

int run_hf_pipeline(int argc, char** argv, int start_index) {
    nanoquant::WorkflowOptions options;
    for (int index = start_index; index < argc; ++index) {
        const std::string_view arg(argv[index]);
        if (arg == "--model-id") {
            options.model_id = next_value(index, argc, argv, arg);
        } else if (arg == "--output-dir") {
            options.output_dir = next_value(index, argc, argv, arg);
        } else if (arg == "--ollama-name") {
            options.ollama_name = next_value(index, argc, argv, arg);
        } else if (arg == "--base-ollama-name") {
            options.base_ollama_name = next_value(index, argc, argv, arg);
        } else if (arg == "--reference-ollama-name") {
            options.reference_ollama_name = next_value(index, argc, argv, arg);
        } else if (arg == "--quant") {
            options.quantization = next_value(index, argc, argv, arg);
        } else if (arg == "--prompt") {
            options.prompt = next_value(index, argc, argv, arg);
        } else if (arg == "--llama-cpp") {
            options.llama_cpp_dir = next_value(index, argc, argv, arg);
        } else if (arg == "--ollama-push") {
            options.ollama_push = true;
        } else if (arg == "--execute") {
            options.execute = true;
        } else {
            throw std::invalid_argument("unknown hf-pipeline option: " + std::string(arg));
        }
    }
    return nanoquant::run_workflow(options);
}

int run_prove_small_model(int argc, char** argv, int start_index) {
    nanoquant::WorkflowOptions options;
    options.model_id = "HuggingFaceTB/SmolLM2-135M-Instruct";
    options.ollama_name = "smollm2-nanoquant";
    options.reference_ollama_name = "smollm2-reference";
    options.output_dir = "artifacts/smollm2-proof";
    options.quantization = "Q4_K_M";
    options.prompt = "Explain model quantization in one concise sentence.";

    for (int index = start_index; index < argc; ++index) {
        const std::string_view arg(argv[index]);
        if (arg == "--model-id") {
            options.model_id = next_value(index, argc, argv, arg);
        } else if (arg == "--output-dir") {
            options.output_dir = next_value(index, argc, argv, arg);
        } else if (arg == "--ollama-name") {
            options.ollama_name = next_value(index, argc, argv, arg);
        } else if (arg == "--base-ollama-name") {
            options.base_ollama_name = next_value(index, argc, argv, arg);
            options.reference_ollama_name.clear();
        } else if (arg == "--reference-ollama-name") {
            options.reference_ollama_name = next_value(index, argc, argv, arg);
        } else if (arg == "--quant") {
            options.quantization = next_value(index, argc, argv, arg);
        } else if (arg == "--prompt") {
            options.prompt = next_value(index, argc, argv, arg);
        } else if (arg == "--llama-cpp") {
            options.llama_cpp_dir = next_value(index, argc, argv, arg);
        } else if (arg == "--ollama-push") {
            options.ollama_push = true;
        } else if (arg == "--execute") {
            options.execute = true;
        } else {
            throw std::invalid_argument("unknown prove-small-model option: " + std::string(arg));
        }
    }
    return nanoquant::run_workflow(options);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            print_help();
            return EXIT_SUCCESS;
        }

        const std::string_view command(argv[1]);
        if (command == "demo") {
            return run_demo(argc, argv, 2);
        }
        if (command == "levels") {
            return run_levels();
        }
        if (command == "inspect") {
            return run_inspect(argc, argv, 2);
        }
        if (command == "tensor-save") {
            return run_tensor_save(argc, argv, 2);
        }
        if (command == "tensor-inspect") {
            return run_tensor_inspect(argc, argv, 2);
        }
        if (command == "tensor-demo") {
            return run_tensor_demo(argc, argv, 2);
        }
        if (command == "gguf-inspect") {
            return run_gguf_inspect(argc, argv, 2);
        }
        if (command == "metal-info") {
            return run_metal_info();
        }
        if (command == "prove-small-model") {
            return run_prove_small_model(argc, argv, 2);
        }
        if (command == "hf-pipeline") {
            return run_hf_pipeline(argc, argv, 2);
        }
        if (command == "--help" || command == "-h" || command == "help") {
            print_help();
            return EXIT_SUCCESS;
        }

        std::cerr << "unknown command: " << command << "\n\n";
        print_help();
        return EXIT_FAILURE;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
