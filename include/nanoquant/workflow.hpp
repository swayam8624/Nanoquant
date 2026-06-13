#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace nanoquant {

struct WorkflowOptions {
    std::string model_id;
    std::filesystem::path output_dir = "artifacts/nanoquant-model";
    std::string ollama_name;
    std::string base_ollama_name;
    std::string reference_ollama_name;
    std::string quantization = "Q4_K_M";
    std::string prompt = "Explain what model quantization is in two sentences.";
    std::filesystem::path llama_cpp_dir;
    bool execute = false;
    bool ollama_push = false;
    bool prefer_existing_gguf = true;
};

struct CommandStep {
    std::string title;
    std::string command;
    bool required = true;
};

struct WorkflowPlan {
    std::vector<CommandStep> steps;
    std::filesystem::path model_dir;
    std::filesystem::path fp16_gguf;
    std::filesystem::path quantized_gguf;
    std::filesystem::path modelfile;
    std::filesystem::path reference_modelfile;
    std::filesystem::path report_file;
};

struct CommandResult {
    int exit_code = -1;
    std::string output;
};

struct ComparisonReport {
    std::string base_output;
    std::string compressed_output;
    double lexical_overlap = 0.0;
    double length_ratio = 0.0;
    bool likely_degraded = false;
};

WorkflowPlan build_workflow_plan(const WorkflowOptions& options);
int run_workflow(const WorkflowOptions& options);
void print_workflow_plan(const WorkflowPlan& plan);

CommandResult run_command_capture(const std::string& command);
bool command_exists(const std::string& command);

ComparisonReport compare_outputs(const std::string& base_output, const std::string& compressed_output);
std::string build_tuning_recommendation(const WorkflowOptions& options, const ComparisonReport& report);

}  // namespace nanoquant
