#include "nanoquant/c_api.h"

#include "nanoquant/binary_tensor.hpp"
#include "nanoquant/tensor.hpp"

#include <exception>
#include <filesystem>
#include <string>

namespace {

thread_local std::string g_last_error;

int capture_error(const char* fallback) {
    try {
        throw;
    } catch (const std::exception& error) {
        g_last_error = error.what();
    } catch (...) {
        g_last_error = fallback;
    }
    return -1;
}

}  // namespace

const char* nq_version(void) {
    return "0.1.0";
}

int nq_save_demo_tensor(const char* path, uint64_t rows, uint64_t cols, uint64_t seed) {
    try {
        if (path == nullptr) {
            g_last_error = "path is null";
            return -1;
        }
        nanoquant::save_binary_tensor(std::filesystem::path(path),
                                      nanoquant::make_deterministic_weights(static_cast<std::size_t>(rows),
                                                                            static_cast<std::size_t>(cols), seed));
        g_last_error.clear();
        return 0;
    } catch (...) {
        return capture_error("failed to save demo tensor");
    }
}

int nq_inspect_tensor(const char* path, nq_tensor_info* out_info) {
    try {
        if (path == nullptr || out_info == nullptr) {
            g_last_error = "path or out_info is null";
            return -1;
        }
        const nanoquant::BinaryTensorInfo info = nanoquant::inspect_binary_tensor(std::filesystem::path(path));
        out_info->version = info.version;
        out_info->rows = info.rows;
        out_info->cols = info.cols;
        out_info->data_offset = info.data_offset;
        out_info->data_bytes = info.data_bytes;
        g_last_error.clear();
        return 0;
    } catch (...) {
        return capture_error("failed to inspect tensor");
    }
}

const char* nq_last_error(void) {
    return g_last_error.c_str();
}
