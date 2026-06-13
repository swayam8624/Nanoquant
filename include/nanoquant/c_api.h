#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct nq_tensor_info {
    uint32_t version;
    uint64_t rows;
    uint64_t cols;
    uint64_t data_offset;
    uint64_t data_bytes;
} nq_tensor_info;

const char* nq_version(void);
int nq_save_demo_tensor(const char* path, uint64_t rows, uint64_t cols, uint64_t seed);
int nq_inspect_tensor(const char* path, nq_tensor_info* out_info);
const char* nq_last_error(void);

#ifdef __cplusplus
}
#endif
