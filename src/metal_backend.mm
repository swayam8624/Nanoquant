#include "nanoquant/backend.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

namespace nanoquant {

BackendInfo metal_backend_info() {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
        return {"metal", false, true};
    }
    NSString* name = [device name];
    return {std::string("metal: ") + [name UTF8String], true, true};
}

std::string metal_kernel_source() {
    return R"msl(
#include <metal_stdlib>
using namespace metal;

kernel void dequant_int4_to_f32(device const uchar* packed_values [[buffer(0)]],
                                device const float* scales [[buffer(1)]],
                                device float* output [[buffer(2)]],
                                constant uint& group_size [[buffer(3)]],
                                constant uint& value_count [[buffer(4)]],
                                uint gid [[thread_position_in_grid]]) {
    if (gid >= value_count) {
        return;
    }
    const uchar byte_value = packed_values[gid >> 1];
    const uchar nibble = (gid & 1u) == 0u ? (byte_value & 0x0fu) : (byte_value >> 4);
    const int signed_value = int(nibble) - 8;
    output[gid] = float(signed_value) * scales[gid / group_size];
}

kernel void matvec_f32(device const float* matrix [[buffer(0)]],
                       device const float* vector [[buffer(1)]],
                       device float* output [[buffer(2)]],
                       constant uint& cols [[buffer(3)]],
                       uint row [[thread_position_in_grid]]) {
    float sum = 0.0f;
    const uint base = row * cols;
    for (uint col = 0; col < cols; ++col) {
        sum += matrix[base + col] * vector[col];
    }
    output[row] = sum;
}
)msl";
}

}  // namespace nanoquant
