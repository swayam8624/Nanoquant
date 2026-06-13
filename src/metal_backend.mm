#include "nanoquant/backend.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstring>
#include <stdexcept>

namespace nanoquant {
namespace {

id<MTLDevice> require_device() {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
        throw std::runtime_error("Metal device is not available");
    }
    return device;
}

id<MTLComputePipelineState> make_pipeline(id<MTLDevice> device, NSString* function_name) {
    NSError* error = nil;
    NSString* source = [NSString stringWithUTF8String:metal_kernel_source().c_str()];
    id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
    if (library == nil) {
        throw std::runtime_error(std::string("failed to compile Metal kernels: ") +
                                 (error == nil ? "unknown" : [[error localizedDescription] UTF8String]));
    }
    id<MTLFunction> function = [library newFunctionWithName:function_name];
    if (function == nil) {
        throw std::runtime_error("Metal function is missing");
    }
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    if (pipeline == nil) {
        throw std::runtime_error(std::string("failed to create Metal pipeline: ") +
                                 (error == nil ? "unknown" : [[error localizedDescription] UTF8String]));
    }
    return pipeline;
}

void wait_for_command(id<MTLCommandBuffer> command_buffer) {
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if ([command_buffer status] == MTLCommandBufferStatusError) {
        NSError* error = [command_buffer error];
        throw std::runtime_error(std::string("Metal command failed: ") +
                                 (error == nil ? "unknown" : [[error localizedDescription] UTF8String]));
    }
}

}  // namespace

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

Tensor dequantize_int4_metal(const Int4Tensor& input) {
    const std::size_t value_count = input.rows * input.cols;
    static id<MTLDevice> device = require_device();
    static id<MTLCommandQueue> queue = [device newCommandQueue];
    static id<MTLComputePipelineState> pipeline = make_pipeline(device, @"dequant_int4_to_f32");

    id<MTLBuffer> packed_buffer =
        [device newBufferWithBytes:input.packed_values.data()
                            length:input.packed_values.size()
                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> scale_buffer =
        [device newBufferWithBytes:input.scales.data()
                            length:input.scales.size() * sizeof(float)
                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> output_buffer =
        [device newBufferWithLength:value_count * sizeof(float) options:MTLResourceStorageModeShared];

    std::uint32_t group_size = static_cast<std::uint32_t>(input.group_size);
    std::uint32_t count = static_cast<std::uint32_t>(value_count);

    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:packed_buffer offset:0 atIndex:0];
    [encoder setBuffer:scale_buffer offset:0 atIndex:1];
    [encoder setBuffer:output_buffer offset:0 atIndex:2];
    [encoder setBytes:&group_size length:sizeof(group_size) atIndex:3];
    [encoder setBytes:&count length:sizeof(count) atIndex:4];
    const NSUInteger width = pipeline.threadExecutionWidth;
    MTLSize threads_per_group = MTLSizeMake(width, 1, 1);
    MTLSize threads = MTLSizeMake(value_count, 1, 1);
    [encoder dispatchThreads:threads threadsPerThreadgroup:threads_per_group];
    [encoder endEncoding];
    wait_for_command(command_buffer);

    std::vector<float> values(value_count);
    std::memcpy(values.data(), [output_buffer contents], value_count * sizeof(float));
    return Tensor(input.rows, input.cols, std::move(values));
}

std::vector<float> matvec_metal(const Tensor& matrix, std::span<const float> vector) {
    if (matrix.cols() != vector.size()) {
        throw std::invalid_argument("matvec dimension mismatch");
    }

    static id<MTLDevice> device = require_device();
    static id<MTLCommandQueue> queue = [device newCommandQueue];
    static id<MTLComputePipelineState> pipeline = make_pipeline(device, @"matvec_f32");

    id<MTLBuffer> matrix_buffer =
        [device newBufferWithBytes:matrix.values().data()
                            length:matrix.bytes()
                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> vector_buffer =
        [device newBufferWithBytes:vector.data()
                            length:vector.size() * sizeof(float)
                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> output_buffer =
        [device newBufferWithLength:matrix.rows() * sizeof(float) options:MTLResourceStorageModeShared];

    std::uint32_t cols = static_cast<std::uint32_t>(matrix.cols());
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:matrix_buffer offset:0 atIndex:0];
    [encoder setBuffer:vector_buffer offset:0 atIndex:1];
    [encoder setBuffer:output_buffer offset:0 atIndex:2];
    [encoder setBytes:&cols length:sizeof(cols) atIndex:3];
    const NSUInteger width = pipeline.threadExecutionWidth;
    MTLSize threads_per_group = MTLSizeMake(width, 1, 1);
    MTLSize threads = MTLSizeMake(matrix.rows(), 1, 1);
    [encoder dispatchThreads:threads threadsPerThreadgroup:threads_per_group];
    [encoder endEncoding];
    wait_for_command(command_buffer);

    std::vector<float> output(matrix.rows());
    std::memcpy(output.data(), [output_buffer contents], output.size() * sizeof(float));
    return output;
}

}  // namespace nanoquant
