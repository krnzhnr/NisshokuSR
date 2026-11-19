#pragma once
// Fused pre/post kernel. Один запуск на кадр, --use_fast_math.
// pre:  NV12 (device) -> RGB planar FP16 NCHW 0..1
// post: RGB planar FP16 -> NV12 (8-bit) или P010 (10-bit, MSB)
#include <cuda_fp16.h>
#include <cstddef>

void launch_nv12_to_rgb_planar_fp16(const unsigned char* nv12, int pitch,
                                    __half* out_rgb, int w, int h,
                                    void* stream);
void launch_p016_to_rgb_planar_fp16(const unsigned char* p016, int pitch,
                                    __half* out_rgb, int w, int h,
                                    void* stream);
void launch_rgb_planar_fp16_to_nv12(const __half* in_rgb,
                                    unsigned char* nv12, int pitch,
                                    int w, int h, void* stream);
void launch_rgb_planar_fp16_to_p010(const __half* in_rgb,
                                    unsigned char* p010, int pitch,
                                    int w, int h, void* stream);
void launch_fill_p010(unsigned char* p010, size_t bytes,
                      unsigned short value, void* stream);
