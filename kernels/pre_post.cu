// NV12/P016 <-> RGB planar FP16. BT.709 limited <-> RGB full.
// 10-bit: P010/P016, 10-bit значение в старших битах 16-битного слова.
#include "pre_post.h"
#include <cuda_runtime.h>

__device__ __forceinline__ float clamp01(float v) {
  return fminf(fmaxf(v, 0.0f), 1.0f);
}

__device__ __forceinline__ float clamp8(float v) {
  return fminf(fmaxf(v, 0.0f), 255.0f);
}

__device__ __forceinline__ unsigned char to_u8(float v) {
  return (unsigned char)__float2int_rn(clamp8(v));
}

__global__ void kNv12ToRgbPlanar(const unsigned char* y, const unsigned char* uv,
                                 int pitch, __half* r, __half* g, __half* b,
                                 int w, int h) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int yy = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || yy >= h) return;
  int yi = yy * pitch + x;
  int uvi = (yy / 2) * pitch + (x / 2) * 2;
  // Источник — BT.709 limited: Y 16..235, C 16..240. Модель ждет full RGB.
  float Y = (y[yi] - 16.0f) * (255.0f / 219.0f);
  float U = (uv[uvi] - 128.0f) * (255.0f / 224.0f);
  float V = (uv[uvi + 1] - 128.0f) * (255.0f / 224.0f);
  float R = Y + 1.5748f * V;
  float G = Y - 0.187324f * U - 0.468124f * V;
  float B = Y + 1.8556f * U;
  int i = yy * w + x;
  r[i] = __float2half(clamp01(R * (1.0f / 255.0f)));
  g[i] = __float2half(clamp01(G * (1.0f / 255.0f)));
  b[i] = __float2half(clamp01(B * (1.0f / 255.0f)));
}

// P016: 16-битные слова, 10-битные данные в старших битах (NVDEC/NVENC).
__global__ void kP016ToRgbPlanar(const unsigned short* y, const unsigned short* uv,
                                 int pitch, __half* r, __half* g, __half* b,
                                 int w, int h) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int yy = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || yy >= h) return;
  int stride = pitch / 2;
  // 10-bit limited: Y 64..940, C 64..960; извлекаем 10 бит из старших.
  float Y = ((float)(y[yy * stride + x] >> 6) - 64.0f) * (255.0f / 876.0f);
  float U = ((float)(uv[(yy / 2) * stride + (x / 2) * 2] >> 6) - 512.0f) *
            (255.0f / 896.0f);
  float V = ((float)(uv[(yy / 2) * stride + (x / 2) * 2 + 1] >> 6) - 512.0f) *
            (255.0f / 896.0f);
  float R = Y + 1.5748f * V;
  float G = Y - 0.187324f * U - 0.468124f * V;
  float B = Y + 1.8556f * U;
  int i = yy * w + x;
  r[i] = __float2half(clamp01(R * (1.0f / 255.0f)));
  g[i] = __float2half(clamp01(G * (1.0f / 255.0f)));
  b[i] = __float2half(clamp01(B * (1.0f / 255.0f)));
}

__global__ void kRgbPlanarToNv12(const __half* r, const __half* g, const __half* b,
                                 unsigned char* y, unsigned char* uv,
                                 int pitch, int w, int h) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int yy = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || yy >= h) return;
  int i = yy * w + x;
  float R = clamp8(__half2float(r[i]) * 255.0f);
  float G = clamp8(__half2float(g[i]) * 255.0f);
  float B = clamp8(__half2float(b[i]) * 255.0f);
  float Y = 0.2126f * R + 0.7152f * G + 0.0722f * B;
  y[yy * pitch + x] = to_u8(16.0f + Y * (219.0f / 255.0f));
  if ((x & 1) == 0 && (yy & 1) == 0) {
    // Полноценное усреднение 2x2, а не выбор левого верхнего пикселя.
    int x1 = min(x + 1, w - 1);
    int y1 = min(yy + 1, h - 1);
    int i10 = yy * w + x1, i01 = y1 * w + x, i11 = y1 * w + x1;
    float Ra = 0.25f * (R + clamp8(__half2float(r[i10]) * 255.0f) +
                        clamp8(__half2float(r[i01]) * 255.0f) +
                        clamp8(__half2float(r[i11]) * 255.0f));
    float Ga = 0.25f * (G + clamp8(__half2float(g[i10]) * 255.0f) +
                        clamp8(__half2float(g[i01]) * 255.0f) +
                        clamp8(__half2float(g[i11]) * 255.0f));
    float Ba = 0.25f * (B + clamp8(__half2float(b[i10]) * 255.0f) +
                        clamp8(__half2float(b[i01]) * 255.0f) +
                        clamp8(__half2float(b[i11]) * 255.0f));
    float Cb = -0.114572f * Ra - 0.385428f * Ga + 0.5f * Ba;
    float Cr = 0.5f * Ra - 0.454153f * Ga - 0.045847f * Ba;
    int uvi = (yy / 2) * pitch + x;
    uv[uvi] = to_u8(128.0f + Cb * (224.0f / 255.0f));
    uv[uvi + 1] = to_u8(128.0f + Cr * (224.0f / 255.0f));
  }
}

// P010: 16-битные слова, 10-битное значение в старших битах (NVENC MSB).
__device__ __forceinline__ unsigned short to_p010_10(float v10) {
  int v = __float2int_rn(fminf(fmaxf(v10, 0.0f), 1023.0f));
  return (unsigned short)(v << 6);
}

__global__ void kRgbPlanarToP010(const __half* r, const __half* g, const __half* b,
                                 unsigned short* y, unsigned short* uv,
                                 int pitch, int w, int h) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int yy = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || yy >= h) return;
  int i = yy * w + x;
  float R = clamp8(__half2float(r[i]) * 255.0f);
  float G = clamp8(__half2float(g[i]) * 255.0f);
  float B = clamp8(__half2float(b[i]) * 255.0f);
  float Y = 0.2126f * R + 0.7152f * G + 0.0722f * B;
  int ystride = pitch / 2; // 16-битных слов в строке
  // RGB full -> YUV BT.709 limited (10-bit): Y 64..940, C 64..960.
  y[yy * ystride + x] = to_p010_10(64.0f + Y * (876.0f / 255.0f));
  if ((x & 1) == 0 && (yy & 1) == 0) {
    int x1 = min(x + 1, w - 1);
    int y1 = min(yy + 1, h - 1);
    int i10 = yy * w + x1, i01 = y1 * w + x, i11 = y1 * w + x1;
    float Ra = 0.25f * (R + clamp8(__half2float(r[i10]) * 255.0f) +
                        clamp8(__half2float(r[i01]) * 255.0f) +
                        clamp8(__half2float(r[i11]) * 255.0f));
    float Ga = 0.25f * (G + clamp8(__half2float(g[i10]) * 255.0f) +
                        clamp8(__half2float(g[i01]) * 255.0f) +
                        clamp8(__half2float(g[i11]) * 255.0f));
    float Ba = 0.25f * (B + clamp8(__half2float(b[i10]) * 255.0f) +
                        clamp8(__half2float(b[i01]) * 255.0f) +
                        clamp8(__half2float(b[i11]) * 255.0f));
    float Cb = -0.114572f * Ra - 0.385428f * Ga + 0.5f * Ba;
    float Cr = 0.5f * Ra - 0.454153f * Ga - 0.045847f * Ba;
    int uvi = (yy / 2) * ystride + x;
    uv[uvi] = to_p010_10(512.0f + Cb * (896.0f / 255.0f));
    uv[uvi + 1] = to_p010_10(512.0f + Cr * (896.0f / 255.0f));
  }
}

__global__ void kFillP010(unsigned short* p, size_t n, unsigned short v) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = v;
}

void launch_nv12_to_rgb_planar_fp16(const unsigned char* nv12, int pitch,
                                    __half* out_rgb, int w, int h, void* stream) {
  dim3 blk(32, 8), grid((w + 31) / 32, (h + 7) / 8);
  int n = w * h;
  kNv12ToRgbPlanar<<<grid, blk, 0, (cudaStream_t)stream>>>(
      nv12, nv12 + (size_t)pitch * h, pitch,
      out_rgb, out_rgb + n, out_rgb + 2 * n, w, h);
}

void launch_p016_to_rgb_planar_fp16(const unsigned char* p016, int pitch,
                                    __half* out_rgb, int w, int h, void* stream) {
  dim3 blk(32, 8), grid((w + 31) / 32, (h + 7) / 8);
  int n = w * h;
  const unsigned short* y = (const unsigned short*)p016;
  const unsigned short* uv = (const unsigned short*)(p016 + (size_t)pitch * h);
  kP016ToRgbPlanar<<<grid, blk, 0, (cudaStream_t)stream>>>(
      y, uv, pitch, out_rgb, out_rgb + n, out_rgb + 2 * n, w, h);
}

void launch_rgb_planar_fp16_to_nv12(const __half* in_rgb, unsigned char* nv12,
                                    int pitch, int w, int h, void* stream) {
  dim3 blk(32, 8), grid((w + 31) / 32, (h + 7) / 8);
  int n = w * h;
  kRgbPlanarToNv12<<<grid, blk, 0, (cudaStream_t)stream>>>(
      in_rgb, in_rgb + n, in_rgb + 2 * n,
      nv12, nv12 + (size_t)pitch * h, pitch, w, h);
}

void launch_rgb_planar_fp16_to_p010(const __half* in_rgb, unsigned char* p010,
                                    int pitch, int w, int h, void* stream) {
  dim3 blk(32, 8), grid((w + 31) / 32, (h + 7) / 8);
  int n = w * h;
  kRgbPlanarToP010<<<grid, blk, 0, (cudaStream_t)stream>>>(
      in_rgb, in_rgb + n, in_rgb + 2 * n,
      (unsigned short*)p010, (unsigned short*)(p010 + (size_t)pitch * h), pitch, w, h);
}

void launch_fill_p010(unsigned char* p010, size_t bytes, unsigned short value, void* stream) {
  size_t n = bytes / 2;
  int threads = 256;
  int blocks = (int)((n + threads - 1) / threads);
  kFillP010<<<blocks, threads, 0, (cudaStream_t)stream>>>((unsigned short*)p010, n, value);
}
