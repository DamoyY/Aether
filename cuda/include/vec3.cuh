#ifndef VEC3_CUH
#define VEC3_CUH

#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ float3 operator*(const float3& a, const float3& b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__device__ __forceinline__ float3 operator*(const float3& a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ __forceinline__ float3 operator*(float s, const float3& a) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ __forceinline__ float3 operator/(const float3& a, float s) {
    float inv = 1.0f / s;
    return make_float3(a.x * inv, a.y * inv, a.z * inv);
}

__device__ __forceinline__ float3 operator/(const float3& a, const float3& b) {
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

__device__ __forceinline__ float3 operator-(const float3& a) {
    return make_float3(-a.x, -a.y, -a.z);
}

__device__ __forceinline__ float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ float3 cross(const float3& a, const float3& b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__device__ __forceinline__ float length(const float3& v) {
    return sqrtf(dot(v, v));
}

__device__ __forceinline__ float3 normalize(const float3& v) {
    float len = length(v);
    return len > 0.0f ? v / len : make_float3(0.0f, 0.0f, 0.0f);
}

__device__ __forceinline__ float3 reflect(const float3& v, const float3& n) {
    return v - 2.0f * dot(v, n) * n;
}

__device__ __forceinline__ float3 fminf3(const float3& a, const float3& b) {
    return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

__device__ __forceinline__ float3 fmaxf3(const float3& a, const float3& b) {
    return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

__device__ __forceinline__ float3 absf3(const float3& a) {
    return make_float3(fabsf(a.x), fabsf(a.y), fabsf(a.z));
}

__device__ __forceinline__ float max_component(const float3& a) {
    return fmaxf(fmaxf(a.x, a.y), a.z);
}

#endif
