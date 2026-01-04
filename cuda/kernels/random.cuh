#ifndef RANDOM_CUH
#define RANDOM_CUH

#include <curand_kernel.h>
#include "common.cuh"

__device__ __forceinline__
void init_random(curandState* state, unsigned int seed,
                 unsigned int pixel_id, unsigned int sample_id) {
    curand_init(seed + pixel_id * 1337 + sample_id * 7919, 0, 0, state);
}

__device__ __forceinline__
float random_float(curandState* state) {
    return curand_uniform(state);
}

__device__ __forceinline__
float random_float(curandState* state, float min_val, float max_val) {
    return min_val + (max_val - min_val) * curand_uniform(state);
}

__device__ __forceinline__
float3 random_in_unit_sphere(curandState* state) {
    float3 p;
    do {
        p = make_float3(
            random_float(state, -1, 1),
            random_float(state, -1, 1),
            random_float(state, -1, 1)
        );
    } while (dot(p, p) >= 1.0f);
    return p;
}

__device__ __forceinline__
float3 random_unit_vector(curandState* state) {
    return normalize(random_in_unit_sphere(state));
}

__device__ __forceinline__
float henyey_greenstein_phase(float cos_theta, float g) {
    float g2 = g * g;
    float denom = 1.0f + g2 - 2.0f * g * cos_theta;
    return (1.0f - g2) / (4.0f * M_PI * powf(denom, 1.5f));
}

__device__ __forceinline__
float3 sample_hg_phase(curandState* state, const float3& wi, float g) {
    float xi1 = random_float(state);
    float xi2 = random_float(state);

    float cos_theta;
    if (fabsf(g) < 1e-3f) {
        cos_theta = 1.0f - 2.0f * xi1;
    } else {
        float sqr_term = (1.0f - g * g) / (1.0f - g + 2.0f * g * xi1);
        cos_theta = (1.0f + g * g - sqr_term * sqr_term) / (2.0f * g);
    }

    float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));
    float phi = 2.0f * M_PI * xi2;

    float3 w = normalize(-wi);
    float3 u = normalize(cross((fabsf(w.x) > 0.1f) ?
                               make_float3(0, 1, 0) : make_float3(1, 0, 0), w));
    float3 v = cross(w, u);

    return normalize(sin_theta * cosf(phi) * u +
                     sin_theta * sinf(phi) * v +
                     cos_theta * w);
}

#endif
