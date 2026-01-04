#ifndef RANDOM_CUH
#define RANDOM_CUH
#include "common.cuh"
struct PCG32State
{
    unsigned long long state;
    unsigned long long inc;
};
__device__ __forceinline__ unsigned int pcg32_random(PCG32State *rng)
{
    unsigned long long oldstate = rng->state;
    rng->state = oldstate * 6364136223846793005ULL + rng->inc;
    unsigned int xorshifted = (unsigned int)(((oldstate >> 18u) ^ oldstate) >> 27u);
    unsigned int rot = (unsigned int)(oldstate >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31));
}
__device__ __forceinline__ void init_random(PCG32State *state, unsigned int seed,
                                            unsigned int pixel_id, unsigned int sample_id)
{
    state->state = 0ULL;
    state->inc = ((unsigned long long)(pixel_id * 1337 + sample_id * 7919) << 1u) | 1u;
    pcg32_random(state);
    state->state += (unsigned long long)seed;
    pcg32_random(state);
}
__device__ __forceinline__ float random_float(PCG32State *state)
{
    return (float)pcg32_random(state) / 4294967296.0f;
}
__device__ __forceinline__ float random_float(PCG32State *state, float min_val, float max_val)
{
    return min_val + (max_val - min_val) * random_float(state);
}
__device__ __forceinline__
    float3
    random_in_unit_sphere(PCG32State *state)
{
    float3 p;
    do
    {
        p = make_float3(
            random_float(state, -1, 1),
            random_float(state, -1, 1),
            random_float(state, -1, 1));
    } while (dot(p, p) >= 1.0f);
    return p;
}
__device__ __forceinline__
    float3
    random_unit_vector(PCG32State *state)
{
    return normalize(random_in_unit_sphere(state));
}
__device__ __forceinline__ float henyey_greenstein_phase(float cos_theta, float g)
{
    float g2 = g * g;
    float denom = 1.0f + g2 - 2.0f * g * cos_theta;
    return (1.0f - g2) / (4.0f * M_PI * powf(denom, 1.5f));
}
__device__ __forceinline__
    float3
    sample_hg_phase(PCG32State *state, const float3 &wi, float g)
{
    float xi1 = random_float(state);
    float xi2 = random_float(state);
    float cos_theta;
    if (fabsf(g) < 1e-3f)
    {
        cos_theta = 1.0f - 2.0f * xi1;
    }
    else
    {
        float sqr_term = (1.0f - g * g) / (1.0f - g + 2.0f * g * xi1);
        cos_theta = (1.0f + g * g - sqr_term * sqr_term) / (2.0f * g);
    }
    float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));
    float phi = 2.0f * M_PI * xi2;
    float3 w = normalize(-wi);
    float3 u = normalize(cross((fabsf(w.x) > 0.1f) ? make_float3(0, 1, 0) : make_float3(1, 0, 0), w));
    float3 v = cross(w, u);
    return normalize(sin_theta * cosf(phi) * u +
                     sin_theta * sinf(phi) * v +
                     cos_theta * w);
}
#endif
