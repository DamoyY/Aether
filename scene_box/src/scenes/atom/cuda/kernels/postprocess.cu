#define BLOCK_SIZE 256

struct Voxel
{
    float intensity;
    float albedo[3];
    float sigma_t[3];
    float anisotropy;
    float ior;
};

struct MaterialParams
{
    float hue_positive;
    float hue_negative;
    float saturation;
    float value;
    float base_sigma_t;
    float anisotropy;
    float ior;
};

__device__ __forceinline__ void hsv_to_rgb(float hue_input, float saturation, float value, float *rgb)
{
    float hue = fmodf(hue_input, 360.0f);
    if (hue < 0.0f)
        hue += 360.0f;
    float chroma = value * saturation;
    float x_val = chroma * (1.0f - fabsf(fmodf(hue / 60.0f, 2.0f) - 1.0f));
    float m_val = value - chroma;
    float r, g, b;
    if (hue < 60.0f)
    {
        r = chroma;
        g = x_val;
        b = 0.0f;
    }
    else if (hue < 120.0f)
    {
        r = x_val;
        g = chroma;
        b = 0.0f;
    }
    else if (hue < 180.0f)
    {
        r = 0.0f;
        g = chroma;
        b = x_val;
    }
    else if (hue < 240.0f)
    {
        r = 0.0f;
        g = x_val;
        b = chroma;
    }
    else if (hue < 300.0f)
    {
        r = x_val;
        g = 0.0f;
        b = chroma;
    }
    else
    {
        r = chroma;
        g = 0.0f;
        b = x_val;
    }
    rgb[0] = r + m_val;
    rgb[1] = g + m_val;
    rgb[2] = b + m_val;
}

__device__ __forceinline__ float lerp_hue(float factor, float hue_neg, float hue_pos)
{
    float t_normalized = factor * 0.5f + 0.5f;
    float diff = hue_pos - hue_neg;
    float long_diff;
    if (fabsf(diff) > 180.0f)
        long_diff = diff;
    else if (diff >= 0.0f)
        long_diff = diff - 360.0f;
    else
        long_diff = diff + 360.0f;
    float result = hue_neg + long_diff * t_normalized;
    return fmodf(result + 360.0f, 360.0f);
}

extern "C" __global__ void reduce_max_abs(
    const float *__restrict__ input,
    float *__restrict__ output,
    const int n)
{
    __shared__ float sdata[BLOCK_SIZE];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + tid;
    unsigned int grid_size = blockDim.x * 2 * gridDim.x;
    float max_val = 0.0f;
    while (i < n)
    {
        max_val = fmaxf(max_val, fabsf(input[i]));
        if (i + blockDim.x < n)
            max_val = fmaxf(max_val, fabsf(input[i + blockDim.x]));
        i += grid_size;
    }
    sdata[tid] = max_val;
    __syncthreads();
    if (BLOCK_SIZE >= 512 && tid < 256)
        sdata[tid] = fmaxf(sdata[tid], sdata[tid + 256]);
    __syncthreads();
    if (BLOCK_SIZE >= 256 && tid < 128)
        sdata[tid] = fmaxf(sdata[tid], sdata[tid + 128]);
    __syncthreads();
    if (BLOCK_SIZE >= 128 && tid < 64)
        sdata[tid] = fmaxf(sdata[tid], sdata[tid + 64]);
    __syncthreads();
    if (tid < 32)
    {
        volatile float *smem = sdata;
        smem[tid] = fmaxf(smem[tid], smem[tid + 32]);
        smem[tid] = fmaxf(smem[tid], smem[tid + 16]);
        smem[tid] = fmaxf(smem[tid], smem[tid + 8]);
        smem[tid] = fmaxf(smem[tid], smem[tid + 4]);
        smem[tid] = fmaxf(smem[tid], smem[tid + 2]);
        smem[tid] = fmaxf(smem[tid], smem[tid + 1]);
    }
    if (tid == 0)
        output[blockIdx.x] = sdata[0];
}

extern "C" __global__ void finalize_voxels(
    const float *__restrict__ psi_values,
    Voxel *__restrict__ voxels,
    const float max_abs_psi,
    const MaterialParams material,
    const int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;
    float psi_val = psi_values[idx];
    float normalizer = (max_abs_psi > 0.0f) ? (1.0f / max_abs_psi) : 1.0f;
    float normalized_psi = psi_val * normalizer;
    float intensity = fabsf(normalized_psi);
    float hue = lerp_hue(normalized_psi, material.hue_negative, material.hue_positive);
    float rgb[3];
    hsv_to_rgb(hue, material.saturation, material.value, rgb);
    float sigma_t_val = material.base_sigma_t * intensity;
    Voxel v;
    v.intensity = intensity;
    v.albedo[0] = rgb[0];
    v.albedo[1] = rgb[1];
    v.albedo[2] = rgb[2];
    v.sigma_t[0] = sigma_t_val;
    v.sigma_t[1] = sigma_t_val;
    v.sigma_t[2] = sigma_t_val;
    v.anisotropy = material.anisotropy;
    v.ior = material.ior;
    voxels[idx] = v;
}
