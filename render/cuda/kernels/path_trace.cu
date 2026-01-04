#include "common.cuh"
#include "random.cuh"
__device__ __forceinline__ bool ray_aabb_intersect(const Ray &ray, const float3 &box_min, const float3 &box_max,
                                                   float &t_near, float &t_far)
{
    float3 inv_dir = make_float3(1.0f / ray.direction.x,
                                 1.0f / ray.direction.y,
                                 1.0f / ray.direction.z);
    float3 t0 = (box_min - ray.origin) * inv_dir;
    float3 t1 = (box_max - ray.origin) * inv_dir;
    float3 t_min = fminf3(t0, t1);
    float3 t_max = fmaxf3(t0, t1);
    t_near = fmaxf(fmaxf(t_min.x, t_min.y), t_min.z);
    t_far = fminf(fminf(t_max.x, t_max.y), t_max.z);
    return t_near <= t_far && t_far >= 0.0f;
}
__device__ bool delta_tracking_sample(const Ray &ray, const Voxel *voxels,
                                      const VoxelGridParams &params,
                                      float majorant, float t_min, float t_max,
                                      PCG32State *state,
                                      float &sampled_t, Voxel &sampled_voxel)
{
    float t = t_min;
    while (t < t_max)
    {
        float free_path = -logf(1.0f - random_float(state) + 1e-10f) / majorant;
        t += free_path;
        if (t >= t_max)
        {
            return false;
        }
        float3 pos = ray.origin + ray.direction * t;
        Voxel voxel = sample_voxel(voxels, pos.x, pos.y, pos.z, params);
        float density = fmaxf(voxel.intensity, 0.0f);
        if (random_float(state) < density)
        {
            sampled_t = t;
            sampled_voxel = voxel;
            return true;
        }
    }
    return false;
}
__device__ float3 estimate_transmittance_ratio_tracking(float3 start, float3 end, const Voxel *voxels, const VoxelGridParams &params, float majorant, PCG32State *state)
{
    float3 dir = end - start;
    float dist = length(dir);
    if (dist < 1e-6f)
        return make_float3(1.0f, 1.0f, 1.0f);
    dir = dir / dist;
    float3 throughput = make_float3(1.0f, 1.0f, 1.0f);
    float t = 0.0f;
    while (true)
    {
        float delta_t = -logf(1.0f - random_float(state) + 1e-10f) / majorant;
        t += delta_t;
        if (t >= dist)
            break;
        float3 pos = start + dir * t;
        Voxel voxel = sample_voxel(voxels, pos.x, pos.y, pos.z, params);
        float density = fmaxf(voxel.intensity, 0.0f);
        float3 sigma_t = make_float3(
            voxel.sigma_a[0] + voxel.sigma_s[0],
            voxel.sigma_a[1] + voxel.sigma_s[1],
            voxel.sigma_a[2] + voxel.sigma_s[2]);
        float ratio_x = fmaxf(1.0f - density * sigma_t.x / majorant, 0.0f);
        float ratio_y = fmaxf(1.0f - density * sigma_t.y / majorant, 0.0f);
        float ratio_z = fmaxf(1.0f - density * sigma_t.z / majorant, 0.0f);
        throughput.x *= ratio_x;
        throughput.y *= ratio_y;
        throughput.z *= ratio_z;
        if (max_component(throughput) < 1e-6f)
        {
            throughput = make_float3(0.0f, 0.0f, 0.0f);
            break;
        }
    }
    return throughput;
}
__device__ float3 trace_volumetric_path(Ray ray, const Voxel *voxels,
                                        const VoxelGridParams &voxel_params,
                                        const RenderParams &render_params,
                                        PCG32State *state)
{
    float3 color = make_float3(0.0f, 0.0f, 0.0f);
    float3 throughput = make_float3(1.0f, 1.0f, 1.0f);
    float majorant = render_params.majorant;
    float3 box_min = make_float3(0.0f, 0.0f, 0.0f);
    float3 box_max = make_float3(
                                   voxel_params.dim_x * voxel_params.voxel_size,
                                   voxel_params.dim_y * voxel_params.voxel_size,
                                   voxel_params.dim_z * voxel_params.voxel_size);
    for (unsigned int bounce = 0;; bounce++)
    {
        if (bounce > 0)
        {
            float p_continue = fminf(max_component(throughput), 0.95f);
            if (random_float(state) >= p_continue)
            {
                break;
            }
            throughput = throughput / p_continue;
        }
        float t_near, t_far;
        if (!ray_aabb_intersect(ray, box_min, box_max, t_near, t_far))
        {
            color = color + throughput * render_params.background;
            break;
        }
        t_near = fmaxf(t_near, 0.0f);
        float sampled_t;
        Voxel sampled_voxel;
        bool scattered = delta_tracking_sample(
            ray, voxels, voxel_params,
            majorant, t_near, t_far,
            state, sampled_t, sampled_voxel);
        if (!scattered)
        {
            color = color + throughput * render_params.background;
            break;
        }
        float3 scatter_pos = ray.origin + ray.direction * sampled_t;
        float3 sigma_s = make_float3(sampled_voxel.sigma_s[0], sampled_voxel.sigma_s[1], sampled_voxel.sigma_s[2]);
        float3 sigma_t = make_float3(
            sampled_voxel.sigma_a[0] + sampled_voxel.sigma_s[0],
            sampled_voxel.sigma_a[1] + sampled_voxel.sigma_s[1],
            sampled_voxel.sigma_a[2] + sampled_voxel.sigma_s[2]);
        float g = sampled_voxel.anisotropy;
        float3 to_light = render_params.light_pos - scatter_pos;
        float light_dist = length(to_light);
        float3 light_dir = to_light / light_dist;
        float3 light_transmittance = estimate_transmittance_ratio_tracking(
            scatter_pos, render_params.light_pos,
            voxels, voxel_params,
            majorant,
            state);
        float cos_angle = dot(-ray.direction, light_dir);
        float phase = henyey_greenstein_phase(cos_angle, g);
        float sigma_t_max = fmaxf(fmaxf(sigma_t.x, sigma_t.y), sigma_t.z);
        float3 effective_weight = sigma_s / sigma_t_max;
        color = color + throughput * effective_weight * phase *
                            light_transmittance * render_params.light_color *
                            render_params.light_intensity / (light_dist * light_dist);
        ray.origin = scatter_pos;
        ray.direction = sample_hg_phase(state, ray.direction, g);
        throughput = throughput * effective_weight;
    }
    return color;
}
extern "C" __global__ void render_kernel(float4 *framebuffer, float4 *accumulator,
                                         const Voxel *voxels,
                                         const VoxelGridParams *voxel_params_ptr,
                                         const RenderParams *render_params_ptr)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    VoxelGridParams voxel_params = *voxel_params_ptr;
    RenderParams render_params = *render_params_ptr;
    if (x >= render_params.width || y >= render_params.height)
        return;
    unsigned int pixel_idx = y * render_params.width + x;
    PCG32State rand_state;
    init_random(&rand_state, render_params.seed, pixel_idx,
                render_params.current_sample);
    float aspect = (float)render_params.width / (float)render_params.height;
    float fov_scale = tanf(render_params.fov * 0.5f * M_PI / 180.0f);
    float jitter_x = random_float(&rand_state) - 0.5f;
    float jitter_y = random_float(&rand_state) - 0.5f;
    float ndc_x = (2.0f * (x + 0.5f + jitter_x) / render_params.width - 1.0f) * aspect * fov_scale;
    float ndc_y = (1.0f - 2.0f * (y + 0.5f + jitter_y) / render_params.height) * fov_scale;
    Ray ray;
    ray.origin = render_params.camera_pos;
    ray.direction = normalize(
        render_params.camera_forward +
        render_params.camera_right * ndc_x +
        render_params.camera_up * ndc_y);
    float3 color = trace_volumetric_path(ray, voxels, voxel_params, render_params, &rand_state);
    float4 acc = accumulator[pixel_idx];
    acc.x += color.x;
    acc.y += color.y;
    acc.z += color.z;
    acc.w += 1.0f;
    accumulator[pixel_idx] = acc;
    float inv_samples = 1.0f / acc.w;
    framebuffer[pixel_idx] = make_float4(
        acc.x * inv_samples,
        acc.y * inv_samples,
        acc.z * inv_samples,
        1.0f);
}
extern "C" __global__ void clear_accumulator(float4 *accumulator, unsigned int width, unsigned int height)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;
    accumulator[y * width + x] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
}
