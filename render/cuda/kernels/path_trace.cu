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
__device__ bool delta_tracking_sample(const Ray &ray, cudaTextureObject_t density_tex,
                                      const VoxelMaterial *materials,
                                      const VoxelGridParams &params,
                                      float majorant, float t_min, float t_max,
                                      int hero_channel,
                                      PCG32State *state,
                                      float &sampled_t, VoxelMaterial &sampled_material)
{
    float t = t_min;
    while (t < t_max)
    {
        float free_path = -logf(1.0f - random_float(state)) / majorant;
        t += free_path;
        if (t >= t_max)
        {
            return false;
        }
        float3 pos = ray.origin + ray.direction * t;
        float density = fmaxf(sample_density(density_tex, pos.x, pos.y, pos.z, params), 0.0f);
        if (density <= 0.0f)
        {
            continue;
        }
        VoxelMaterial mat = sample_material(materials, pos.x, pos.y, pos.z, params);
        float sigma_t_hero = mat.sigma_t[hero_channel];
        float local_sigma_t = density * sigma_t_hero;
        if (random_float(state) < local_sigma_t / majorant)
        {
            sampled_t = t;
            sampled_material = mat;
            return true;
        }
    }
    return false;
}
__device__ float3 estimate_transmittance_ratio_tracking(float3 start, float3 end, cudaTextureObject_t density_tex, const VoxelMaterial *materials, const VoxelGridParams &params, float majorant, PCG32State *state)
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
        float density = fmaxf(sample_density(density_tex, pos.x, pos.y, pos.z, params), 0.0f);
        if (density <= 0.0f)
        {
            continue;
        }
        VoxelMaterial mat = sample_material(materials, pos.x, pos.y, pos.z, params);
        float3 sigma_t = make_float3(
            mat.sigma_t[0],
            mat.sigma_t[1],
            mat.sigma_t[2]);
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
__device__ float3 trace_volumetric_path(Ray ray, cudaTextureObject_t density_tex,
                                        const VoxelMaterial *materials,
                                        const VoxelGridParams &voxel_params,
                                        const RenderParams &render_params,
                                        unsigned int pixel_idx,
                                        PCG32State *state)
{
    float3 color = make_float3(0.0f, 0.0f, 0.0f);
    float majorant = render_params.majorant;
    float3 box_min = make_float3(0.0f, 0.0f, 0.0f);
    float3 box_max = make_float3(
        voxel_params.dim_x * voxel_params.voxel_size,
        voxel_params.dim_y * voxel_params.voxel_size,
        voxel_params.dim_z * voxel_params.voxel_size);
    int hero_channel = (pixel_idx + render_params.current_sample) % 3;
    float hero_throughput = 1.0f;
    for (unsigned int bounce = 0;; bounce++)
    {
        if (bounce > 0)
        {
            float p_continue = fminf(hero_throughput, 0.95f);
            if (random_float(state) >= p_continue)
            {
                break;
            }
            hero_throughput = hero_throughput / p_continue;
        }
        float t_near, t_far;
        if (!ray_aabb_intersect(ray, box_min, box_max, t_near, t_far))
        {
            float bg_value = (hero_channel == 0) ? render_params.background.x : (hero_channel == 1) ? render_params.background.y
                                                                                                    : render_params.background.z;
            float hero_contrib = hero_throughput * bg_value;
            if (hero_channel == 0)
                color.x += hero_contrib;
            else if (hero_channel == 1)
                color.y += hero_contrib;
            else
                color.z += hero_contrib;
            break;
        }
        t_near = fmaxf(t_near, 0.0f);
        float sampled_t;
        VoxelMaterial sampled_material;
        bool scattered = delta_tracking_sample(
            ray, density_tex, materials, voxel_params,
            majorant, t_near, t_far,
            hero_channel,
            state, sampled_t, sampled_material);
        if (!scattered)
        {
            float bg_value = (hero_channel == 0) ? render_params.background.x : (hero_channel == 1) ? render_params.background.y
                                                                                                    : render_params.background.z;
            float hero_contrib = hero_throughput * bg_value;
            if (hero_channel == 0)
                color.x += hero_contrib;
            else if (hero_channel == 1)
                color.y += hero_contrib;
            else
                color.z += hero_contrib;
            break;
        }
        float3 scatter_pos = ray.origin + ray.direction * sampled_t;
        float scattering_albedo = sampled_material.albedo[hero_channel];
        float g = sampled_material.anisotropy;
        float3 to_light = render_params.light_pos - scatter_pos;
        float light_dist = length(to_light);
        float3 light_dir = to_light / light_dist;
        float3 light_transmittance = estimate_transmittance_ratio_tracking(
            scatter_pos, render_params.light_pos,
            density_tex, materials, voxel_params,
            majorant,
            state);
        float cos_angle = dot(-ray.direction, light_dir);
        float phase = henyey_greenstein_phase(cos_angle, g);
        float3 light_color_scaled = render_params.light_color * render_params.light_intensity / (light_dist * light_dist);
        float3 incoming_light = light_transmittance * light_color_scaled;
        float light_comp = (hero_channel == 0) ? incoming_light.x : (hero_channel == 1) ? incoming_light.y
                                                                                        : incoming_light.z;
        float current_contrib = light_comp * phase * scattering_albedo * hero_throughput;
        if (hero_channel == 0)
            color.x += current_contrib;
        else if (hero_channel == 1)
            color.y += current_contrib;
        else
            color.z += current_contrib;
        ray.origin = scatter_pos;
        ray.direction = sample_hg_phase(state, ray.direction, g);
        hero_throughput = hero_throughput * scattering_albedo;
    }
    return color * 3.0f;
}
extern "C" __global__ void render_kernel(float4 *framebuffer, float4 *accumulator,
                                         cudaTextureObject_t density_tex,
                                         const VoxelMaterial *materials,
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
    float3 color = trace_volumetric_path(ray, density_tex, materials, voxel_params, render_params, pixel_idx, &rand_state);
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
