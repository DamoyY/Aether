#include "common.cuh"
#include "random.cuh"

__device__ __forceinline__
bool ray_aabb_intersect(const Ray& ray, const float3& box_min, const float3& box_max,
                        float& t_near, float& t_far) {
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

__device__ __forceinline__
float3 transmittance(float optical_depth, const float3& sigma_t) {
    return make_float3(
        expf(-sigma_t.x * optical_depth),
        expf(-sigma_t.y * optical_depth),
        expf(-sigma_t.z * optical_depth)
    );
}

__device__
float compute_optical_depth(const float3& start, const float3& end,
                            const Voxel* voxels, const VoxelGridParams& params) {
    float3 dir = end - start;
    float dist = length(dir);
    if (dist < 1e-6f) return 0.0f;

    dir = dir / dist;

    float step_size = params.voxel_size * 0.25f;
    int num_steps = (int)ceilf(dist / step_size);
    step_size = dist / (float)num_steps;

    float optical_depth = 0.0f;
    for (int i = 0; i < num_steps; i++) {
        float t = (i + 0.5f) * step_size;
        float3 pos = start + dir * t;
        float density = sample_voxel_trilinear(voxels, pos.x, pos.y, pos.z, params);
        optical_depth += fmaxf(density, 0.0f) * step_size;
    }

    return optical_depth;
}

__device__
float get_max_density(const Voxel* voxels, const VoxelGridParams& params) {
    return 1.0f;
}

__device__
bool delta_tracking_sample(const Ray& ray, const Voxel* voxels,
                           const VoxelGridParams& params,
                           float sigma_t_max, float t_min, float t_max,
                           curandState* state,
                           float& sampled_t, float& sampled_density) {
    float t = t_min;

    while (t < t_max) {
        float free_path = -logf(1.0f - random_float(state) + 1e-10f) / sigma_t_max;
        t += free_path;

        if (t >= t_max) {
            return false;
        }

        float3 pos = ray.origin + ray.direction * t;
        float density = sample_voxel_trilinear(voxels, pos.x, pos.y, pos.z, params);

        float p_scatter = fmaxf(density, 0.0f);
        if (random_float(state) < p_scatter) {
            sampled_t = t;
            sampled_density = density;
            return true;
        }
    }

    return false;
}

__device__
float3 trace_volumetric_path(Ray ray, const Voxel* voxels,
                              const VoxelGridParams& voxel_params,
                              const RenderParams& render_params,
                              curandState* state) {
    float3 color = make_float3(0.0f, 0.0f, 0.0f);
    float3 throughput = make_float3(1.0f, 1.0f, 1.0f);
    float3 sigma_t = render_params.sigma_a + render_params.sigma_s;
    float sigma_t_max = fmaxf(fmaxf(sigma_t.x, sigma_t.y), sigma_t.z);

    float3 box_min = make_float3(voxel_params.origin_x, voxel_params.origin_y, voxel_params.origin_z);
    float3 box_max = box_min + make_float3(
        voxel_params.dim_x * voxel_params.voxel_size,
        voxel_params.dim_y * voxel_params.voxel_size,
        voxel_params.dim_z * voxel_params.voxel_size
    );

    for (unsigned int bounce = 0; ; bounce++) {
        if (bounce > 0) {
            float p_continue = fminf(max_component(throughput), 0.95f);
            if (random_float(state) >= p_continue) {
                break;
            }
            throughput = throughput / p_continue;
        }

        if (bounce >= 10000) break;

        float t_near, t_far;
        if (!ray_aabb_intersect(ray, box_min, box_max, t_near, t_far)) {
            color = color + throughput * render_params.background;
            break;
        }
        t_near = fmaxf(t_near, 0.001f);

        float sampled_t, sampled_density;
        bool scattered = delta_tracking_sample(
            ray, voxels, voxel_params,
            sigma_t_max, t_near, t_far,
            state, sampled_t, sampled_density
        );

        if (!scattered) {
            color = color + throughput * render_params.background;
            break;
        }

        float3 scatter_pos = ray.origin + ray.direction * sampled_t;

        float3 to_light = render_params.light_pos - scatter_pos;
        float light_dist = length(to_light);
        float3 light_dir = to_light / light_dist;

        float optical_depth = compute_optical_depth(
            scatter_pos, render_params.light_pos,
            voxels, voxel_params
        );
        float3 light_transmittance = transmittance(optical_depth, sigma_t);

        float cos_angle = dot(-ray.direction, light_dir);
        float phase = henyey_greenstein_phase(cos_angle, render_params.g);

        float3 scatter_albedo = render_params.sigma_s / sigma_t;
        color = color + throughput * scatter_albedo * phase *
                light_transmittance * render_params.light_color *
                render_params.light_intensity / (light_dist * light_dist);

        ray.origin = scatter_pos;
        ray.direction = sample_hg_phase(state, ray.direction, render_params.g);
        throughput = throughput * scatter_albedo;
    }

    return color;
}

extern "C" __global__
void render_kernel(float4* framebuffer, float4* accumulator,
                   const Voxel* voxels,
                   const VoxelGridParams* voxel_params_ptr,
                   const RenderParams* render_params_ptr) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    VoxelGridParams voxel_params = *voxel_params_ptr;
    RenderParams render_params = *render_params_ptr;

    if (x >= render_params.width || y >= render_params.height) return;

    unsigned int pixel_idx = y * render_params.width + x;

    curandState rand_state;
    init_random(&rand_state, render_params.seed, pixel_idx,
                render_params.current_sample);

    float aspect = (float)render_params.width / (float)render_params.height;
    float fov_scale = tanf(render_params.fov * 0.5f * M_PI / 180.0f);

    float jitter_x = random_float(&rand_state) - 0.5f;
    float jitter_y = random_float(&rand_state) - 0.5f;

    float ndc_x = (2.0f * (x + 0.5f + jitter_x) / render_params.width - 1.0f)
                  * aspect * fov_scale;
    float ndc_y = (1.0f - 2.0f * (y + 0.5f + jitter_y) / render_params.height)
                  * fov_scale;

    Ray ray;
    ray.origin = render_params.camera_pos;
    ray.direction = normalize(
        render_params.camera_forward +
        render_params.camera_right * ndc_x +
        render_params.camera_up * ndc_y
    );

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
        1.0f
    );
}

extern "C" __global__
void clear_accumulator(float4* accumulator, unsigned int width, unsigned int height) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    accumulator[y * width + x] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
}
