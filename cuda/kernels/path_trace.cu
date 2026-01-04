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

__device__
HitRecord ray_march_voxels(const Ray& ray, const Voxel* voxels,
                           const VoxelGridParams& params, float max_dist) {
    HitRecord hit;
    hit.hit = false;

    float3 box_min = make_float3(params.origin_x, params.origin_y, params.origin_z);
    float3 box_max = box_min + make_float3(
        params.dim_x * params.voxel_size,
        params.dim_y * params.voxel_size,
        params.dim_z * params.voxel_size
    );

    float t_near, t_far;
    if (!ray_aabb_intersect(ray, box_min, box_max, t_near, t_far)) {
        return hit;
    }

    t_near = fmaxf(t_near, 0.001f);
    t_far = fminf(t_far, max_dist);

    float step_size = params.voxel_size * 0.5f;
    float t = t_near;

    while (t < t_far) {
        float3 pos = ray.origin + ray.direction * t;
        float density = sample_voxel_trilinear(voxels, pos.x, pos.y, pos.z, params);

        if (density > 0.01f) {
            hit.hit = true;
            hit.t = t;
            hit.position = pos;
            hit.voxel_value = density;

            float eps = params.voxel_size * 0.5f;
            hit.normal = normalize(make_float3(
                sample_voxel_trilinear(voxels, pos.x + eps, pos.y, pos.z, params) -
                sample_voxel_trilinear(voxels, pos.x - eps, pos.y, pos.z, params),
                sample_voxel_trilinear(voxels, pos.x, pos.y + eps, pos.z, params) -
                sample_voxel_trilinear(voxels, pos.x, pos.y - eps, pos.z, params),
                sample_voxel_trilinear(voxels, pos.x, pos.y, pos.z + eps, params) -
                sample_voxel_trilinear(voxels, pos.x, pos.y, pos.z - eps, params)
            ));

            if (dot(hit.normal, ray.direction) > 0) {
                hit.normal = -hit.normal;
                hit.inside = true;
            } else {
                hit.inside = false;
            }

            return hit;
        }

        t += step_size;
    }

    return hit;
}

__device__ __forceinline__
float3 transmittance(float distance, const float3& sigma_t) {
    return make_float3(
        expf(-sigma_t.x * distance),
        expf(-sigma_t.y * distance),
        expf(-sigma_t.z * distance)
    );
}

__device__ __forceinline__
float sample_free_path(curandState* state, float sigma_t) {
    return -logf(1.0f - random_float(state) + 1e-10f) / sigma_t;
}

__device__
float3 volume_scatter(Ray ray, const Voxel* voxels,
                      const VoxelGridParams& voxel_params,
                      const RenderParams& render_params,
                      curandState* state, int max_scatter_events) {
    float3 throughput = make_float3(1.0f, 1.0f, 1.0f);
    float3 radiance = make_float3(0.0f, 0.0f, 0.0f);
    float3 sigma_t = render_params.sigma_a + render_params.sigma_s;
    float sigma_t_avg = (sigma_t.x + sigma_t.y + sigma_t.z) / 3.0f;

    for (int scatter = 0; scatter < max_scatter_events; scatter++) {
        float free_path = sample_free_path(state, sigma_t_avg);
        float3 scatter_pos = ray.origin + ray.direction * free_path;

        float density = sample_voxel_trilinear(
            voxels, scatter_pos.x, scatter_pos.y, scatter_pos.z, voxel_params
        );

        if (density <= 0.0f) break;

        throughput = throughput * transmittance(free_path, sigma_t);

        float p_continue = max_component(throughput);
        if (random_float(state) > p_continue) break;
        throughput = throughput / p_continue;

        float3 to_light = render_params.light_pos - scatter_pos;
        float light_dist = length(to_light);
        float3 light_dir = to_light / light_dist;

        float3 light_transmittance = transmittance(light_dist * density * 0.5f, sigma_t);

        float cos_angle = dot(-ray.direction, light_dir);
        float phase = henyey_greenstein_phase(cos_angle, render_params.g);

        float3 scatter_albedo = render_params.sigma_s / sigma_t;
        radiance = radiance + throughput * scatter_albedo * phase *
                   light_transmittance * render_params.light_color *
                   render_params.light_intensity / (light_dist * light_dist);

        ray.origin = scatter_pos;
        ray.direction = sample_hg_phase(state, ray.direction, render_params.g);
        throughput = throughput * scatter_albedo;
    }

    return radiance;
}

__device__ __forceinline__
bool refract_ray(const float3& v, const float3& n, float ni_over_nt, float3& refracted) {
    float3 uv = normalize(v);
    float dt = dot(uv, n);
    float discriminant = 1.0f - ni_over_nt * ni_over_nt * (1.0f - dt * dt);

    if (discriminant > 0) {
        refracted = ni_over_nt * (uv - n * dt) - n * sqrtf(discriminant);
        return true;
    }
    return false;
}

__device__ __forceinline__
float fresnel_schlick(float cosine, float ior) {
    float r0 = (1.0f - ior) / (1.0f + ior);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * powf(1.0f - cosine, 5.0f);
}

__device__
float3 trace_path(Ray ray, const Voxel* voxels,
                  const VoxelGridParams& voxel_params,
                  const RenderParams& render_params,
                  curandState* state) {
    float3 color = make_float3(0.0f, 0.0f, 0.0f);
    float3 throughput = make_float3(1.0f, 1.0f, 1.0f);
    float3 sigma_t = render_params.sigma_a + render_params.sigma_s;
    bool inside_medium = false;

    for (unsigned int bounce = 0; ; bounce++) {
        if (bounce > 0) {
            float p_continue = fminf(max_component(throughput), 0.95f);
            if (random_float(state) >= p_continue) {
                break;
            }
            throughput = throughput / p_continue;
        }

        if (bounce >= 10000) break;

        HitRecord hit = ray_march_voxels(ray, voxels, voxel_params, 1000.0f);

        if (!hit.hit) {
            color = color + throughput * render_params.background;
            break;
        }

        if (inside_medium) {
            throughput = throughput * transmittance(hit.t, sigma_t);
        }

        float eta = hit.inside ? render_params.ior : (1.0f / render_params.ior);
        float cos_theta = fminf(dot(-ray.direction, hit.normal), 1.0f);
        float fresnel = fresnel_schlick(cos_theta, render_params.ior);

        if (random_float(state) < fresnel) {
            float3 reflected = reflect(ray.direction, hit.normal);
            ray.origin = hit.position + hit.normal * 0.001f;
            ray.direction = reflected;
        } else {
            float3 refracted;
            if (refract_ray(ray.direction, hit.normal, eta, refracted)) {
                ray.origin = hit.position - hit.normal * 0.001f;
                ray.direction = normalize(refracted);

                if (!hit.inside) {
                    inside_medium = true;
                    float3 sss_contribution = volume_scatter(
                        ray, voxels, voxel_params, render_params, state, 10
                    );
                    color = color + throughput * sss_contribution;
                } else {
                    inside_medium = false;
                }
            } else {
                float3 reflected = reflect(ray.direction, hit.normal);
                ray.origin = hit.position + hit.normal * 0.001f;
                ray.direction = reflected;
            }
        }
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

    float3 color = trace_path(ray, voxels, voxel_params, render_params, &rand_state);

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
