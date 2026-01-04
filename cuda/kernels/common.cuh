#ifndef COMMON_CUH
#define COMMON_CUH

#include <cuda_runtime.h>
#include "../include/vec3.cuh"
#include "../include/voxel.cuh"

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

struct RenderParams {
    unsigned int width;
    unsigned int height;
    unsigned int _pad0[2];

    float3 camera_pos;
    float _pad1;
    float3 camera_forward;
    float _pad2;
    float3 camera_right;
    float _pad3;
    float3 camera_up;
    float fov;

    float3 light_pos;
    float _pad4;
    float3 light_color;
    float light_intensity;

    unsigned int max_bounces;
    unsigned int samples_per_pixel;
    unsigned int current_sample;
    unsigned int _pad5;

    float3 sigma_a;
    float _pad6;
    float3 sigma_s;
    float g;
    float ior;

    unsigned int seed;
    unsigned int _pad7[2];
};

struct Ray {
    float3 origin;
    float3 direction;
};

struct HitRecord {
    float t;
    float3 position;
    float3 normal;
    float voxel_value;
    bool hit;
    bool inside;
};

#endif
