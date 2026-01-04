#ifndef VOXEL_CUH
#define VOXEL_CUH
struct Voxel
{
    float intensity;
};
struct VoxelGridParams
{
    unsigned int dim_x;
    unsigned int dim_y;
    unsigned int dim_z;
    float voxel_size;
    float origin_x;
    float origin_y;
    float origin_z;
    unsigned int _padding;
};
__device__ __forceinline__ float sample_voxel_texture(cudaTextureObject_t tex, float wx, float wy, float wz, const VoxelGridParams &params)
{
    float vx = (wx - params.origin_x) / params.voxel_size + 0.5f;
    float vy = (wy - params.origin_y) / params.voxel_size + 0.5f;
    float vz = (wz - params.origin_z) / params.voxel_size + 0.5f;
    return tex3D<float>(tex, vx, vy, vz);
}
#endif
