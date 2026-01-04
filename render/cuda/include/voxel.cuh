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
};
__device__ __forceinline__ float sample_voxel_texture(cudaTextureObject_t tex, float wx, float wy, float wz, const VoxelGridParams &params)
{
    float vx = wx / params.voxel_size + 0.5f;
    float vy = wy / params.voxel_size + 0.5f;
    float vz = wz / params.voxel_size + 0.5f;
    return tex3D<float>(tex, vx, vy, vz);
}
#endif
