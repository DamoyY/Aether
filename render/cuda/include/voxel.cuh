#ifndef VOXEL_CUH
#define VOXEL_CUH
struct Voxel
{
    float intensity;
    float sigma_a[3];
    float sigma_s[3];
    float anisotropy;
    float ior;
};
struct VoxelGridParams
{
    unsigned int dim_x;
    unsigned int dim_y;
    unsigned int dim_z;
    float voxel_size;
};
__device__ __forceinline__ Voxel sample_voxel(
    const Voxel *voxels,
    float wx, float wy, float wz,
    const VoxelGridParams &params)
{
    int x = (int)(wx / params.voxel_size);
    int y = (int)(wy / params.voxel_size);
    int z = (int)(wz / params.voxel_size);
    if (x < 0 || x >= (int)params.dim_x ||
        y < 0 || y >= (int)params.dim_y ||
        z < 0 || z >= (int)params.dim_z)
    {
        Voxel empty;
        empty.intensity = 0.0f;
        empty.sigma_a[0] = empty.sigma_a[1] = empty.sigma_a[2] = 0.0f;
        empty.sigma_s[0] = empty.sigma_s[1] = empty.sigma_s[2] = 0.0f;
        empty.anisotropy = 0.0f;
        empty.ior = 1.0f;
        return empty;
    }
    int idx = z * params.dim_y * params.dim_x + y * params.dim_x + x;
    return voxels[idx];
}
#endif
