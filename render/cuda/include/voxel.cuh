#ifndef VOXEL_CUH
#define VOXEL_CUH

// Material buffer: only accessed after real collision
struct VoxelMaterial
{
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

// Sample density using 3D texture with hardware interpolation and border handling
__device__ __forceinline__ float sample_density(
    cudaTextureObject_t density_tex,
    float wx, float wy, float wz,
    const VoxelGridParams &params)
{
    // Convert world coordinates to texture coordinates
    // With unnormalized coordinates, 0.5 is the center of the first voxel
    float tx = wx / params.voxel_size + 0.5f;
    float ty = wy / params.voxel_size + 0.5f;
    float tz = wz / params.voxel_size + 0.5f;
    // Hardware texture sampling handles boundary (returns 0.0 for out-of-bounds)
    return tex3D<float>(density_tex, tx, ty, tz);
}

__device__ __forceinline__ VoxelMaterial sample_material(
    const VoxelMaterial *materials,
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
        VoxelMaterial empty;
        empty.sigma_a[0] = empty.sigma_a[1] = empty.sigma_a[2] = 0.0f;
        empty.sigma_s[0] = empty.sigma_s[1] = empty.sigma_s[2] = 0.0f;
        empty.anisotropy = 0.0f;
        empty.ior = 1.0f;
        return empty;
    }
    int idx = z * params.dim_y * params.dim_x + y * params.dim_x + x;
    return materials[idx];
}

__device__ __forceinline__ int voxel_index(
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
        return -1;
    }
    return z * params.dim_y * params.dim_x + y * params.dim_x + x;
}

#endif
