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
__device__ __forceinline__ unsigned int voxel_index(unsigned int x, unsigned int y, unsigned int z, const VoxelGridParams &params)
{
    return z * params.dim_y * params.dim_x + y * params.dim_x + x;
}
__device__ __forceinline__ float sample_voxel(const Voxel *voxels, int x, int y, int z, const VoxelGridParams &params)
{
    if (x < 0 || x >= (int)params.dim_x ||
        y < 0 || y >= (int)params.dim_y ||
        z < 0 || z >= (int)params.dim_z)
    {
        return 0.0f;
    }
    return voxels[voxel_index(x, y, z, params)].intensity;
}
__device__ __forceinline__ float sample_voxel_trilinear(const Voxel *voxels, float wx, float wy, float wz, const VoxelGridParams &params)
{
    float vx = (wx - params.origin_x) / params.voxel_size;
    float vy = (wy - params.origin_y) / params.voxel_size;
    float vz = (wz - params.origin_z) / params.voxel_size;
    int x0 = (int)floorf(vx), y0 = (int)floorf(vy), z0 = (int)floorf(vz);
    float fx = vx - x0, fy = vy - y0, fz = vz - z0;
    float c000 = sample_voxel(voxels, x0, y0, z0, params);
    float c001 = sample_voxel(voxels, x0, y0, z0 + 1, params);
    float c010 = sample_voxel(voxels, x0, y0 + 1, z0, params);
    float c011 = sample_voxel(voxels, x0, y0 + 1, z0 + 1, params);
    float c100 = sample_voxel(voxels, x0 + 1, y0, z0, params);
    float c101 = sample_voxel(voxels, x0 + 1, y0, z0 + 1, params);
    float c110 = sample_voxel(voxels, x0 + 1, y0 + 1, z0, params);
    float c111 = sample_voxel(voxels, x0 + 1, y0 + 1, z0 + 1, params);
    float c00 = c000 * (1 - fx) + c100 * fx;
    float c01 = c001 * (1 - fx) + c101 * fx;
    float c10 = c010 * (1 - fx) + c110 * fx;
    float c11 = c011 * (1 - fx) + c111 * fx;
    float c0 = c00 * (1 - fy) + c10 * fy;
    float c1 = c01 * (1 - fy) + c11 * fy;
    return c0 * (1 - fz) + c1 * fz;
}
#endif
