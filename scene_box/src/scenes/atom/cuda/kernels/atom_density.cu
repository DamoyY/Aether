#define MAX_DEGREE 32
__device__ __forceinline__ float eval_poly_horner(float x, const float *__restrict__ coeffs, int degree)
{
    float res = coeffs[degree];
    for (int i = degree - 1; i >= 0; --i)
    {
        res = fmaf(res, x, coeffs[i]);
    }
    return res;
}
extern "C" __global__ void compute_atom_density(
    float *__restrict__ output_grid,
    const float *__restrict__ rad_coeffs,
    const float *__restrict__ ang_coeffs,
    const int rad_deg,
    const int ang_deg,
    const float prefactor,
    const float scale,
    const int dimX, const int dimY, const int dimZ,
    const float voxel_size,
    const int l, const int m)
{
    int vx = blockIdx.x * blockDim.x + threadIdx.x;
    int vy = blockIdx.y * blockDim.y + threadIdx.y;
    int vz = blockIdx.z;
    if (vx >= dimX || vy >= dimY || vz >= dimZ)
        return;
    float centerX = dimX / 2.0f;
    float centerY = dimY / 2.0f;
    float centerZ = dimZ / 2.0f;
    float x = (vx - centerX) * voxel_size;
    float y = (vy - centerY) * voxel_size;
    float z = (vz - centerZ) * voxel_size;
    float r2 = fmaf(x, x, fmaf(y, y, z * z));
    float r = sqrtf(r2);
    float rho = r * scale;
    float cos_theta = (r > 1e-10f) ? __fdividef(z, r) : 1.0f;
    float R_poly = eval_poly_horner(rho, rad_coeffs, rad_deg);
    float Y_poly = eval_poly_horner(cos_theta, ang_coeffs, ang_deg);
    float sin_sq = 1.0f - cos_theta * cos_theta;
    int abs_m = (m >= 0) ? m : -m;
    float phi = atan2f(y, x);
    float phi_part = 1.0f;
    if (m > 0)
        phi_part = cosf((float)m * phi);
    else if (m < 0)
        phi_part = sinf((float)abs_m * phi);
    float rho_pow = 1.0f;
    if (l > 0)
        rho_pow = powf(rho, (float)l);
    float sin_pow = 1.0f;
    if (abs_m > 0)
        sin_pow = powf(sin_sq, (float)abs_m * 0.5f);
    float exp_val = __expf(-rho * 0.5f);
    float psi = sqrtf(prefactor) * exp_val * rho_pow * sin_pow * R_poly * Y_poly * phi_part;
    size_t idx = (size_t)vz * dimY * dimX + (size_t)vy * dimX + vx;
    output_grid[idx] = psi;
}
