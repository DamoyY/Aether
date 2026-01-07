#define MAX_DEGREE 32
extern "C" __constant__ float c_rad_coeffs[MAX_DEGREE];
extern "C" __constant__ float c_ang_coeffs[MAX_DEGREE];
__device__ __forceinline__ float eval_poly_horner_const(float x, const float *coeffs, int degree)
{
    float res = coeffs[degree];
    for (int i = degree - 1; i >= 0; --i)
    {
        res = fmaf(res, x, coeffs[i]);
    }
    return res;
}
__device__ __forceinline__ float pow_int(float base, int exp)
{
    float res = 1.0f;
    float acc = base;
    while (exp > 0)
    {
        if (exp & 1)
            res *= acc;
        acc *= acc;
        exp >>= 1;
    }
    return res;
}
extern "C" __global__ void compute_density(
    float *__restrict__ output_grid,
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
    float centerX = dimX * 0.5f;
    float centerY = dimY * 0.5f;
    float centerZ = dimZ * 0.5f;
    float x = (vx - centerX) * voxel_size;
    float y = (vy - centerY) * voxel_size;
    float z = (vz - centerZ) * voxel_size;
    float r2 = fmaf(x, x, fmaf(y, y, z * z));
    float r = sqrtf(r2);
    float rho = r * scale;
    float cos_theta = (r > 1e-10f) ? __fdividef(z, r) : 1.0f;
    float R_poly = eval_poly_horner_const(rho, c_rad_coeffs, rad_deg);
    float Y_poly = eval_poly_horner_const(cos_theta, c_ang_coeffs, ang_deg);
    float sin_sq = 1.0f - cos_theta * cos_theta;
    int abs_m = (m >= 0) ? m : -m;
    float phi_part = 1.0f;
    if (abs_m > 0)
    {
        float rho_xy2 = fmaf(x, x, y * y);
        if (rho_xy2 > 1e-20f)
        {
            float rho_xy = sqrtf(rho_xy2);
            float inv_rho = 1.0f / rho_xy;
            float u = x * inv_rho;
            float v = y * inv_rho;
            float re = 1.0f;
            float im = 0.0f;
            for (int k = 0; k < abs_m; ++k)
            {
                float n_re = re * u - im * v;
                float n_im = re * v + im * u;
                re = n_re;
                im = n_im;
            }
            if (m > 0)
                phi_part = re;
            else
                phi_part = im;
        }
        else
        {
            phi_part = 0.0f;
        }
    }
    float rho_pow = (l > 0) ? pow_int(rho, l) : 1.0f;
    float sin_pow = 1.0f;
    if (abs_m > 0)
    {
        int p = abs_m >> 1;
        sin_pow = pow_int(sin_sq, p);
        if ((abs_m & 1) != 0)
            sin_pow *= sqrtf(sin_sq);
    }
    float exp_val = __expf(-rho * 0.5f);
    float psi = sqrtf(prefactor) * exp_val * rho_pow * sin_pow * R_poly * Y_poly * phi_part;
    size_t idx = (size_t)vz * dimY * dimX + (size_t)vy * dimX + vx;
    output_grid[idx] = psi;
}
