use core::{
    f32::consts::PI,
    ops::{Add as _, Div as _, Mul as _, Sub as _},
};

use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use cudarc::{
    driver::{CudaContext, DeviceRepr, LaunchConfig, PushKernelArg as _},
    nvrtc::Ptx,
};

use super::config::{MaterialConfig, OrbitalConfig};
use crate::Voxel;
const MAX_DEGREE: usize = 32;
const BLOCK_SIZE: u32 = 256;
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct MaterialParams {
    hue_positive: f32,
    hue_negative: f32,
    saturation: f32,
    value: f32,
    base_sigma_t: f32,
    anisotropy: f32,
    ior: f32,
}
// SAFETY: MaterialParams is #[repr(C)] and contains only f32 which is valid for GPU transfer.
unsafe impl DeviceRepr for MaterialParams {}
fn factorial(n: u32) -> f32 {
    let mut result = 1.0_f32;
    for i in 2..=n {
        let i_f: f32 = u16::try_from(i).unwrap_or(u16::MAX).into();
        result = result.mul(i_f);
    }
    result
}
fn get_laguerre_coeffs(n_param: u32, l_param: u32) -> Vec<f32> {
    let n_val = i32::try_from(n_param).unwrap_or(i32::MAX);
    let l_val = i32::try_from(l_param).unwrap_or(i32::MAX);
    let numerator = n_val.saturating_sub(l_val).saturating_sub(1_i32);
    let alpha = l_val.saturating_mul(2_i32).saturating_add(1_i32);
    if numerator < 0_i32 {
        return vec![1.0_f32];
    }
    let numerator_u = u32::try_from(numerator).unwrap_or(0_u32);
    let alpha_u = u32::try_from(alpha).unwrap_or(0_u32);
    let mut coeffs = vec![0.0_f32; usize::try_from(numerator_u).unwrap_or(0).saturating_add(1)];
    for coeff_idx in 0..=numerator_u {
        let binom = factorial(numerator_u.saturating_add(alpha_u)).div(
            factorial(numerator_u.saturating_sub(coeff_idx))
                .mul(factorial(alpha_u.saturating_add(coeff_idx))),
        );
        let mut term = binom.div(factorial(coeff_idx));
        if coeff_idx.rem_euclid(2) == 1 {
            term = term.mul(-1.0_f32);
        }
        if let Some(slot) = coeffs.get_mut(usize::try_from(coeff_idx).unwrap_or(0)) {
            *slot = term;
        }
    }
    coeffs
}
fn get_legendre_poly_part(l_param: u32, m_param: i32) -> Vec<f32> {
    let abs_m = m_param.unsigned_abs();
    let abs_m_i = i32::try_from(abs_m).unwrap_or(i32::MAX);
    let mut double_fact = 1.0_f32;
    for index in 1..=abs_m {
        let val = index.saturating_mul(2).saturating_sub(1);
        let val_f: f32 = u16::try_from(val).unwrap_or(u16::MAX).into();
        double_fact = double_fact.mul(val_f).mul(-1.0_f32);
    }
    let q_prev = vec![double_fact];
    if l_param == abs_m {
        return q_prev;
    }
    let mut q_curr = vec![0.0_f32; q_prev.len().saturating_add(1)];
    let init_factor_val = abs_m_i.saturating_mul(2_i32).saturating_add(1_i32);
    let factor: f32 = u16::try_from(init_factor_val).unwrap_or(u16::MAX).into();
    for (idx, &prev_coeff) in q_prev.iter().enumerate() {
        if let Some(slot) = q_curr.get_mut(idx.saturating_add(1)) {
            *slot = prev_coeff.mul(factor);
        }
    }
    if l_param == abs_m.saturating_add(1_u32) {
        return q_curr;
    }
    let mut q_prev2 = q_prev;
    let mut q_prev1 = q_curr;
    for poly_degree in (abs_m.saturating_add(2_u32))..=l_param {
        let poly_degree_i = i32::try_from(poly_degree).unwrap_or(i32::MAX);
        let val1 = poly_degree_i.saturating_mul(2_i32).saturating_sub(1_i32);
        let factor1: f32 = u16::try_from(val1).unwrap_or(u16::MAX).into();
        let val2 = poly_degree_i.saturating_add(abs_m_i).saturating_sub(1_i32);
        let factor2: f32 = u16::try_from(val2).unwrap_or(u16::MAX).into();
        let val3 = poly_degree_i.saturating_sub(abs_m_i);
        let divisor: f32 = u16::try_from(val3).unwrap_or(u16::MAX).into();
        let new_len = q_prev1.len().saturating_add(1);
        let mut q_new = vec![0.0_f32; new_len];
        for (idx, &prev1_coeff) in q_prev1.iter().enumerate() {
            if let Some(slot) = q_new.get_mut(idx.saturating_add(1)) {
                *slot = slot.add(prev1_coeff.mul(factor1));
            }
        }
        for (idx, &prev2_coeff) in q_prev2.iter().enumerate() {
            if let Some(slot) = q_new.get_mut(idx) {
                *slot = slot.sub(prev2_coeff.mul(factor2));
            }
        }
        for coeff in &mut q_new {
            *coeff = coeff.div(divisor);
        }
        q_prev2 = q_prev1;
        q_prev1 = q_new;
    }
    q_prev1
}
pub(super) fn generate(
    voxels: &mut [Voxel],
    dims: [u32; 3],
    voxel_size: f32,
    orbital: OrbitalConfig,
    material: MaterialConfig,
) -> Result<()> {
    let n_quantum = orbital.n_quantum;
    let l_quantum = orbital.l_quantum;
    let m_quantum = orbital.m_quantum;
    let charge = orbital.z_charge;
    let rad_coeffs = get_laguerre_coeffs(n_quantum, l_quantum);
    let ang_coeffs = get_legendre_poly_part(l_quantum, m_quantum);
    let n_f: f32 = u16::try_from(n_quantum).unwrap_or(u16::MAX).into();
    let l_f: f32 = u16::try_from(l_quantum).unwrap_or(u16::MAX).into();
    let abs_m = m_quantum.unsigned_abs();
    let term1 = 2.0_f32.mul(charge).div(n_f);
    let term1_pow3 = term1.powi(3);
    let fact1 = factorial(n_quantum.saturating_sub(l_quantum).saturating_sub(1));
    let fact2 = factorial(n_quantum.saturating_add(l_quantum));
    let denom = 2.0_f32.mul(n_f).mul(fact2);
    let n_rad_sq = term1_pow3.mul(fact1).div(denom);
    let term2 = 2.0_f32.mul(l_f).add(1.0_f32);
    let term3 = 4.0_f32.mul(PI);
    let fact3 = factorial(l_quantum.saturating_sub(abs_m));
    let fact4 = factorial(l_quantum.saturating_add(abs_m));
    let n_ang_sq = term2.div(term3).mul(fact3).div(fact4);
    let total_norm = n_rad_sq.mul(n_ang_sq);
    let scale = 2.0_f32.mul(charge).div(n_f);
    let material_params = MaterialParams {
        hue_positive: material.hue_positive,
        hue_negative: material.hue_negative,
        saturation: material.saturation,
        value: material.value,
        base_sigma_t: material.base_sigma_t,
        anisotropy: material.anisotropy,
        ior: material.ior,
    };
    let args = GpuArgs {
        dims,
        voxel_size,
        rad_coeffs: &rad_coeffs,
        ang_coeffs: &ang_coeffs,
        prefactor: total_norm,
        scale,
        l_quantum,
        m_quantum,
        material: material_params,
    };
    compute_voxels_gpu(voxels, args)
}
#[derive(Clone, Copy)]
struct GpuArgs<'src> {
    dims: [u32; 3],
    voxel_size: f32,
    rad_coeffs: &'src [f32],
    ang_coeffs: &'src [f32],
    prefactor: f32,
    scale: f32,
    l_quantum: u32,
    m_quantum: i32,
    material: MaterialParams,
}
struct GpuContext {
    ctx: alloc::sync::Arc<CudaContext>,
    density_module: alloc::sync::Arc<cudarc::driver::CudaModule>,
    density_kernel: cudarc::driver::CudaFunction,
    reduce_kernel: cudarc::driver::CudaFunction,
    finalize_kernel: cudarc::driver::CudaFunction,
}
impl GpuContext {
    fn new() -> Result<Self> {
        let ctx = CudaContext::new(0)?;
        let density_ptx = include_str!(concat!(env!("OUT_DIR"), "/density.ptx"));
        let postprocess_ptx = include_str!(concat!(env!("OUT_DIR"), "/postprocess.ptx"));
        let density_module = ctx.load_module(Ptx::from_src(density_ptx))?;
        let postprocess_module = ctx.load_module(Ptx::from_src(postprocess_ptx))?;
        let density_kernel = density_module.load_function("compute_density")?;
        let reduce_kernel = postprocess_module.load_function("reduce_max_abs")?;
        let finalize_kernel = postprocess_module.load_function("finalize_voxels")?;
        Ok(Self {
            ctx,
            density_module,
            density_kernel,
            reduce_kernel,
            finalize_kernel,
        })
    }
}

fn compute_voxels_gpu(voxels: &mut [Voxel], args: GpuArgs<'_>) -> Result<()> {
    let gpu = GpuContext::new()?;
    let mut rad_padded = [0.0_f32; MAX_DEGREE];
    for (idx, &val) in args.rad_coeffs.iter().enumerate() {
        if let Some(slot) = rad_padded.get_mut(idx) {
            *slot = val;
        }
    }
    let mut ang_padded = [0.0_f32; MAX_DEGREE];
    for (idx, &val) in args.ang_coeffs.iter().enumerate() {
        if let Some(slot) = ang_padded.get_mut(idx) {
            *slot = val;
        }
    }
    let dim_x = args.dims[0];
    let dim_y = args.dims[1];
    let dim_z = args.dims[2];
    let total_size = usize::try_from(dim_x)
        .unwrap_or(0)
        .saturating_mul(usize::try_from(dim_y).unwrap_or(0))
        .saturating_mul(usize::try_from(dim_z).unwrap_or(0));
    let total_size_i32 = i32::try_from(total_size).unwrap_or(i32::MAX);
    let stream = gpu.ctx.default_stream();

    let mut d_rad_coeffs_const = gpu.density_module.get_global("c_rad_coeffs", &stream)?;
    let mut d_ang_coeffs_const = gpu.density_module.get_global("c_ang_coeffs", &stream)?;

    let rad_bytes: &[u8] = bytemuck::cast_slice(&rad_padded);
    stream.memcpy_htod(rad_bytes, &mut d_rad_coeffs_const)?;

    let ang_bytes: &[u8] = bytemuck::cast_slice(&ang_padded);
    stream.memcpy_htod(ang_bytes, &mut d_ang_coeffs_const)?;

    let mut d_psi = stream.alloc_zeros::<f32>(total_size)?;
    launch_density_kernel(&gpu, &stream, &args, &mut d_psi)?;
    let max_abs_psi = compute_max_abs(&gpu, &stream, &d_psi, total_size_i32)?;
    let mut d_voxels = stream.alloc_zeros::<Voxel>(total_size)?;
    launch_finalize_kernel(
        &gpu,
        &stream,
        &d_psi,
        &mut d_voxels,
        max_abs_psi,
        &args.material,
        total_size_i32,
    )?;
    stream.synchronize()?;
    let result = stream.clone_dtoh(&d_voxels)?;
    voxels.copy_from_slice(&result);
    Ok(())
}

fn launch_density_kernel(
    gpu: &GpuContext,
    stream: &alloc::sync::Arc<cudarc::driver::CudaStream>,
    args: &GpuArgs<'_>,
    d_psi: &mut cudarc::driver::CudaSlice<f32>,
) -> Result<()> {
    let rad_deg = i32::try_from(args.rad_coeffs.len().saturating_sub(1)).unwrap_or(0_i32);
    let ang_deg = i32::try_from(args.ang_coeffs.len().saturating_sub(1)).unwrap_or(0_i32);
    let block_x = 16_u32;
    let block_y = 16_u32;
    let dim_x = args.dims[0];
    let dim_y = args.dims[1];
    let dim_z = args.dims[2];
    let grid_x = dim_x
        .saturating_add(block_x)
        .saturating_sub(1_u32)
        .div(block_x);
    let grid_y = dim_y
        .saturating_add(block_y)
        .saturating_sub(1_u32)
        .div(block_y);
    let density_config = LaunchConfig {
        grid_dim: (grid_x, grid_y, dim_z),
        block_dim: (block_x, block_y, 1),
        shared_mem_bytes: 0,
    };
    let l_i = i32::try_from(args.l_quantum).unwrap_or(0_i32);
    let dims_x = i32::try_from(dim_x).unwrap_or(0_i32);
    let dims_y = i32::try_from(dim_y).unwrap_or(0_i32);
    let dims_z = i32::try_from(dim_z).unwrap_or(0_i32);
    let mut builder = stream.launch_builder(&gpu.density_kernel);
    builder.arg(d_psi);
    builder.arg(&rad_deg);
    builder.arg(&ang_deg);
    builder.arg(&args.prefactor);
    builder.arg(&args.scale);
    builder.arg(&dims_x);
    builder.arg(&dims_y);
    builder.arg(&dims_z);
    builder.arg(&args.voxel_size);
    builder.arg(&l_i);
    builder.arg(&args.m_quantum);
    // SAFETY: Kernel parameters match the CUDA function signature
    unsafe {
        builder.launch(density_config)?;
    }
    Ok(())
}

fn compute_max_abs(
    gpu: &GpuContext,
    stream: &alloc::sync::Arc<cudarc::driver::CudaStream>,
    d_psi: &cudarc::driver::CudaSlice<f32>,
    total_size_i32: i32,
) -> Result<f32> {
    let total_size_u32 = u32::try_from(total_size_i32).unwrap_or(u32::MAX);
    let num_blocks_reduce = total_size_u32
        .saturating_add(BLOCK_SIZE.saturating_mul(2))
        .saturating_sub(1)
        .checked_div(BLOCK_SIZE.saturating_mul(2))
        .unwrap_or(1)
        .max(1);
    let num_blocks_usize = usize::try_from(num_blocks_reduce).unwrap_or(usize::MAX);
    let mut d_partial_max = stream.alloc_zeros::<f32>(num_blocks_usize)?;
    let reduce_config = LaunchConfig {
        grid_dim: (num_blocks_reduce, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut reduce_builder = stream.launch_builder(&gpu.reduce_kernel);
    reduce_builder.arg(d_psi);
    reduce_builder.arg(&mut d_partial_max);
    reduce_builder.arg(&total_size_i32);
    // SAFETY: Kernel parameters match the CUDA function signature
    unsafe {
        reduce_builder.launch(reduce_config)?;
    }
    if num_blocks_reduce > 1 {
        let mut d_final_max = stream.alloc_zeros::<f32>(1)?;
        let num_blocks_i32 = i32::try_from(num_blocks_reduce).unwrap_or(i32::MAX);
        let final_reduce_config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut final_builder = stream.launch_builder(&gpu.reduce_kernel);
        final_builder.arg(&d_partial_max);
        final_builder.arg(&mut d_final_max);
        final_builder.arg(&num_blocks_i32);
        // SAFETY: Kernel parameters match the CUDA function signature
        unsafe {
            final_builder.launch(final_reduce_config)?;
        }
        let result = stream.clone_dtoh(&d_final_max)?;
        Ok(result.first().copied().unwrap_or(1.0_f32))
    } else {
        let result = stream.clone_dtoh(&d_partial_max)?;
        Ok(result.first().copied().unwrap_or(1.0_f32))
    }
}

fn launch_finalize_kernel(
    gpu: &GpuContext,
    stream: &alloc::sync::Arc<cudarc::driver::CudaStream>,
    d_psi: &cudarc::driver::CudaSlice<f32>,
    d_voxels: &mut cudarc::driver::CudaSlice<Voxel>,
    max_abs_psi: f32,
    material: &MaterialParams,
    total_size_i32: i32,
) -> Result<()> {
    let total_size_u32 = u32::try_from(total_size_i32).unwrap_or(u32::MAX);
    let finalize_blocks = total_size_u32
        .saturating_add(BLOCK_SIZE)
        .saturating_sub(1)
        .div(BLOCK_SIZE);
    let finalize_config = LaunchConfig {
        grid_dim: (finalize_blocks, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = stream.launch_builder(&gpu.finalize_kernel);
    builder.arg(d_psi);
    builder.arg(d_voxels);
    builder.arg(&max_abs_psi);
    builder.arg(material);
    builder.arg(&total_size_i32);
    // SAFETY: Kernel parameters match the CUDA function signature
    unsafe {
        builder.launch(finalize_config)?;
    }
    Ok(())
}
