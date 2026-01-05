use core::f64::consts::PI;
use core::ops::{Add as _, Div as _, Mul as _, Sub as _};
use anyhow::Result;
use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg as _};
use cudarc::nvrtc::Ptx;
use super::config::{MaterialConfig, OrbitalConfig};
use crate::Voxel;
const MAX_DEGREE: usize = 32;
fn factorial(n: u32) -> f64 {
    let mut result = 1.0_f64;
    for i in 2..=n {
        result = result.mul(f64::from(i));
    }
    result
}
fn get_laguerre_coeffs(n_param: u32, l_param: u32) -> Vec<f32> {
    let n_val = i32::try_from(n_param).unwrap_or(i32::MAX);
    let l_val = i32::try_from(l_param).unwrap_or(i32::MAX);
    let numerator = n_val.sub(l_val).sub(1_i32);
    let alpha = l_val.mul(2_i32).add(1_i32);
    if numerator < 0_i32 {
        return vec![1.0_f32];
    }
    let numerator_u = u32::try_from(numerator).unwrap_or(0_u32);
    let alpha_u = u32::try_from(alpha).unwrap_or(0_u32);
    let mut coeffs = vec![0.0_f64; usize::try_from(numerator_u).unwrap_or(0).add(1)];
    for coeff_idx in 0..=numerator_u {
        let binom = factorial(numerator_u.add(alpha_u))
            .div(factorial(numerator_u.sub(coeff_idx)).mul(factorial(alpha_u.add(coeff_idx))));
        let mut term = binom.div(factorial(coeff_idx));
        if coeff_idx % 2 == 1 {
            term = term.mul(-1.0_f64);
        }
        if let Some(slot) = coeffs.get_mut(usize::try_from(coeff_idx).unwrap_or(0)) {
            *slot = term;
        }
    }
    coeffs.iter().map(|&val| val as f32).collect()
}
fn get_legendre_poly_part(l_param: u32, m_param: i32) -> Vec<f32> {
    let abs_m = m_param.unsigned_abs();
    let abs_m_i = i32::try_from(abs_m).unwrap_or(i32::MAX);
    let mut double_fact = 1.0_f64;
    for index in 1..=abs_m {
        double_fact = double_fact.mul(f64::from(index.mul(2).sub(1)).mul(-1.0_f64));
    }
    let q_prev = vec![double_fact];
    if l_param == abs_m {
        return q_prev.iter().map(|&val| val as f32).collect();
    }
    let mut q_curr = vec![0.0_f64; q_prev.len().add(1)];
    let factor = f64::from(abs_m_i.mul(2_i32).add(1_i32));
    for (idx, &val) in q_prev.iter().enumerate() {
        if let Some(slot) = q_curr.get_mut(idx.add(1)) {
            *slot = val.mul(factor);
        }
    }
    if l_param == abs_m.add(1_u32) {
        return q_curr.iter().map(|&val| val as f32).collect();
    }
    let mut q_prev2 = q_prev;
    let mut q_prev1 = q_curr;
    for poly_degree in (abs_m.add(2_u32))..=l_param {
        let poly_degree_i = i32::try_from(poly_degree).unwrap_or(i32::MAX);
        let factor1 = f64::from(poly_degree_i.mul(2_i32).sub(1_i32));
        let factor2 = f64::from(poly_degree_i.add(abs_m_i).sub(1_i32));
        let divisor = f64::from(poly_degree_i.sub(abs_m_i));
        let new_len = q_prev1.len().add(1);
        let mut q_new = vec![0.0_f64; new_len];
        for (idx, &val) in q_prev1.iter().enumerate() {
            if let Some(slot) = q_new.get_mut(idx.add(1)) {
                *slot = slot.add(val.mul(factor1));
            }
        }
        for (idx, &val) in q_prev2.iter().enumerate() {
            if let Some(slot) = q_new.get_mut(idx) {
                *slot = slot.sub(val.mul(factor2));
            }
        }
        for val in &mut q_new {
            *val = val.div(divisor);
        }
        q_prev2 = q_prev1;
        q_prev1 = q_new;
    }
    q_prev1.iter().map(|&val| val as f32).collect()
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
    let n_f = f64::from(n_quantum);
    let l_f = f64::from(l_quantum);
    let abs_m = m_quantum.unsigned_abs();
    let n_rad_sq = (2.0_f64.mul(f64::from(charge)).div(n_f))
        .powi(3_i32)
        .mul(factorial(n_quantum.sub(l_quantum).sub(1)))
        .div(2.0_f64.mul(n_f).mul(factorial(n_quantum.add(l_quantum))));
    let n_ang_sq = (2.0_f64.mul(l_f).add(1.0_f64))
        .div(4.0_f64.mul(PI))
        .mul(factorial(l_quantum.sub(abs_m)))
        .div(factorial(l_quantum.add(abs_m)));
    let total_norm = n_rad_sq.mul(n_ang_sq) as f32;
    let scale = 2.0_f32.mul(charge).div(n_quantum as f32);
    let density = compute_density_gpu(
        dims,
        voxel_size,
        &rad_coeffs,
        &ang_coeffs,
        total_norm,
        scale,
        l_quantum,
        abs_m,
    )?;
    let max_density = density.iter().copied().fold(0.0_f32, f32::max);
    let normalizer = if max_density > 0.0 {
        1.0_f32.div(max_density)
    } else {
        1.0_f32
    };
    for (idx, &density_val) in density.iter().enumerate() {
        let intensity = density_val.mul(normalizer);
        if let Some(slot) = voxels.get_mut(idx) {
            *slot = Voxel {
                intensity,
                sigma_a: material.sigma_a,
                sigma_s: material.sigma_s,
                anisotropy: material.anisotropy,
                ior: material.ior,
            };
        }
    }
    Ok(())
}
fn compute_density_gpu(
    dims: [u32; 3],
    voxel_size: f32,
    rad_coeffs: &[f32],
    ang_coeffs: &[f32],
    prefactor: f32,
    scale: f32,
    l_quantum: u32,
    abs_m: u32,
) -> Result<Vec<f32>> {
    let ctx = CudaContext::new(0)?;
    let ptx = include_str!(concat!(env!("OUT_DIR"), "/atom_density.ptx"));
    let module = ctx.load_module(Ptx::from_src(ptx))?;
    let kernel = module.load_function("compute_atom_density")?;
    let mut rad_padded = [0.0_f32; MAX_DEGREE];
    for (idx, &val) in rad_coeffs.iter().enumerate() {
        if let Some(slot) = rad_padded.get_mut(idx) {
            *slot = val;
        }
    }
    let rad_deg = i32::try_from(rad_coeffs.len().saturating_sub(1)).unwrap_or(0_i32);
    let mut ang_padded = [0.0_f32; MAX_DEGREE];
    for (idx, &val) in ang_coeffs.iter().enumerate() {
        if let Some(slot) = ang_padded.get_mut(idx) {
            *slot = val;
        }
    }
    let ang_deg = i32::try_from(ang_coeffs.len().saturating_sub(1)).unwrap_or(0_i32);
    let dim_x = dims[0];
    let dim_y = dims[1];
    let dim_z = dims[2];
    let total_size = usize::try_from(dim_x)
        .unwrap_or(0)
        .saturating_mul(usize::try_from(dim_y).unwrap_or(0))
        .saturating_mul(usize::try_from(dim_z).unwrap_or(0));
    let stream = ctx.default_stream();
    let d_rad_coeffs = stream.clone_htod(&rad_padded)?;
    let d_ang_coeffs = stream.clone_htod(&ang_padded)?;
    let mut d_output = stream.alloc_zeros::<f32>(total_size)?;
    let block_x = 16_u32;
    let block_y = 16_u32;
    let grid_x = (dim_x).add(block_x).sub(1_u32).div(block_x);
    let grid_y = (dim_y).add(block_y).sub(1_u32).div(block_y);
    let grid_z = dim_z;
    let config = LaunchConfig {
        grid_dim: (grid_x, grid_y, grid_z),
        block_dim: (block_x, block_y, 1),
        shared_mem_bytes: 0,
    };
    let l_i = i32::try_from(l_quantum).unwrap_or(0_i32);
    let abs_m_i = i32::try_from(abs_m).unwrap_or(0_i32);
    let dims_x = i32::try_from(dim_x).unwrap_or(0_i32);
    let dims_y = i32::try_from(dim_y).unwrap_or(0_i32);
    let dims_z = i32::try_from(dim_z).unwrap_or(0_i32);
    let mut builder = stream.launch_builder(&kernel);
    builder.arg(&mut d_output);
    builder.arg(&d_rad_coeffs);
    builder.arg(&d_ang_coeffs);
    builder.arg(&rad_deg);
    builder.arg(&ang_deg);
    builder.arg(&prefactor);
    builder.arg(&scale);
    builder.arg(&dims_x);
    builder.arg(&dims_y);
    builder.arg(&dims_z);
    builder.arg(&voxel_size);
    builder.arg(&l_i);
    builder.arg(&abs_m_i);
    // SAFETY: Kernel parameters match the CUDA function signature
    unsafe {
        builder.launch(config)?;
    }
    stream.synchronize()?;
    let result = stream.clone_dtoh(&d_output)?;
    Ok(result)
}
