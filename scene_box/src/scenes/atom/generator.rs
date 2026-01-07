use core::{
    f32::consts::PI,
    ops::{Add as _, Div as _, Mul as _, Sub as _},
};

use anyhow::Result;
use cudarc::{
    driver::{CudaContext, LaunchConfig, PushKernelArg as _},
    nvrtc::Ptx,
};

use super::config::{MaterialConfig, OrbitalConfig};
use crate::Voxel;
const MAX_DEGREE: usize = 32;

fn hsv_to_rgb(hue_input: f32, saturation: f32, value: f32) -> [f32; 3] {
    let hue = hue_input.rem_euclid(360.0);
    let chroma = value.mul(saturation);
    let x_val = chroma.mul(1.0_f32.sub((hue.div(60.0).rem_euclid(2.0).sub(1.0)).abs()));
    let m_val = value.sub(chroma);
    let (red, green, blue) = if hue < 60.0 {
        (chroma, x_val, 0.0_f32)
    } else if hue < 120.0 {
        (x_val, chroma, 0.0_f32)
    } else if hue < 180.0 {
        (0.0_f32, chroma, x_val)
    } else if hue < 240.0 {
        (0.0_f32, x_val, chroma)
    } else if hue < 300.0 {
        (x_val, 0.0_f32, chroma)
    } else {
        (chroma, 0.0_f32, x_val)
    };
    [red.add(m_val), green.add(m_val), blue.add(m_val)]
}

fn lerp_hue(factor: f32, hue_neg: f32, hue_pos: f32) -> f32 {
    let t_normalized = factor.mul(0.5_f32).add(0.5_f32);
    let diff = hue_pos.sub(hue_neg);
    let short_diff = if diff.abs() <= 180.0 {
        diff
    } else if diff > 0.0 {
        diff.sub(360.0)
    } else {
        diff.add(360.0)
    };
    hue_neg.add(short_diff.mul(t_normalized)).rem_euclid(360.0)
}

fn factorial(n: u32) -> f32 {
    let mut result = 1.0_f32;
    for i in 2..=n {
        // Safe cast: n is small for factorial to fit in f32
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
    let mut coeffs = vec![
        0.0_f32;
        usize::try_from(numerator_u)
            .unwrap_or(0)
            .saturating_add(1)
    ];
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
    
    // 2.0 * charge / n_f
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
    
    let args = DensityArgs {
        dims,
        voxel_size,
        rad_coeffs: &rad_coeffs,
        ang_coeffs: &ang_coeffs,
        prefactor: total_norm,
        scale,
        l_quantum,
        m_quantum,
    };
    
    let psi_values = compute_density_gpu(args)?;
    let max_abs_psi = psi_values
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0_f32, f32::max);
    let normalizer = if max_abs_psi > 0.0 {
        1.0_f32.div(max_abs_psi)
    } else {
        1.0_f32
    };
    for (idx, &psi_val) in psi_values.iter().enumerate() {
        let normalized_psi = psi_val.mul(normalizer);
        let intensity = normalized_psi.abs();
        let hue = lerp_hue(
            normalized_psi,
            material.hue_negative,
            material.hue_positive,
        );
        let albedo = hsv_to_rgb(hue, material.saturation, material.value);
        let sigma_t_val = material.base_sigma_t.mul(intensity);
        if let Some(slot) = voxels.get_mut(idx) {
            *slot = Voxel {
                intensity,
                albedo,
                sigma_t: [sigma_t_val, sigma_t_val, sigma_t_val],
                anisotropy: material.anisotropy,
                ior: material.ior,
            };
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct DensityArgs<'src> {
    dims: [u32; 3],
    voxel_size: f32,
    rad_coeffs: &'src [f32],
    ang_coeffs: &'src [f32],
    prefactor: f32,
    scale: f32,
    l_quantum: u32,
    m_quantum: i32,
}

fn compute_density_gpu(args: DensityArgs<'_>) -> Result<Vec<f32>> {
    let ctx = CudaContext::new(0)?;
    let ptx = include_str!(concat!(env!("OUT_DIR"), "/atom_density.ptx"));
    let module = ctx.load_module(Ptx::from_src(ptx))?;
    let kernel = module.load_function("compute_atom_density")?;
    let mut rad_padded = [0.0_f32; MAX_DEGREE];
    for (idx, &val) in args.rad_coeffs.iter().enumerate() {
        if let Some(slot) = rad_padded.get_mut(idx) {
            *slot = val;
        }
    }
    let rad_deg = i32::try_from(args.rad_coeffs.len().saturating_sub(1)).unwrap_or(0_i32);
    let mut ang_padded = [0.0_f32; MAX_DEGREE];
    for (idx, &val) in args.ang_coeffs.iter().enumerate() {
        if let Some(slot) = ang_padded.get_mut(idx) {
            *slot = val;
        }
    }
    let ang_deg = i32::try_from(args.ang_coeffs.len().saturating_sub(1)).unwrap_or(0_i32);
    let dim_x = args.dims[0];
    let dim_y = args.dims[1];
    let dim_z = args.dims[2];
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
    let grid_x = (dim_x).saturating_add(block_x).saturating_sub(1_u32).div(block_x);
    let grid_y = (dim_y).saturating_add(block_y).saturating_sub(1_u32).div(block_y);
    let grid_z = dim_z;
    let config = LaunchConfig {
        grid_dim: (grid_x, grid_y, grid_z),
        block_dim: (block_x, block_y, 1),
        shared_mem_bytes: 0,
    };
    let l_i = i32::try_from(args.l_quantum).unwrap_or(0_i32);
    let dims_x = i32::try_from(dim_x).unwrap_or(0_i32);
    let dims_y = i32::try_from(dim_y).unwrap_or(0_i32);
    let dims_z = i32::try_from(dim_z).unwrap_or(0_i32);
    let mut builder = stream.launch_builder(&kernel);
    builder.arg(&mut d_output);
    builder.arg(&d_rad_coeffs);
    builder.arg(&d_ang_coeffs);
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
        builder.launch(config)?;
    }
    stream.synchronize()?;
    let result = stream.clone_dtoh(&d_output)?;
    Ok(result)
}