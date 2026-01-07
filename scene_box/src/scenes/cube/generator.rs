use core::ops::{Add as _, Div as _, Mul as _, Sub as _};

use super::config::{GradientConfig, MaterialConfig};
use crate::Voxel;

fn index(coord_x: u32, coord_y: u32, coord_z: u32, dims: [u32; 3]) -> usize {
    let raw_idx = coord_z
        .saturating_mul(dims[1])
        .saturating_mul(dims[0])
        .saturating_add(coord_y.saturating_mul(dims[0]))
        .saturating_add(coord_x);
    usize::try_from(raw_idx).unwrap_or(usize::MAX)
}

fn lerp(from: f32, to: f32, ratio: f32) -> f32 {
    from.add(to.sub(from).mul(ratio))
}

fn lerp3(from: [f32; 3], to: [f32; 3], ratio: f32) -> [f32; 3] {
    [
        lerp(from[0], to[0], ratio),
        lerp(from[1], to[1], ratio),
        lerp(from[2], to[2], ratio),
    ]
}

pub(super) fn generate(
    voxels: &mut [Voxel],
    dims: [u32; 3],
    voxel_size: f32,
    center: [f32; 3],
    half_size: f32,
    material: MaterialConfig,
    gradient: &GradientConfig,
) {
    let y_min = center[1].sub(half_size);
    let y_range = half_size.mul(2.0);

    for coord_z in 0..dims[2] {
        for coord_y in 0..dims[1] {
            for coord_x in 0..dims[0] {
                let cx = u16::try_from(coord_x).unwrap_or(u16::MAX);
                let cy = u16::try_from(coord_y).unwrap_or(u16::MAX);
                let cz = u16::try_from(coord_z).unwrap_or(u16::MAX);
                let world_x = (f32::from(cx).add(0.5)).mul(voxel_size);
                let world_y = (f32::from(cy).add(0.5)).mul(voxel_size);
                let world_z = (f32::from(cz).add(0.5)).mul(voxel_size);
                let rel_x = world_x.sub(center[0]);
                let rel_y = world_y.sub(center[1]);
                let rel_z = world_z.sub(center[2]);
                if rel_x.abs() <= half_size && rel_y.abs() <= half_size && rel_z.abs() <= half_size
                {
                    let dist_x = rel_x.abs();
                    let dist_y = rel_y.abs();
                    let dist_z = rel_z.abs();
                    let max_dist = dist_x.max(dist_y).max(dist_z);
                    let dist = max_dist.div(half_size);
                    let intensity = 1.0_f32.sub(dist.mul(0.3));

                    let y_ratio = world_y.sub(y_min).div(y_range).clamp(0.0, 1.0);
                    let albedo = lerp3(gradient.bottom.albedo, gradient.top.albedo, y_ratio);
                    let sigma_t = lerp3(gradient.bottom.sigma_t, gradient.top.sigma_t, y_ratio);

                    let idx = index(coord_x, coord_y, coord_z, dims);
                    if let Some(slot) = voxels.get_mut(idx) {
                        *slot = Voxel {
                            intensity,
                            albedo,
                            sigma_t,
                            anisotropy: material.anisotropy,
                            ior: material.ior,
                        };
                    }
                }
            }
        }
    }
}
