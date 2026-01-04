use core::ops::{Add as _, Div as _, Mul as _, Sub as _};

use bytemuck::{Pod, Zeroable};
use cudarc::driver::DeviceRepr;
use glam::Vec3;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Voxel {
    pub intensity: f32,
}
// SAFETY: Voxel is #[repr(C)] and contains only f32 which is valid for GPU transfer.
unsafe impl DeviceRepr for Voxel {}

#[derive(Clone, Copy, Debug)]
pub struct GridConfig {
    pub dimensions: [u32; 3],
    pub voxel_size: f32,
    pub origin: Vec3,
}

#[derive(Debug)]
pub struct Grid {
    pub config: GridConfig,
    pub data: Vec<Voxel>,
}

impl Default for Grid {
    #[inline]
    fn default() -> Self {
        Self {
            config: GridConfig {
                dimensions: [0; 3],
                voxel_size: 0.0,
                origin: Vec3::ZERO,
            },
            data: Vec::new(),
        }
    }
}

impl Grid {
    #[inline]
    pub fn init(&mut self, config: GridConfig) {
        let dim0 = usize::try_from(config.dimensions[0]).unwrap_or(usize::MAX);
        let dim1 = usize::try_from(config.dimensions[1]).unwrap_or(usize::MAX);
        let dim2 = usize::try_from(config.dimensions[2]).unwrap_or(usize::MAX);
        let size = dim0.saturating_mul(dim1).saturating_mul(dim2);
        self.config = config;
        self.data = vec![Voxel { intensity: 0.0 }; size];
    }

    #[inline]
    pub fn set(&mut self, coord_x: u32, coord_y: u32, coord_z: u32, voxel: Voxel) {
        let dims = &self.config.dimensions;
        let raw_idx = coord_z
            .saturating_mul(dims[1])
            .saturating_mul(dims[0])
            .saturating_add(coord_y.saturating_mul(dims[0]))
            .saturating_add(coord_x);
        let idx = usize::try_from(raw_idx).unwrap_or(usize::MAX);
        if let Some(slot) = self.data.get_mut(idx) {
            *slot = voxel;
        }
    }
}

#[inline]
pub fn fill_cube_voxels(grid: &mut Grid, center: Vec3, half_size: f32) {
    let dims = grid.config.dimensions;
    let origin = grid.config.origin;
    let voxel_size = grid.config.voxel_size;

    for coord_z in 0..dims[2] {
        for coord_y in 0..dims[1] {
            for coord_x in 0..dims[0] {
                let cx = u16::try_from(coord_x).unwrap_or(u16::MAX);
                let cy = u16::try_from(coord_y).unwrap_or(u16::MAX);
                let cz = u16::try_from(coord_z).unwrap_or(u16::MAX);
                let offset_x = (f32::from(cx).add(0.5)).mul(voxel_size);
                let offset_y = (f32::from(cy).add(0.5)).mul(voxel_size);
                let offset_z = (f32::from(cz).add(0.5)).mul(voxel_size);
                let world_pos = origin.add(Vec3::new(offset_x, offset_y, offset_z));

                let rel = world_pos.sub(center);
                if rel.x.abs() <= half_size && rel.y.abs() <= half_size && rel.z.abs() <= half_size
                {
                    let dist = rel.abs().max_element().div(half_size);
                    let intensity = 1.0_f32.sub(dist.mul(0.3));
                    grid.set(coord_x, coord_y, coord_z, Voxel { intensity });
                }
            }
        }
    }
}
