mod config;
mod generator;
use crate::{Camera, Light, SceneData, Voxel};
use anyhow::Result;
use config::AtomConfig;
use core::ops::Mul as _;
use std::path::Path;
pub(crate) fn generate<P: AsRef<Path>>(config_path: P) -> Result<SceneData> {
    let config = AtomConfig::load(config_path)?;
    let dim0 = usize::try_from(config.voxel.dimensions[0]).unwrap_or(usize::MAX);
    let dim1 = usize::try_from(config.voxel.dimensions[1]).unwrap_or(usize::MAX);
    let dim2 = usize::try_from(config.voxel.dimensions[2]).unwrap_or(usize::MAX);
    let size = dim0.saturating_mul(dim1).saturating_mul(dim2);
    let mut voxels = vec![
        Voxel {
            intensity: 0.0,
            albedo: [0.0; 3],
            sigma_t: [0.0; 3],
            anisotropy: 0.0,
            ior: 1.0,
        };
        size
    ];
    generator::generate(
        &mut voxels,
        config.voxel.dimensions,
        config.voxel.voxel_size,
        config.orbital,
        config.material,
    )?;
    let dim_x: f32 = u16::try_from(config.voxel.dimensions[0])
        .unwrap_or(u16::MAX)
        .into();
    let dim_y: f32 = u16::try_from(config.voxel.dimensions[1])
        .unwrap_or(u16::MAX)
        .into();
    let dim_z: f32 = u16::try_from(config.voxel.dimensions[2])
        .unwrap_or(u16::MAX)
        .into();
    let center = [
        dim_x.mul(config.voxel.voxel_size).mul(0.5),
        dim_y.mul(config.voxel.voxel_size).mul(0.5),
        dim_z.mul(config.voxel.voxel_size).mul(0.5),
    ];
    Ok(SceneData {
        voxels,
        dimensions: config.voxel.dimensions,
        voxel_size: config.voxel.voxel_size,
        camera: Camera {
            position: config.camera.position(),
            target: center,
            fov: config.camera.fov,
        },
        light: Light {
            position: config.light.position(),
            color: config.light.color,
            intensity: config.light.intensity,
        },
        background: config.background,
    })
}
