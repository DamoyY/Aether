mod config;
mod generator;

use std::path::Path;

use anyhow::Result;
use config::CubeConfig;

use crate::{Camera, Light, Material, SceneData, Voxel};

pub(crate) fn generate<P: AsRef<Path>>(config_path: P) -> Result<SceneData> {
    let config = CubeConfig::load(config_path)?;

    let dim0 = usize::try_from(config.voxel.dimensions[0]).unwrap_or(usize::MAX);
    let dim1 = usize::try_from(config.voxel.dimensions[1]).unwrap_or(usize::MAX);
    let dim2 = usize::try_from(config.voxel.dimensions[2]).unwrap_or(usize::MAX);
    let size = dim0.saturating_mul(dim1).saturating_mul(dim2);
    let mut voxels = vec![Voxel { intensity: 0.0 }; size];

    generator::generate(
        &mut voxels,
        config.voxel.dimensions,
        config.voxel.voxel_size,
        config.generator.center,
        config.generator.half_size,
    );

    Ok(SceneData {
        voxels,
        dimensions: config.voxel.dimensions,
        voxel_size: config.voxel.voxel_size,
        camera: Camera {
            position: config.camera.position,
            target: config.camera.target,
            up: config.camera.up,
            fov: config.camera.fov,
        },
        light: Light {
            position: config.light.position,
            color: config.light.color,
            intensity: config.light.intensity,
        },
        material: Material {
            sigma_a: config.material.sigma_a,
            sigma_s: config.material.sigma_s,
            anisotropy: config.material.anisotropy,
            ior: config.material.ior,
        },
        background: config.background,
    })
}
