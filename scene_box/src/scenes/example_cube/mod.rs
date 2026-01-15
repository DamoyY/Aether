mod config;
mod generator;

use std::path::Path;

use anyhow::Result;
use config::ExampleCubeConfig;

use crate::{Camera, Light, SceneData, Voxel};

pub(crate) fn generate<P: AsRef<Path>>(config_path: P) -> Result<SceneData> {
    let config = ExampleCubeConfig::load(config_path)?;
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
        config.generator.center,
        config.generator.half_size,
        config.material,
        &config.gradient,
    );
    Ok(SceneData {
        voxels,
        dimensions: config.voxel.dimensions,
        voxel_size: config.voxel.voxel_size,
        camera: Camera {
            position: config.camera.position,
            target: config.camera.target,
            fov: config.camera.fov,
        },
        light: Light {
            position: config.light.position,
            color: config.light.color,
            intensity: config.light.intensity,
        },
        background: config.background,
    })
}
