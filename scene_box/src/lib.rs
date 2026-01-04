mod config;

#[path = "scenes/cube/cube.rs"]
mod cube;

use anyhow::{bail, Result};
use bytemuck::{Pod, Zeroable};
use config::{SceneConfig, SceneSelector};
use cudarc::driver::DeviceRepr;
use std::path::Path;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Voxel {
    pub intensity: f32,
}

// SAFETY: Voxel is #[repr(C)] and contains only f32 which is valid for GPU transfer.
unsafe impl DeviceRepr for Voxel {}

#[derive(Debug, Clone, Copy)]
pub struct Camera {
    pub position: [f32; 3],
    pub target: [f32; 3],
    pub up: [f32; 3],
    pub fov: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct Light {
    pub position: [f32; 3],
    pub color: [f32; 3],
    pub intensity: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct Material {
    pub sigma_a: [f32; 3],
    pub sigma_s: [f32; 3],
    pub anisotropy: f32,
    pub ior: f32,
}

#[derive(Debug)]
pub struct SceneData {
    pub voxels: Vec<Voxel>,
    pub dimensions: [u32; 3],
    pub voxel_size: f32,
    pub origin: [f32; 3],
    pub camera: Camera,
    pub light: Light,
    pub material: Material,
    pub background: [f32; 3],
}
#[inline]
pub fn generate<P: AsRef<Path>>(path: P) -> Result<SceneData> {
    let base_path = path.as_ref().parent().unwrap_or_else(|| Path::new("."));
    let selector = SceneSelector::load(&path)?;
    let scene_name = &selector.scene_name;

    let scene_path = base_path
        .join("src")
        .join("scenes")
        .join(scene_name)
        .join(format!("{scene_name}.yaml"));
    let config = SceneConfig::load(&scene_path)?;

    let dim0 = usize::try_from(config.voxel.dimensions[0]).unwrap_or(usize::MAX);
    let dim1 = usize::try_from(config.voxel.dimensions[1]).unwrap_or(usize::MAX);
    let dim2 = usize::try_from(config.voxel.dimensions[2]).unwrap_or(usize::MAX);
    let size = dim0.saturating_mul(dim1).saturating_mul(dim2);
    let mut voxels = vec![Voxel { intensity: 0.0 }; size];

    match scene_name.as_str() {
        "cube" => {
            cube::generate(
                &mut voxels,
                config.voxel.dimensions,
                config.voxel.voxel_size,
                config.voxel.origin,
                config.generator.center,
                config.generator.half_size,
            );
        }
        _ => bail!("Unknown scene type: {scene_name}"),
    }

    Ok(SceneData {
        voxels,
        dimensions: config.voxel.dimensions,
        voxel_size: config.voxel.voxel_size,
        origin: config.voxel.origin,
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
