mod config;
mod scenes {
    #[expect(clippy::mod_module_files, reason = "To ensure a clear document structure")]
    pub(crate) mod atom;
    #[expect(clippy::mod_module_files, reason = "To ensure a clear document structure")]
    pub(crate) mod cube;
}

use anyhow::{bail, Result};
use bytemuck::{Pod, Zeroable};
use config::SceneSelector;
use cudarc::driver::DeviceRepr;
use std::path::Path;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Voxel {
    pub intensity: f32,
    pub albedo: [f32; 3],
    pub sigma_t: [f32; 3],
    pub anisotropy: f32,
    pub ior: f32,
}

// SAFETY: Voxel is #[repr(C)] and contains only f32 which is valid for GPU transfer.
unsafe impl DeviceRepr for Voxel {}

#[derive(Debug, Clone, Copy)]
pub struct Camera {
    pub position: [f32; 3],
    pub target: [f32; 3],
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
    pub albedo: [f32; 3],
    pub sigma_t: [f32; 3],
    pub anisotropy: f32,
    pub ior: f32,
}

#[derive(Debug)]
pub struct SceneData {
    pub voxels: Vec<Voxel>,
    pub dimensions: [u32; 3],
    pub voxel_size: f32,
    pub camera: Camera,
    pub light: Light,
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
    match scene_name.as_str() {
        "atom" => scenes::atom::generate(&scene_path),
        "cube" => scenes::cube::generate(&scene_path),
        _ => bail!("Unknown scene type: {scene_name}"),
    }
}
