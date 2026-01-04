use std::path::Path;

use anyhow::Result;
use serde::Deserialize;

#[derive(Debug, Clone, Copy, Deserialize)]
pub(super) struct CubeConfig {
    pub camera: CameraConfig,
    pub light: LightConfig,
    pub voxel: VoxelConfig,
    pub material: MaterialConfig,
    pub background: [f32; 3],
    pub generator: GeneratorConfig,
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub(super) struct CameraConfig {
    pub position: [f32; 3],
    pub target: [f32; 3],
    pub up: [f32; 3],
    pub fov: f32,
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub(super) struct LightConfig {
    pub position: [f32; 3],
    pub color: [f32; 3],
    pub intensity: f32,
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub(super) struct VoxelConfig {
    pub dimensions: [u32; 3],
    pub voxel_size: f32,
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub(super) struct MaterialConfig {
    pub sigma_a: [f32; 3],
    pub sigma_s: [f32; 3],
    pub anisotropy: f32,
    pub ior: f32,
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub(super) struct GeneratorConfig {
    pub center: [f32; 3],
    pub half_size: f32,
}

impl CubeConfig {
    pub(super) fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = serde_yaml::from_str(&content)?;
        Ok(config)
    }
}
