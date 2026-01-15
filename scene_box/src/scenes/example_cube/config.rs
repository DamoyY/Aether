use std::path::Path;

use anyhow::Result;
use serde::Deserialize;

#[derive(Debug, Clone, Copy, Deserialize)]
pub(super) struct ExampleCubeConfig {
    pub camera: CameraConfig,
    pub light: LightConfig,
    pub voxel: VoxelConfig,
    pub material: MaterialConfig,
    pub gradient: GradientConfig,
    pub background: [f32; 3],
    pub generator: GeneratorConfig,
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub(super) struct CameraConfig {
    pub position: [f32; 3],
    pub target: [f32; 3],
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
    pub anisotropy: f32,
    pub ior: f32,
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub(super) struct GeneratorConfig {
    pub center: [f32; 3],
    pub half_size: f32,
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub(super) struct GradientMaterialConfig {
    pub albedo: [f32; 3],
    pub sigma_t: [f32; 3],
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub(super) struct GradientConfig {
    pub bottom: GradientMaterialConfig,
    pub top: GradientMaterialConfig,
}

impl ExampleCubeConfig {
    pub(super) fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = serde_yaml::from_str(&content)?;
        Ok(config)
    }
}
