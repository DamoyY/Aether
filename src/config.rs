use std::path::Path;

use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct Config {
    pub render: RenderConfig,
    pub camera: CameraConfig,
    pub light: LightConfig,
    pub voxel: VoxelConfig,
    pub material: MaterialConfig,
    pub scene: SceneConfig,
    pub window: WindowConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct RenderConfig {
    pub width: u32,
    pub height: u32,
    pub target_samples: u32,
    #[serde(default = "default_samples_per_frame")]
    pub samples_per_frame: u32,
}

const fn default_samples_per_frame() -> u32 {
    10
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct CameraConfig {
    pub position: [f32; 3],
    pub target: [f32; 3],
    pub up: [f32; 3],
    pub fov: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct LightConfig {
    pub position: [f32; 3],
    pub color: [f32; 3],
    pub intensity: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct VoxelConfig {
    pub dimensions: [u32; 3],
    pub voxel_size: f32,
    pub origin: [f32; 3],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct MaterialConfig {
    pub sigma_a: [f32; 3],
    pub sigma_s: [f32; 3],
    pub anisotropy: f32,
    pub ior: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct SceneConfig {
    pub background: [f32; 3],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct WindowConfig {
    pub title: String,
}

impl Config {
    pub(crate) fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = serde_yaml::from_str(&content)?;
        Ok(config)
    }
}
