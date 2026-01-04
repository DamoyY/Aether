use std::path::Path;

use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Scene {
    pub camera: Camera,
    pub light: Light,
    pub voxel: VoxelSettings,
    pub material: Material,
    pub background: [f32; 3],
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Camera {
    pub position: [f32; 3],
    pub target: [f32; 3],
    pub up: [f32; 3],
    pub fov: f32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Light {
    pub position: [f32; 3],
    pub color: [f32; 3],
    pub intensity: f32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct VoxelSettings {
    pub dimensions: [u32; 3],
    pub voxel_size: f32,
    pub origin: [f32; 3],
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Material {
    pub sigma_a: [f32; 3],
    pub sigma_s: [f32; 3],
    pub anisotropy: f32,
    pub ior: f32,
}

impl Scene {
    #[inline]
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = serde_yaml::from_str(&content)?;
        Ok(config)
    }
}
