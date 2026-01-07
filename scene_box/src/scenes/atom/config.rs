use std::path::Path;
use core::ops::Mul as _;

use anyhow::Result;
use serde::Deserialize;

#[derive(Debug, Clone, Copy, Deserialize)]
pub(super) struct AtomConfig {
    pub camera: CameraConfig,
    pub light: LightConfig,
    pub voxel: VoxelConfig,
    pub material: MaterialConfig,
    pub background: [f32; 3],
    pub orbital: OrbitalConfig,
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub(super) struct CameraConfig {
    pub height: f32,
    pub radius: f32,
    pub angle: f32,
    pub fov: f32,
}

impl CameraConfig {
    pub(super) fn position(&self) -> [f32; 3] {
        cylindrical_to_cartesian(self.height, self.radius, self.angle)
    }
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub(super) struct LightConfig {
    pub height: f32,
    pub radius: f32,
    pub angle: f32,
    pub color: [f32; 3],
    pub intensity: f32,
}

impl LightConfig {
    pub(super) fn position(&self) -> [f32; 3] {
        cylindrical_to_cartesian(self.height, self.radius, self.angle)
    }
}

fn cylindrical_to_cartesian(height: f32, radius: f32, angle_deg: f32) -> [f32; 3] {
    let angle_rad = angle_deg.to_radians();
    [
        radius.mul(angle_rad.cos()),
        height,
        radius.mul(angle_rad.sin()),
    ]
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
    pub hue_positive: f32,
    pub hue_negative: f32,
    pub saturation: f32,
    pub value: f32,
    pub base_sigma_t: f32,
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub(super) struct OrbitalConfig {
    pub n_quantum: u32,
    pub l_quantum: u32,
    pub m_quantum: i32,
    pub z_charge: f32,
}

impl AtomConfig {
    pub(super) fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = serde_yaml::from_str(&content)?;
        Ok(config)
    }
}
