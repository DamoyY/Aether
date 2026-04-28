use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct Config {
    pub render: RenderConfig,
    pub window: WindowConfig,
    pub scene_path: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct RenderConfig {
    pub width: u32,
    pub height: u32,
    pub target_samples: u32,
    pub samples_per_frame: u32,
    pub output: Option<String>,
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
