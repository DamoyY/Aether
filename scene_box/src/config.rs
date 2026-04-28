use anyhow::Result;
use serde::Deserialize;
use std::path::Path;
#[derive(Debug, Deserialize)]
pub(crate) struct SceneSelector {
    pub scene_name: String,
}
impl SceneSelector {
    pub(crate) fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let selector: Self = serde_yaml::from_str(&content)?;
        Ok(selector)
    }
}
