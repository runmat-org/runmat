use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeaConfig {
    #[serde(default)]
    pub artifact_store: Option<FeaArtifactStoreMode>,
    #[serde(default)]
    pub artifact_root: Option<PathBuf>,
    #[serde(default)]
    pub artifact_max_runs: Option<usize>,
    #[serde(default)]
    pub artifact_max_runs_per_kind: Option<usize>,
    #[serde(default)]
    pub study_artifact_root: Option<PathBuf>,
    #[serde(default)]
    pub thermo_field_artifact_root: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum FeaArtifactStoreMode {
    #[default]
    InMemory,
    Filesystem,
}
