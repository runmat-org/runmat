use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisConfig {
    #[serde(default)]
    pub artifact_store: Option<AnalysisArtifactStoreMode>,
    #[serde(default)]
    pub artifact_root: Option<PathBuf>,
    #[serde(default)]
    pub artifact_max_runs: Option<usize>,
    #[serde(default)]
    pub artifact_max_runs_per_kind: Option<usize>,
    #[serde(default)]
    pub study_artifact_root: Option<PathBuf>,
    #[serde(default)]
    pub geometry_prep_artifact_root: Option<PathBuf>,
    #[serde(default)]
    pub geometry_prep_max_artifacts: Option<usize>,
    #[serde(default)]
    pub geometry_prep_max_artifacts_per_geometry: Option<usize>,
    #[serde(default)]
    pub geometry_prep_max_age_seconds: Option<u64>,
    #[serde(default)]
    pub geometry_prep_require_latest_revision: Option<bool>,
    #[serde(default)]
    pub thermo_field_artifact_root: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum AnalysisArtifactStoreMode {
    #[default]
    InMemory,
    Filesystem,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            artifact_store: None,
            artifact_root: None,
            artifact_max_runs: None,
            artifact_max_runs_per_kind: None,
            study_artifact_root: None,
            geometry_prep_artifact_root: None,
            geometry_prep_max_artifacts: None,
            geometry_prep_max_artifacts_per_geometry: None,
            geometry_prep_max_age_seconds: None,
            geometry_prep_require_latest_revision: None,
            thermo_field_artifact_root: None,
        }
    }
}
