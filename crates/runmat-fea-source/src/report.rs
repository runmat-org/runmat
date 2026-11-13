use runmat_analysis_language::{FeaDiagnostic, LogicalSourcePath};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeaCheckStatus {
    Valid,
    Invalid,
    Incomplete,
}

impl FeaCheckStatus {
    pub const fn is_valid(self) -> bool {
        matches!(self, Self::Valid)
    }
}

impl std::fmt::Display for FeaCheckStatus {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Valid => "valid",
            Self::Invalid => "invalid",
            Self::Incomplete => "incomplete",
        })
    }
}

/// Successfully inspected imported CAD available to read-only preview. This is
/// not an admitted analysis geometry definition or evidence of solve readiness.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct FeaGeometryPreview {
    pub module_path: LogicalSourcePath,
    pub geometry_id: String,
    pub source_path: LogicalSourcePath,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct FeaSourceCheckResult {
    pub contract: FeaSourceCheckContract,
    pub revision: runmat_analysis_language::SourceDigest,
    pub status: FeaCheckStatus,
    pub root_module: LogicalSourcePath,
    pub program: crate::FeaSemanticProgram,
    pub source_index: crate::FeaSourceIndex,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub geometry_previews: Vec<FeaGeometryPreview>,
    pub diagnostics: Vec<FeaDiagnostic>,
}

/// Checked-source compatibility is independent of optional empty semantic inventories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeaSourceCheckContract {
    AssessmentsV1,
}
