//! Captured source identities and candidate edits shared by every host.
use runmat_analysis_language::LogicalSourcePath;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "tool-schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct FeaWorkspaceFile {
    pub path: LogicalSourcePath,
    pub version: String,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeaWorkspaceSnapshot {
    pub source_root: PathBuf,
    pub files: Vec<FeaWorkspaceFile>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "tool-schema", derive(schemars::JsonSchema))]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum FeaWorkspaceEdit {
    Write {
        path: LogicalSourcePath,
        bytes: Vec<u8>,
    },
    Remove {
        path: LogicalSourcePath,
    },
    Rename {
        from: LogicalSourcePath,
        to: LogicalSourcePath,
    },
}

pub const MAX_FEA_WORKSPACE_FILES: usize = 10_000;
pub const MAX_FEA_SOURCE_BYTES: usize = 32 * 1024 * 1024;
pub const MAX_FEA_GEOMETRY_BYTES: usize = 64 * 1024 * 1024;
pub const MAX_FEA_CAPTURE_BYTES: usize = MAX_FEA_SOURCE_BYTES + MAX_FEA_GEOMETRY_BYTES;
