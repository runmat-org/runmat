//! Text authoring changes and checked publication receipts shared by hosts.
use crate::{FeaSourceCheckResult, FeaWorkspaceEdit};
use runmat_analysis_language::{LogicalSourcePath, SourceDigest};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "tool-schema", derive(schemars::JsonSchema))]
pub struct FeaChangeSetId(pub [u8; 16]);

/// Text stays text at the model boundary; conversion to compiler bytes occurs once.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "tool-schema", derive(schemars::JsonSchema))]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum FeaSourceChange {
    Write {
        /// Workspace-relative source file, e.g. `bracket.fea` or `checks/verify.m`.
        /// The host binds the workspace; absolute paths and traversal are rejected.
        #[serde(deserialize_with = "deserialize_source_file")]
        path: LogicalSourcePath,
        source: String,
    },
    Remove {
        /// Workspace-relative source file.
        #[serde(deserialize_with = "deserialize_source_file")]
        path: LogicalSourcePath,
    },
    Rename {
        /// Existing workspace-relative source file.
        #[serde(deserialize_with = "deserialize_source_file")]
        from: LogicalSourcePath,
        /// New workspace-relative source file.
        #[serde(deserialize_with = "deserialize_source_file")]
        to: LogicalSourcePath,
    },
}
impl From<FeaSourceChange> for FeaWorkspaceEdit {
    fn from(change: FeaSourceChange) -> Self {
        match change {
            FeaSourceChange::Write { path, source } => Self::Write {
                path,
                bytes: source.into_bytes(),
            },
            FeaSourceChange::Remove { path } => Self::Remove { path },
            FeaSourceChange::Rename { from, to } => Self::Rename { from, to },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "tool-schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct FeaChangeSetRequest {
    /// Complete replacement proposal when repairing a prior stage. Its base is
    /// retained; rebase onto collaborator edits must be an explicit new proposal.
    pub base: Option<FeaChangeSetId>,
    pub changes: Vec<FeaSourceChange>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeaChangeSetCheck {
    pub id: FeaChangeSetId,
    pub roots: Vec<FeaSourceCheckResult>,
    pub publication_valid: bool,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum FeaChangeSetPublication {
    /// The original commit receipt exists; durability and visibility remain unresolved.
    DurabilityIndeterminate {
        id: FeaChangeSetId,
        revision: SourceDigest,
    },
    /// Source edits are committed durably; provider recovery must finish before reads.
    CommittedPendingRecovery {
        id: FeaChangeSetId,
        revision: SourceDigest,
    },
    Published {
        id: FeaChangeSetId,
        revision: SourceDigest,
    },
    Conflict {
        current: SourceDigest,
    },
    Invalid {
        roots: Vec<FeaSourceCheckResult>,
    },
    IdReused,
}

/// Bounded source review. Offsets address UTF-8 bytes in the immutable staged
/// source, never a live file. The next offset always falls on a character boundary.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "tool-schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct FeaChangeSetReviewRequest {
    pub id: FeaChangeSetId,
    pub change_index: u32,
    pub before_offset: u32,
    pub after_offset: u32,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum FeaSourceChangeSummary {
    Write {
        path: LogicalSourcePath,
    },
    Remove {
        path: LogicalSourcePath,
    },
    Rename {
        from: LogicalSourcePath,
        to: LogicalSourcePath,
    },
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeaSourceExcerpt {
    pub offset: u32,
    pub total_bytes: u32,
    pub text: String,
    pub next_offset: Option<u32>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeaChangeSetReview {
    pub id: FeaChangeSetId,
    pub base_revision: SourceDigest,
    pub change_index: u32,
    pub change_count: u32,
    pub change: FeaSourceChangeSummary,
    pub before: Option<FeaSourceExcerpt>,
    pub after: Option<FeaSourceExcerpt>,
}

fn deserialize_source_file<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<LogicalSourcePath, D::Error> {
    LogicalSourcePath::file(String::deserialize(deserializer)?).map_err(serde::de::Error::custom)
}
