mod admission;
mod compile;
mod geometry_resolver;
mod host;
mod project_host;
mod query;
mod report;
mod root;
mod semantic_program;
mod snapshot;
mod source_index;
mod workspace;
pub use query::{
    FeaBrowseOutcome, FeaBrowsePage, FeaBrowseRequest, FeaQueryCursor, FeaQueryDetail,
    FeaQueryError, FeaQueryFilter,
};
pub use workspace::{
    FeaWorkspaceCandidate, FeaWorkspaceEdit, FeaWorkspaceError, FeaWorkspaceFile,
    FeaWorkspaceSnapshot,
};

pub use compile::{
    check_fea_path, check_fea_path_in_root, check_fea_source_in_root, check_fea_sources_in_root,
    prepare_fea_path_in_root, prepare_fea_sources_in_root, CheckedFeaSource, PreparedFeaSource,
};
pub use host::{logical_source_path, physical_source_path, source_digest, FilesystemFeaHost};
pub use project_host::ProjectFeaHost;
pub use report::{FeaCheckStatus, FeaRootSelector, FeaSourceCheckResult};
pub use root::{FeaRootSelectionError, ResolvedFeaRoot};
pub use semantic_program::{
    FeaDeclaredMaterial, FeaExecutionBinding, FeaExecutionTarget, FeaSemanticProgram,
};
pub use snapshot::{
    FeaSourceSnapshotBuilder, FeaSourceSnapshotRequests, PreparedFeaSourceSnapshot,
};
pub use source_index::{
    FeaModelMemberKind, FeaSourceIdentity, FeaSourceIndex, FeaSourceIndexEntry, FeaStudyMemberKind,
};

#[cfg(test)]
mod tests;
