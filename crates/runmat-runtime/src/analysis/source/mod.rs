mod compile;
mod geometry_resolver;
mod host;
mod presentation;
mod project_host;
mod report;
mod root;
mod snapshot;

pub use compile::{
    check_fea_path, check_fea_path_in_root, check_fea_source_in_root, check_fea_sources_in_root,
    prepare_fea_path_in_root, prepare_fea_sources_in_root, CheckedFeaSource, PreparedFeaSource,
};
pub use host::{logical_source_path, physical_source_path, source_digest, FilesystemFeaHost};
pub use presentation::{
    FeaBoundaryPresentationKind, FeaConnectorPresentationKind, FeaConstraintPresentationKind,
    FeaInitialConditionPresentationKind, FeaInterfacePresentationKind, FeaOutputPresentation,
    FeaPresentationTarget, FeaProcedurePresentation, FeaRandomExcitationPresentation,
    FeaResponseSpectrumPresentation, FeaSignalPresentation, FeaSignalRepresentation,
    FeaStudyInputPresentation, FeaStudyObjectDetails, FeaStudyObjectKind,
    FeaStudyObjectPresentation, FeaStudyPresentation,
};
pub use project_host::ProjectFeaHost;
pub use report::{
    FeaCheckStatus, FeaRootSelector, FeaSourceCheckResult, FeaStudyCheckSummary,
    FeaStudyObjectCounts, FeaStudyProcedureSummary,
};
pub use root::{FeaRootSelectionError, ResolvedFeaRoot};
pub use snapshot::{
    FeaSourceSnapshotBuilder, FeaSourceSnapshotRequests, PreparedFeaSourceSnapshot,
};

#[cfg(test)]
mod tests;
