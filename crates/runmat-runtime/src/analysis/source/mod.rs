mod compile;
mod geometry_resolver;
mod host;
mod project_host;
mod report;
mod root;

pub use compile::{
    check_fea_path, check_fea_path_in_root, check_fea_source_in_root, check_fea_sources_in_root,
    prepare_fea_path_in_root, prepare_fea_sources_in_root, CheckedFeaSource, PreparedFeaSource,
};
pub use host::{source_digest, FilesystemFeaHost};
pub use project_host::ProjectFeaHost;
pub use report::{FeaCheckStatus, FeaRootSelector, FeaSourceCheckResult};
pub use root::{FeaRootSelectionError, ResolvedFeaRoot};

#[cfg(test)]
mod tests;
