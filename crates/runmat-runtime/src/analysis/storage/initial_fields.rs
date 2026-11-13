//! Original typed field catalogs share publication, chunk storage and bounded reads.
use super::*;
use runmat_analysis_fea::execution::InitialFieldBinding;
use runmat_analysis_fea::results::field::{
    RecoveredField, RecoveredFieldArtifact, RecoveredFieldArtifactEntry,
    RecoveredFieldArtifactManifest,
};
mod catalog;
pub(in crate::analysis::storage) mod read;
mod resolution;
pub(super) use catalog::validate_catalog;
pub use resolution::{
    export_initial_field, resolve_initial_field_references, resolve_initial_fields,
};
#[cfg(test)]
mod tests;
