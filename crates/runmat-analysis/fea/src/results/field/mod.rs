//! Physics-owned recovered fields exposed to observation evaluation.

mod catalog;
mod contract;
mod topology;

pub use catalog::RecoveredFieldCatalog;
pub use contract::{
    RecoveredField, RecoveredFieldArtifact, RecoveredFieldArtifactEntry,
    RecoveredFieldArtifactIdentity, RecoveredFieldArtifactManifest, RecoveredFieldDefinition,
    RecoveredFieldDefinitionIdentity, RecoveredFieldEntityCounts, RecoveredFieldEntityDomain,
    RecoveredFieldError, RecoveredFieldIdentity, RecoveredFieldNumericRepresentation,
    RecoveredFieldSample, RecoveredFieldSelection, RecoveredFieldSelectionIdentity,
};
pub use topology::StructuralRecoveredFieldTopology;

#[cfg(test)]
mod tests;
