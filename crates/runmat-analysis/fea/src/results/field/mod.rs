//! Physics-owned recovered fields exposed to observation evaluation.

mod catalog;
mod contract;

pub use catalog::RecoveredFieldCatalog;
pub use contract::{
    RecoveredField, RecoveredFieldArtifact, RecoveredFieldArtifactEntry,
    RecoveredFieldArtifactIdentity, RecoveredFieldArtifactManifest, RecoveredFieldDefinition,
    RecoveredFieldDefinitionIdentity, RecoveredFieldError, RecoveredFieldIdentity,
    RecoveredFieldNumericRepresentation, RecoveredFieldSample, RecoveredFieldSelection,
    RecoveredFieldSelectionIdentity,
};

#[cfg(test)]
mod tests;
