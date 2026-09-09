//! Physics-owned recovered fields exposed to observation evaluation.

mod catalog;
mod contract;

pub use catalog::RecoveredFieldCatalog;
pub use contract::{
    RecoveredField, RecoveredFieldArtifact, RecoveredFieldArtifactIdentity,
    RecoveredFieldDefinition, RecoveredFieldDefinitionIdentity, RecoveredFieldError,
    RecoveredFieldNumericRepresentation, RecoveredFieldSample,
};

#[cfg(test)]
mod tests;
