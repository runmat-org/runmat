//! Original-field publication descriptions; numerical values stay with their source owner.
use super::*;
use crate::execution::FieldRecoveryRequest;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OriginalFieldSourceEntry {
    pub definition: RecoveredFieldDefinition,
    pub sample: RecoveredFieldSample,
}

/// Describes retained original channels. This is neither a recovery kernel nor a proof.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OriginalFieldSourcePlan {
    pub request: FieldRecoveryRequest,
    pub entity_counts: RecoveredFieldEntityCounts,
    pub fields: Vec<OriginalFieldSourceEntry>,
}
impl OriginalFieldSourcePlan {
    pub fn validate(&self) -> Result<(), RecoveredFieldError> {
        self.request
            .validate()
            .map_err(|error| RecoveredFieldError::InvalidArtifact(error.to_string()))?;
        self.entity_counts.validate()?;
        if self.fields.is_empty() {
            return Err(RecoveredFieldError::InvalidArtifact(
                "original field source is empty".into(),
            ));
        }
        let mut ids = std::collections::BTreeSet::new();
        for entry in &self.fields {
            entry.sample.validate()?;
            entry
                .definition
                .storage_shape(entry.definition.stored_entities(self.entity_counts))?;
            if !ids.insert(entry.definition.field_id()) {
                return Err(RecoveredFieldError::InvalidArtifact(
                    "original field source has an ambiguous channel".into(),
                ));
            }
        }
        Ok(())
    }
}
