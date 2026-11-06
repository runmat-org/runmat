//! Pre-value bound for the shared complete recovered-field identity materialization.
use runmat_canonical_codec::{profile_serialized_shape, SerializedCanonicalProfile};

use super::*;

struct ProjectedValues(usize);

impl Serialize for ProjectedValues {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(std::iter::repeat_n(0.0_f64, self.0))
    }
}

impl RecoveredField {
    /// Bounds one host-f64 field identity using the same identity tuple and core field schema
    /// as `identity_for`. Scalars are counted at the shared codec's maximum encoded width.
    /// No numerical arrays or canonical value tree are allocated; small shape/identity metadata
    /// may allocate. This is not a bound for histories, partitions, or overlapping output buffers.
    pub fn identity_profile_shape(
        definition: &RecoveredFieldDefinition,
        sample: RecoveredFieldSample,
        entity_count: usize,
    ) -> Result<SerializedCanonicalProfile, RecoveredFieldError> {
        definition.validate()?;
        sample.validate()?;
        let value_count = entity_count
            .checked_mul(definition.storage_component_count()?)
            .ok_or(RecoveredFieldError::StorageShapeMismatch)?;
        let shape = definition.storage_shape(entity_count)?;
        let values = ProjectedValues(value_count);
        let layout = definition.profile_layout();
        let storage =
            AnalysisField::host_f64_serialization(definition.field_id(), &layout, &shape, &values);
        let content = identity_content(definition, sample, &storage);
        profile_serialized_shape(&content, FIELD_LIMITS)
            .map_err(|error| RecoveredFieldError::Encoding(error.to_string()))
    }

    /// Requested workspace of `identity_for`: the direct encoder's profile of the same identity
    /// tuple (its reserved output plus held buffers). No value tree is built by that encoder.
    /// Scalars are counted at the shared codec's maximum encoded width.
    pub fn identity_encoding_workspace_bytes(
        definition: &RecoveredFieldDefinition,
        sample: RecoveredFieldSample,
        entity_count: usize,
    ) -> Result<u64, RecoveredFieldError> {
        definition.validate()?;
        sample.validate()?;
        let value_count = entity_count
            .checked_mul(definition.storage_component_count()?)
            .ok_or(RecoveredFieldError::StorageShapeMismatch)?;
        let shape = definition.storage_shape(entity_count)?;
        let values = ProjectedValues(value_count);
        let layout = definition.profile_layout();
        let storage =
            AnalysisField::host_f64_serialization(definition.field_id(), &layout, &shape, &values);
        let content = identity_content(definition, sample, &storage);
        runmat_canonical_codec::profile_canonical_encoding(&content, FIELD_LIMITS)
            .map(|profile| profile.resident_workspace_bytes_estimate)
            .map_err(|error| RecoveredFieldError::Encoding(error.to_string()))
    }
}

#[cfg(test)]
mod tests;
