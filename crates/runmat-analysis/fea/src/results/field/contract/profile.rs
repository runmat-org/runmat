//! Whole catalog shape projection using the actual manifest and entry serializers.
use super::*;
use runmat_canonical_codec::{profile_serialized_shape, SerializedCanonicalProfile};

impl RecoveredFieldArtifactManifest {
    /// Header and all final-sample entries, without numerical arrays or hypothetical descriptors.
    pub fn profile_final_shape(
        request: &crate::execution::FieldRecoveryRequest,
        definitions: &[RecoveredFieldDefinition],
        counts: &[usize],
    ) -> Result<SerializedCanonicalProfile, RecoveredFieldError> {
        Self::profile_sampled_shape(request, definitions, counts, RecoveredFieldSample::Final)
    }

    /// One catalog in the actual sample variant, without field values or real descriptors.
    pub fn profile_sampled_shape(
        request: &crate::execution::FieldRecoveryRequest,
        definitions: &[RecoveredFieldDefinition],
        counts: &[usize],
        sample: RecoveredFieldSample,
    ) -> Result<SerializedCanonicalProfile, RecoveredFieldError> {
        Self::profile_sampled_history_shape(request, definitions, counts, sample, 1)
    }

    /// One catalog for repeated native fields over a retained sample sequence. This profiles
    /// the actual entry serializer without materializing field values or placeholder arrays.
    pub fn profile_sampled_history_shape(
        request: &crate::execution::FieldRecoveryRequest,
        definitions: &[RecoveredFieldDefinition],
        counts: &[usize],
        sample: RecoveredFieldSample,
        sample_count: usize,
    ) -> Result<SerializedCanonicalProfile, RecoveredFieldError> {
        let entries = RecoveredFieldArtifactEntry::profile_sampled_catalog_shape(
            definitions,
            counts,
            sample,
            sample_count,
        )?;
        let header = Self {
            schema_version: 1,
            identity: RecoveredFieldArtifactIdentity(Digest::sha256([])),
            recovery_request: request.identity(),
            source: request.source,
            topology_digest: request.topology.discretization_digest(),
            entity_counts: RecoveredFieldEntityCounts {
                oriented_faces: 0,
                nodes: 0,
                volume_elements: 0,
                volume_integration_points: 0,
                oriented_edges: 0,
                beam_elements: 0,
                shell_elements: 0,
                connectors: 0,
                constraint_groups: 0,
            },
            fields: vec![],
        };
        let profile = profile_serialized_shape(&header, FIELD_LIMITS)
            .map_err(|error| RecoveredFieldError::Encoding(error.to_string()))?;
        Ok(SerializedCanonicalProfile {
            encoded_bytes_upper_bound: profile
                .encoded_bytes_upper_bound
                .checked_add(entries.encoded_bytes_upper_bound)
                .ok_or(RecoveredFieldError::StorageShapeMismatch)?,
            materialization_resident_bytes_estimate: profile
                .materialization_resident_bytes_estimate
                .checked_add(entries.materialization_resident_bytes_estimate)
                .ok_or(RecoveredFieldError::StorageShapeMismatch)?,
        })
    }
}
