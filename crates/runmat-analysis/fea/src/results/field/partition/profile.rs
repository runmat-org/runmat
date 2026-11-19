use runmat_canonical_codec::{profile_serialized_shape, SerializedCanonicalProfile};

use super::*;

struct Repeated<T> {
    value: T,
    count: usize,
}

impl<T: Serialize> Serialize for Repeated<T> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(std::iter::repeat_n(&self.value, self.count))
    }
}

impl RecoveredFieldPartition {
    /// Bound one canonical host partition before numerical values exist.
    ///
    /// Uses the actual partition and core field wire schemas without allocating dummy numeric
    /// arrays. This bounds encoding/materialization of one contract, not retained solver fields,
    /// all partitions, transport chunks, or concurrent producer lifetimes.
    pub fn profile_shape(
        definition: &RecoveredFieldDefinition,
        sample: RecoveredFieldSample,
        entity_count: usize,
    ) -> Result<SerializedCanonicalProfile, RecoveredFieldError> {
        Self::profile_shapes(definition, sample, entity_count).map(|(profile, _)| profile)
    }

    /// Typed storage of one decoded canonical host partition of this shape, priced from the
    /// serialization shape as `canonical_payload_typed_resident_bytes_estimate` prices the wire
    /// (without the encoded bytes that estimate keeps alongside).
    pub fn profile_shape_typed_resident_bytes(
        definition: &RecoveredFieldDefinition,
        sample: RecoveredFieldSample,
        entity_count: usize,
    ) -> Result<u64, RecoveredFieldError> {
        Self::profile_shapes(definition, sample, entity_count).map(|(_, typed)| typed)
    }

    /// The wire profile and the typed resident estimate of one partition shape.
    fn profile_shapes(
        definition: &RecoveredFieldDefinition,
        sample: RecoveredFieldSample,
        entity_count: usize,
    ) -> Result<(SerializedCanonicalProfile, u64), RecoveredFieldError> {
        definition.validate()?;
        sample.validate()?;
        if entity_count == 0 || entity_count > super::super::RECOVERED_FIELD_PARTITION_ENTITY_COUNT
        {
            return Err(RecoveredFieldError::InvalidArtifact(
                "profile requires a nonempty canonical field partition".into(),
            ));
        }
        let count = entity_count
            .checked_mul(definition.storage_component_count()?)
            .ok_or(RecoveredFieldError::StorageShapeMismatch)?;
        let shape = definition.storage_shape(entity_count)?;
        let values = Repeated {
            value: 0.0_f64,
            count,
        };
        let indices = Repeated {
            value: 0_usize,
            count: entity_count,
        };
        let layout = definition.profile_layout();
        let storage =
            AnalysisField::host_f64_serialization(definition.field_id(), &layout, &shape, &values);
        // Transparent digest identities always serialize to fixed-width hexadecimal strings.
        // These are profile-only placeholders; no artifact or semantic identity is published.
        let placeholder = Digest::sha256([]);
        let shape = PartitionSerialization {
            schema_version: 1,
            identity: placeholder,
            source_field: placeholder,
            definition,
            sample,
            source_indices: &indices,
            storage: &storage,
        };
        let encoding = |error: runmat_canonical_codec::CanonicalCodecError| {
            RecoveredFieldError::Encoding(error.to_string())
        };
        Ok((
            profile_serialized_shape(&shape, FIELD_PARTITION_LIMITS).map_err(encoding)?,
            runmat_canonical_codec::profile_typed_resident_bytes(&shape, FIELD_PARTITION_LIMITS)
                .map_err(encoding)?,
        ))
    }
}
