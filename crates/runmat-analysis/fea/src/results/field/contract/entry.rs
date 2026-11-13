//! One wire authority for retained catalog entries and sampled shape projections.
use runmat_canonical_codec::{profile_serialized_shape, SerializedCanonicalProfile};
use serde::ser::SerializeSeq;

use super::*;
use crate::results::field::RECOVERED_FIELD_PARTITION_ENTITY_COUNT;

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecoveredFieldArtifactEntry {
    pub field: RecoveredFieldIdentity,
    pub definition: RecoveredFieldDefinition,
    pub sample: RecoveredFieldSample,
    pub partitions: Vec<RecoveredFieldPartitionDescriptor>,
}

#[derive(Serialize)]
#[serde(rename = "RecoveredFieldArtifactEntry")]
struct EntrySerialization<'a, P: Serialize + ?Sized> {
    field: RecoveredFieldIdentity,
    definition: &'a RecoveredFieldDefinition,
    sample: RecoveredFieldSample,
    partitions: &'a P,
}

impl Serialize for RecoveredFieldArtifactEntry {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        EntrySerialization {
            field: self.field,
            definition: &self.definition,
            sample: self.sample,
            partitions: &self.partitions,
        }
        .serialize(serializer)
    }
}

struct Repeated<'a, T> {
    value: &'a T,
    count: usize,
}

impl<T: Serialize> Serialize for Repeated<'_, T> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(std::iter::repeat_n(self.value, self.count))
    }
}

struct CatalogRow<'a> {
    definitions: &'a [RecoveredFieldDefinition],
    entity_counts: &'a [usize],
    sample: RecoveredFieldSample,
    descriptor: RecoveredFieldPartitionDescriptor,
}

impl Serialize for CatalogRow<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut sequence = serializer.serialize_seq(Some(self.definitions.len()))?;
        for (definition, count) in self.definitions.iter().zip(self.entity_counts) {
            let partitions = Repeated {
                value: &self.descriptor,
                count: count.div_ceil(RECOVERED_FIELD_PARTITION_ENTITY_COUNT),
            };
            sequence.serialize_element(&EntrySerialization {
                field: RecoveredFieldIdentity(Digest::from_bytes([0; 32])),
                definition,
                sample: self.sample,
                partitions: &partitions,
            })?;
        }
        sequence.end()
    }
}

impl RecoveredFieldArtifactEntry {
    /// Profiles a nonempty sample-by-field catalog without constructing entry/descriptor arrays
    /// or numerical values. Definitions (including identifier lengths) and sample variant are
    /// supplied by the caller; numeric coordinates, counts and indices receive codec-wide bounds.
    /// Repeated descriptors are projection-only and do not claim actual coverage or identities.
    pub fn profile_sampled_catalog_shape(
        definitions: &[RecoveredFieldDefinition],
        entity_counts: &[usize],
        sample: RecoveredFieldSample,
        sample_count: usize,
    ) -> Result<SerializedCanonicalProfile, RecoveredFieldError> {
        if definitions.is_empty()
            || definitions.len() != entity_counts.len()
            || sample_count == 0
            || entity_counts.contains(&0)
        {
            return Err(RecoveredFieldError::InvalidArtifact(
                "catalog projection requires nonempty aligned definitions, entities and samples"
                    .into(),
            ));
        }
        sample.validate()?;
        let mut partition_count = 0usize;
        for (definition, count) in definitions.iter().zip(entity_counts) {
            definition.validate()?;
            count
                .checked_mul(definition.storage_component_count()?)
                .ok_or(RecoveredFieldError::StorageShapeMismatch)?;
            partition_count = partition_count
                .checked_add(count.div_ceil(RECOVERED_FIELD_PARTITION_ENTITY_COUNT))
                .ok_or(RecoveredFieldError::StorageShapeMismatch)?;
        }
        definitions
            .len()
            .checked_mul(sample_count)
            .ok_or(RecoveredFieldError::StorageShapeMismatch)?;
        partition_count
            .checked_mul(sample_count)
            .ok_or(RecoveredFieldError::StorageShapeMismatch)?;
        let row = CatalogRow {
            definitions,
            entity_counts,
            sample,
            descriptor: RecoveredFieldPartitionDescriptor {
                partition:
                    super::super::partition::RecoveredFieldPartitionIdentity::profile_placeholder(),
                first_source_index: 0,
                entity_count: 1,
            },
        };
        profile_serialized_shape(
            &Repeated {
                value: &row,
                count: sample_count,
            },
            FIELD_LIMITS,
        )
        .map_err(|error| RecoveredFieldError::Encoding(error.to_string()))
    }
}

#[cfg(test)]
mod tests;
