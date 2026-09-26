//! Borrowed validated field content and incremental canonical storage extraction.
use runmat_analysis_core::AnalysisField;

use super::{
    RecoveredField, RecoveredFieldDefinition, RecoveredFieldError, RecoveredFieldIdentity,
    RecoveredFieldPartition, RecoveredFieldPartitionDescriptor, RecoveredFieldSample,
    RECOVERED_FIELD_PARTITION_ENTITY_COUNT,
};

/// Validated identity over borrowed storage. Construction still encodes the full field identity;
/// borrowing removes numeric storage clones, not canonical identity materialization.
#[derive(Clone, Copy, Debug)]
pub struct RecoveredFieldView<'a> {
    identity: RecoveredFieldIdentity,
    definition: &'a RecoveredFieldDefinition,
    sample: RecoveredFieldSample,
    storage: &'a AnalysisField,
}

impl<'a> RecoveredFieldView<'a> {
    pub fn new(
        definition: &'a RecoveredFieldDefinition,
        sample: RecoveredFieldSample,
        storage: &'a AnalysisField,
    ) -> Result<Self, RecoveredFieldError> {
        Ok(Self {
            identity: RecoveredField::identity_for(definition, sample, storage)?,
            definition,
            sample,
            storage,
        })
    }

    pub fn from_field(field: &'a RecoveredField) -> Result<Self, RecoveredFieldError> {
        let view = Self::new(field.definition(), field.sample(), field.storage())?;
        if view.identity != field.identity() {
            return Err(RecoveredFieldError::IdentityMismatch);
        }
        Ok(view)
    }

    pub const fn identity(&self) -> RecoveredFieldIdentity {
        self.identity
    }
    pub const fn definition(&self) -> &'a RecoveredFieldDefinition {
        self.definition
    }
    pub const fn sample(&self) -> RecoveredFieldSample {
        self.sample
    }
    pub const fn storage(&self) -> &'a AnalysisField {
        self.storage
    }

    /// Validate host extraction once, then yield one bounded owned partition per iterator step.
    pub fn canonical_partitions(
        &self,
    ) -> Result<CanonicalFieldPartitions<'a>, RecoveredFieldError> {
        if self.storage.as_host_f64().is_none() {
            return Err(RecoveredFieldError::InvalidArtifact(
                "field partition extraction requires host-resident f64 values".into(),
            ));
        }
        Ok(CanonicalFieldPartitions {
            source: *self,
            next: 0,
            entity_count: self.storage.shape[0],
        })
    }

    /// Retains descriptors only; each extracted numeric partition is dropped before the next.
    pub fn canonical_partition_descriptors(
        &self,
    ) -> Result<Vec<RecoveredFieldPartitionDescriptor>, RecoveredFieldError> {
        self.canonical_partitions()?
            .map(|partition| {
                let partition = partition?;
                Ok(RecoveredFieldPartitionDescriptor {
                    partition: partition.identity,
                    first_source_index: partition.source_indices[0],
                    entity_count: partition.source_indices.len(),
                })
            })
            .collect()
    }
}

/// Incremental canonical partition extraction; callers choose whether to retain yielded storage.
#[derive(Debug)]
pub struct CanonicalFieldPartitions<'a> {
    source: RecoveredFieldView<'a>,
    next: usize,
    entity_count: usize,
}

impl Iterator for CanonicalFieldPartitions<'_> {
    type Item = Result<RecoveredFieldPartition, RecoveredFieldError>;

    fn size_hint(&self) -> (usize, Option<usize>) {
        let count =
            (self.entity_count - self.next).div_ceil(RECOVERED_FIELD_PARTITION_ENTITY_COUNT);
        (count, Some(count))
    }

    fn next(&mut self) -> Option<Self::Item> {
        if self.next == self.entity_count {
            return None;
        }
        let start = self.next;
        self.next = start
            .saturating_add(RECOVERED_FIELD_PARTITION_ENTITY_COUNT)
            .min(self.entity_count);
        Some(RecoveredFieldPartition::select_view(
            &self.source,
            (start..self.next).collect(),
        ))
    }
}

impl ExactSizeIterator for CanonicalFieldPartitions<'_> {}
impl std::iter::FusedIterator for CanonicalFieldPartitions<'_> {}

#[cfg(test)]
mod tests;
