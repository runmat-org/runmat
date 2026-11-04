use std::collections::BTreeSet;

use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

use super::{
    RecoveredField, RecoveredFieldDefinition, RecoveredFieldError, RecoveredFieldIdentity,
    RecoveredFieldPartition, RecoveredFieldPartitionIdentity, RecoveredFieldSample,
    RecoveredFieldView,
};

mod selection;
pub use selection::{
    select_canonical_field_partitions, validate_canonical_field_partition_descriptors,
    RecoveredFieldPartitionSelection,
};

const FIELD_PARTITION_SET_PREFIX: &[u8] = b"runmat-fea-recovered-field-partition-set/v1\0";
/// Partition-set size is admitted by the recovery resource plan and checked against it by the
/// object inventory; the codec bounds it only by the wire range.
const FIELD_PARTITION_SET_LIMITS: CanonicalLimits =
    crate::contracts::canonical_digest::wire_bounded(CanonicalLimits::new(0, 0, 4_000_000, 32));

/// Canonical field-storage block size. It is part of the artifact contract rather than a
/// placement decision, so the same recovered field has the same storage partition identities on
/// every worker and every execution backend.
pub const RECOVERED_FIELD_PARTITION_ENTITY_COUNT: usize = 16_384;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RecoveredFieldPartitionSetIdentity(Digest);

impl RecoveredFieldPartitionSetIdentity {
    pub const fn digest(self) -> Digest {
        self.0
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecoveredFieldPartitionDescriptor {
    pub partition: RecoveredFieldPartitionIdentity,
    pub first_source_index: usize,
    pub entity_count: usize,
}

impl RecoveredFieldPartitionDescriptor {
    pub fn contains(&self, source_index: usize) -> bool {
        source_index >= self.first_source_index
            && source_index < self.first_source_index.saturating_add(self.entity_count)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "RecoveredFieldPartitionSetWire")]
pub struct RecoveredFieldPartitionSet {
    pub schema_version: u16,
    pub identity: RecoveredFieldPartitionSetIdentity,
    pub source_field: RecoveredFieldIdentity,
    pub definition: RecoveredFieldDefinition,
    pub sample: RecoveredFieldSample,
    pub partitions: Vec<RecoveredFieldPartition>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RecoveredFieldPartitionSetWire {
    schema_version: u16,
    identity: RecoveredFieldPartitionSetIdentity,
    source_field: RecoveredFieldIdentity,
    definition: RecoveredFieldDefinition,
    sample: RecoveredFieldSample,
    partitions: Vec<RecoveredFieldPartition>,
}

impl TryFrom<RecoveredFieldPartitionSetWire> for RecoveredFieldPartitionSet {
    type Error = RecoveredFieldError;

    fn try_from(wire: RecoveredFieldPartitionSetWire) -> Result<Self, Self::Error> {
        let set = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            source_field: wire.source_field,
            definition: wire.definition,
            sample: wire.sample,
            partitions: wire.partitions,
        };
        set.validate()?;
        Ok(set)
    }
}

impl RecoveredFieldPartitionSet {
    pub fn new(partitions: Vec<RecoveredFieldPartition>) -> Result<Self, RecoveredFieldError> {
        let first = partitions.first().ok_or_else(|| {
            RecoveredFieldError::InvalidArtifact("field partition set is empty".into())
        })?;
        let mut set = Self {
            schema_version: 1,
            identity: RecoveredFieldPartitionSetIdentity(Digest::sha256([])),
            source_field: first.source_field,
            definition: first.definition.clone(),
            sample: first.sample,
            partitions,
        };
        set.identity = set.derive_identity()?;
        set.validate()?;
        Ok(set)
    }

    pub fn value(&self, source_index: usize) -> Option<&[f64]> {
        self.partitions
            .iter()
            .find_map(|partition| partition.value(source_index))
    }

    pub fn source_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.partitions
            .iter()
            .flat_map(|partition| partition.source_indices.iter().copied())
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, RecoveredFieldError> {
        self.validate()?;
        encode_contract(
            FIELD_PARTITION_SET_PREFIX,
            "recovered-field-partition-set",
            self,
            FIELD_PARTITION_SET_LIMITS,
        )
        .map_err(|error| RecoveredFieldError::Encoding(error.to_string()))
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, RecoveredFieldError> {
        decode_contract(
            FIELD_PARTITION_SET_PREFIX,
            "recovered-field-partition-set",
            bytes,
            FIELD_PARTITION_SET_LIMITS,
        )
        .map_err(|error| RecoveredFieldError::Encoding(error.to_string()))
    }

    pub fn validate(&self) -> Result<(), RecoveredFieldError> {
        self.definition.validate()?;
        self.sample.validate()?;
        let mut indices = BTreeSet::new();
        for partition in &self.partitions {
            partition.validate()?;
            if partition.source_field != self.source_field
                || partition.definition != self.definition
                || partition.sample != self.sample
                || partition
                    .source_indices
                    .iter()
                    .any(|index| !indices.insert(*index))
            {
                return Err(RecoveredFieldError::InvalidArtifact(
                    "field partition set contains unrelated or overlapping partitions".into(),
                ));
            }
        }
        if self.schema_version != 1
            || self.partitions.is_empty()
            || self.identity != self.derive_identity()?
        {
            return Err(RecoveredFieldError::InvalidArtifact(
                "field partition set metadata or identity is invalid".into(),
            ));
        }
        Ok(())
    }

    fn derive_identity(&self) -> Result<RecoveredFieldPartitionSetIdentity, RecoveredFieldError> {
        derive_partition_set_identity(
            self.schema_version,
            self.source_field,
            &self.definition,
            self.sample,
            PartitionIdentities::Payloads(&self.partitions),
        )
    }
}

enum PartitionIdentities<'a> {
    Payloads(&'a [RecoveredFieldPartition]),
    Descriptors(&'a [&'a RecoveredFieldPartitionDescriptor]),
}

impl Serialize for PartitionIdentities<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Payloads(partitions) => {
                serializer.collect_seq(partitions.iter().map(|p| p.identity))
            }
            Self::Descriptors(partitions) => {
                serializer.collect_seq(partitions.iter().map(|p| p.partition))
            }
        }
    }
}

fn derive_partition_set_identity(
    schema_version: u16,
    source_field: RecoveredFieldIdentity,
    definition: &RecoveredFieldDefinition,
    sample: RecoveredFieldSample,
    partitions: PartitionIdentities<'_>,
) -> Result<RecoveredFieldPartitionSetIdentity, RecoveredFieldError> {
    crate::contracts::canonical_digest::canonical_contract_digest(
        FIELD_PARTITION_SET_PREFIX,
        "recovered-field-partition-set-identity",
        &(schema_version, source_field, definition, sample, partitions),
        FIELD_PARTITION_SET_LIMITS,
    )
    .map(RecoveredFieldPartitionSetIdentity)
    .map_err(|error| RecoveredFieldError::Encoding(error.to_string()))
}

pub fn canonical_field_partitions(
    field: &RecoveredField,
) -> Result<Vec<RecoveredFieldPartition>, RecoveredFieldError> {
    RecoveredFieldView::from_field(field)?
        .canonical_partitions()?
        .collect()
}

pub fn canonical_field_partition_descriptors(
    field: &RecoveredField,
) -> Result<Vec<RecoveredFieldPartitionDescriptor>, RecoveredFieldError> {
    RecoveredFieldView::from_field(field)?.canonical_partition_descriptors()
}

pub fn reconstruct_recovered_field(
    expected_identity: RecoveredFieldIdentity,
    definition: RecoveredFieldDefinition,
    sample: RecoveredFieldSample,
    partitions: Vec<RecoveredFieldPartition>,
    entity_count: usize,
) -> Result<RecoveredField, RecoveredFieldError> {
    let set = RecoveredFieldPartitionSet::new(partitions)?;
    if set.source_field != expected_identity
        || set.definition != definition
        || set.sample != sample
        || set.source_indices().ne(0..entity_count)
    {
        return Err(RecoveredFieldError::IdentityMismatch);
    }
    let components = definition.storage_component_count()?;
    let mut values = Vec::with_capacity(entity_count.saturating_mul(components));
    for partition in &set.partitions {
        values.extend_from_slice(
            partition
                .storage
                .as_host_f64()
                .ok_or(RecoveredFieldError::StorageShapeMismatch)?,
        );
    }
    let shape = definition.storage_shape(entity_count)?;
    // Every partition carries its source field's layout; the source identity covers it.
    let layout = set
        .partitions
        .first()
        .map(|partition| partition.storage.layout.clone())
        .ok_or(RecoveredFieldError::IdentityMismatch)?;
    let field = RecoveredField::new(
        definition.clone(),
        sample,
        runmat_analysis_core::AnalysisField::host_f64(definition.field_id(), layout, shape, values),
    )?;
    if field.identity() != expected_identity {
        return Err(RecoveredFieldError::IdentityMismatch);
    }
    Ok(field)
}
