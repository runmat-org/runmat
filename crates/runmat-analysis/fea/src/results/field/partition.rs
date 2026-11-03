use runmat_analysis_core::AnalysisField;
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

mod profile;

#[cfg(test)]
mod tests;

use super::{
    RecoveredField, RecoveredFieldDefinition, RecoveredFieldError, RecoveredFieldIdentity,
    RecoveredFieldSample, RecoveredFieldView,
};

const FIELD_PARTITION_PREFIX: &[u8] = b"runmat-fea-recovered-field-partition/v1\0";
/// Partition size is admitted by the recovery resource plan and checked against it by the
/// object inventory; the codec bounds it only by the wire range.
const FIELD_PARTITION_LIMITS: CanonicalLimits =
    crate::contracts::canonical_digest::wire_bounded(CanonicalLimits::new(0, 0, 4_000_000, 32));

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RecoveredFieldPartitionIdentity(Digest);

impl RecoveredFieldPartitionIdentity {
    /// Fixed-width shape projection only; never published as an artifact identity.
    pub(super) const fn profile_placeholder() -> Self {
        Self(Digest::from_bytes([0; 32]))
    }

    pub const fn digest(self) -> Digest {
        self.0
    }
}

/// Canonical compact field values needed by one observation partition.
///
/// `source_indices` are global indices in the recovered field's entity domain. Values are stored
/// in that canonical order, so a worker can evaluate its partition without loading unrelated
/// field entities.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields, try_from = "RecoveredFieldPartitionWire")]
pub struct RecoveredFieldPartition {
    pub schema_version: u16,
    pub identity: RecoveredFieldPartitionIdentity,
    pub source_field: RecoveredFieldIdentity,
    pub definition: RecoveredFieldDefinition,
    pub sample: RecoveredFieldSample,
    pub source_indices: Vec<usize>,
    pub storage: AnalysisField,
}

// Shared wire authority for retained partitions and pre-solve shape profiling.
#[derive(Serialize)]
#[serde(rename = "RecoveredFieldPartition")]
struct PartitionSerialization<'a, I: Serialize + ?Sized, S: Serialize + ?Sized> {
    schema_version: u16,
    identity: Digest,
    source_field: Digest,
    definition: &'a RecoveredFieldDefinition,
    sample: RecoveredFieldSample,
    source_indices: &'a I,
    storage: &'a S,
}

impl Serialize for RecoveredFieldPartition {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        PartitionSerialization {
            schema_version: self.schema_version,
            identity: self.identity.digest(),
            source_field: self.source_field.digest(),
            definition: &self.definition,
            sample: self.sample,
            source_indices: &self.source_indices,
            storage: &self.storage,
        }
        .serialize(serializer)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RecoveredFieldPartitionWire {
    schema_version: u16,
    identity: RecoveredFieldPartitionIdentity,
    source_field: RecoveredFieldIdentity,
    definition: RecoveredFieldDefinition,
    sample: RecoveredFieldSample,
    source_indices: Vec<usize>,
    storage: AnalysisField,
}

impl TryFrom<RecoveredFieldPartitionWire> for RecoveredFieldPartition {
    type Error = RecoveredFieldError;

    fn try_from(wire: RecoveredFieldPartitionWire) -> Result<Self, Self::Error> {
        let partition = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            source_field: wire.source_field,
            definition: wire.definition,
            sample: wire.sample,
            source_indices: wire.source_indices,
            storage: wire.storage,
        };
        partition.validate()?;
        Ok(partition)
    }
}

impl RecoveredFieldPartition {
    pub fn select(
        source: &RecoveredField,
        source_indices: Vec<usize>,
    ) -> Result<Self, RecoveredFieldError> {
        Self::select_view(&RecoveredFieldView::from_field(source)?, source_indices)
    }

    /// Select a region-supported field: `support` holds the ascending global indices its values
    /// are stored at. Every requested index must lie inside the support (fails closed otherwise).
    pub fn select_supported(
        source: &RecoveredField,
        source_indices: Vec<usize>,
        support: &[usize],
    ) -> Result<Self, RecoveredFieldError> {
        let stored = source_indices
            .iter()
            .map(|index| {
                support.binary_search(index).map_err(|_| {
                    RecoveredFieldError::InvalidArtifact(
                        "observation target lies outside the field's supported entities".into(),
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::select_stored(
            &RecoveredFieldView::from_field(source)?,
            source_indices,
            &stored,
        )
    }

    /// Extract from content already validated by the shared borrowed-field identity boundary.
    pub fn select_view(
        source: &RecoveredFieldView<'_>,
        source_indices: Vec<usize>,
    ) -> Result<Self, RecoveredFieldError> {
        let stored = source_indices.clone();
        Self::select_stored(source, source_indices, &stored)
    }

    /// `source_indices` are global entity indices; `stored` their positions in storage.
    fn select_stored(
        source: &RecoveredFieldView<'_>,
        source_indices: Vec<usize>,
        stored: &[usize],
    ) -> Result<Self, RecoveredFieldError> {
        if source_indices.is_empty() || source_indices.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(RecoveredFieldError::InvalidArtifact(
                "field partition indices must be nonempty and strictly increasing".into(),
            ));
        }
        let components = source.definition().storage_component_count()?;
        let source_values =
            source
                .storage()
                .as_host_f64()
                .ok_or(RecoveredFieldError::InvalidArtifact(
                    "field partition extraction requires host-resident f64 values".into(),
                ))?;
        let mut values = Vec::with_capacity(source_indices.len().saturating_mul(components));
        for index in stored {
            let start = index
                .checked_mul(components)
                .ok_or(RecoveredFieldError::StorageShapeMismatch)?;
            let end = start
                .checked_add(components)
                .ok_or(RecoveredFieldError::StorageShapeMismatch)?;
            values.extend_from_slice(
                source_values
                    .get(start..end)
                    .ok_or(RecoveredFieldError::StorageShapeMismatch)?,
            );
        }
        let shape = source.definition().storage_shape(source_indices.len())?;
        let mut partition = Self {
            schema_version: 1,
            identity: RecoveredFieldPartitionIdentity(Digest::sha256([])),
            source_field: source.identity(),
            definition: source.definition().clone(),
            sample: source.sample(),
            source_indices,
            // The partition keeps its source's storage layout, published state (ordinal)
            // included, so the reassembled field has the source's identity.
            storage: AnalysisField::host_f64(
                source.definition().field_id(),
                source.storage().layout.clone(),
                shape,
                values,
            ),
        };
        partition.identity = partition.derive_identity()?;
        partition.validate()?;
        Ok(partition)
    }

    pub fn value(&self, source_index: usize) -> Option<&[f64]> {
        let local_index = self.source_indices.binary_search(&source_index).ok()?;
        let components = self.definition.storage_component_count().ok()?;
        let start = local_index.checked_mul(components)?;
        let end = start.checked_add(components)?;
        self.storage.as_host_f64()?.get(start..end)
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, RecoveredFieldError> {
        self.validate()?;
        encode_contract(
            FIELD_PARTITION_PREFIX,
            "recovered-field-partition",
            self,
            FIELD_PARTITION_LIMITS,
        )
        .map_err(|error| RecoveredFieldError::Encoding(error.to_string()))
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, RecoveredFieldError> {
        decode_contract(
            FIELD_PARTITION_PREFIX,
            "recovered-field-partition",
            bytes,
            FIELD_PARTITION_LIMITS,
        )
        .map_err(|error| RecoveredFieldError::Encoding(error.to_string()))
    }

    pub fn validate(&self) -> Result<(), RecoveredFieldError> {
        self.definition.validate()?;
        self.sample.validate()?;
        self.definition.validate_storage_shape(&self.storage)?;
        if self.storage.layout
            != self
                .definition
                .layout()
                .with_ordinal(self.storage.layout.ordinal)
        {
            return Err(RecoveredFieldError::StorageLayoutMismatch);
        }
        let expected_shape = self.definition.storage_shape(self.source_indices.len())?;
        if self.schema_version != 1
            || self.source_indices.is_empty()
            || self
                .source_indices
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            || self.storage.shape != expected_shape
            || self
                .storage
                .as_host_f64()
                .is_none_or(|values| values.iter().any(|value| !value.is_finite()))
            || self.identity != self.derive_identity()?
        {
            return Err(RecoveredFieldError::InvalidArtifact(
                "field partition metadata, values, or identity is invalid".into(),
            ));
        }
        Ok(())
    }

    fn derive_identity(&self) -> Result<RecoveredFieldPartitionIdentity, RecoveredFieldError> {
        crate::contracts::canonical_digest::canonical_contract_digest(
            FIELD_PARTITION_PREFIX,
            "recovered-field-partition-identity",
            &(
                self.schema_version,
                self.source_field,
                &self.definition,
                self.sample,
                &self.source_indices,
                &self.storage,
            ),
            FIELD_PARTITION_LIMITS,
        )
        .map(RecoveredFieldPartitionIdentity)
        .map_err(|error| RecoveredFieldError::Encoding(error.to_string()))
    }
}
