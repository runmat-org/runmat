use std::ops::Deref;

use runmat_analysis_core::{
    AnalysisField, AnalysisFieldValues, FieldLocation, ObservationQuantity, ObservationValueShape,
};
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::execution::{FieldRecoveryRequestIdentity, FieldRecoverySource};

use super::partition_set::{
    canonical_field_partition_descriptors, RecoveredFieldPartitionDescriptor,
    RECOVERED_FIELD_PARTITION_ENTITY_COUNT,
};

const FIELD_PREFIX: &[u8] = b"runmat-fea-recovered-field/v1\0";
const FIELD_LIMITS: CanonicalLimits = CanonicalLimits::new(64 * 1024 * 1024, 1_000_000, 4096, 32);

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RecoveredFieldDefinitionIdentity(Digest);

impl RecoveredFieldDefinitionIdentity {
    pub const fn digest(self) -> Digest {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RecoveredFieldArtifactIdentity(Digest);

impl RecoveredFieldArtifactIdentity {
    pub const fn digest(self) -> Digest {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RecoveredFieldIdentity(Digest);

impl RecoveredFieldIdentity {
    pub const fn digest(self) -> Digest {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RecoveredFieldSelectionIdentity(Digest);

impl RecoveredFieldSelectionIdentity {
    pub const fn digest(self) -> Digest {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecoveredFieldNumericRepresentation {
    Real,
    ComplexInterleaved,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecoveredFieldEntityDomain {
    StructuralNodes,
    SolidElements,
    BeamElements,
    ShellElements,
    ConstraintGroups,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecoveredFieldEntityCounts {
    pub structural_nodes: usize,
    pub solid_elements: usize,
    pub beam_elements: usize,
    pub shell_elements: usize,
    pub constraint_groups: usize,
}

impl RecoveredFieldEntityCounts {
    pub const fn entity_count(self, domain: RecoveredFieldEntityDomain) -> usize {
        match domain {
            RecoveredFieldEntityDomain::StructuralNodes => self.structural_nodes,
            RecoveredFieldEntityDomain::SolidElements => self.solid_elements,
            RecoveredFieldEntityDomain::BeamElements => self.beam_elements,
            RecoveredFieldEntityDomain::ShellElements => self.shell_elements,
            RecoveredFieldEntityDomain::ConstraintGroups => self.constraint_groups,
        }
    }

    pub(super) fn validate(self) -> Result<(), RecoveredFieldError> {
        if self.structural_nodes == 0
            || self
                .solid_elements
                .checked_add(self.beam_elements)
                .and_then(|count| count.checked_add(self.shell_elements))
                .is_none_or(|count| count == 0)
        {
            return Err(RecoveredFieldError::InvalidArtifact(
                "field topology requires structural nodes and at least one element".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "domain", content = "coordinate")]
pub enum RecoveredFieldSample {
    Final,
    TimeSeconds(f64),
    ModeIndex(usize),
    FrequencyHertz(f64),
    LoadFactor(f64),
    StudyParameterValue(f64),
}

impl RecoveredFieldSample {
    pub(super) fn validate(self) -> Result<(), RecoveredFieldError> {
        match self {
            Self::Final | Self::ModeIndex(_) => Ok(()),
            Self::TimeSeconds(value)
            | Self::FrequencyHertz(value)
            | Self::LoadFactor(value)
            | Self::StudyParameterValue(value)
                if value.is_finite() && value.to_bits() != (-0.0_f64).to_bits() =>
            {
                Ok(())
            }
            _ => Err(RecoveredFieldError::InvalidSample),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "RecoveredFieldDefinitionData")]
pub struct RecoveredFieldDefinition {
    identity: RecoveredFieldDefinitionIdentity,
    field_id: String,
    quantity: ObservationQuantity,
    location: FieldLocation,
    entity_domain: RecoveredFieldEntityDomain,
    value_shape: ObservationValueShape,
    numeric_representation: RecoveredFieldNumericRepresentation,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RecoveredFieldDefinitionData {
    identity: RecoveredFieldDefinitionIdentity,
    field_id: String,
    quantity: ObservationQuantity,
    location: FieldLocation,
    entity_domain: RecoveredFieldEntityDomain,
    value_shape: ObservationValueShape,
    numeric_representation: RecoveredFieldNumericRepresentation,
}

impl TryFrom<RecoveredFieldDefinitionData> for RecoveredFieldDefinition {
    type Error = RecoveredFieldError;

    fn try_from(data: RecoveredFieldDefinitionData) -> Result<Self, Self::Error> {
        let definition = Self {
            identity: data.identity,
            field_id: data.field_id,
            quantity: data.quantity,
            location: data.location,
            entity_domain: data.entity_domain,
            value_shape: data.value_shape,
            numeric_representation: data.numeric_representation,
        };
        definition.validate()?;
        Ok(definition)
    }
}

impl RecoveredFieldDefinition {
    pub fn new(
        field_id: impl Into<String>,
        quantity: ObservationQuantity,
        location: FieldLocation,
        entity_domain: RecoveredFieldEntityDomain,
        value_shape: ObservationValueShape,
        numeric_representation: RecoveredFieldNumericRepresentation,
    ) -> Result<Self, RecoveredFieldError> {
        let mut definition = Self {
            identity: RecoveredFieldDefinitionIdentity(Digest::sha256([])),
            field_id: field_id.into(),
            quantity,
            location,
            entity_domain,
            value_shape,
            numeric_representation,
        };
        definition.identity = definition.derive_identity()?;
        definition.validate()?;
        Ok(definition)
    }

    pub const fn identity(&self) -> RecoveredFieldDefinitionIdentity {
        self.identity
    }

    pub fn field_id(&self) -> &str {
        &self.field_id
    }

    pub const fn quantity(&self) -> ObservationQuantity {
        self.quantity
    }

    pub const fn location(&self) -> FieldLocation {
        self.location
    }

    pub const fn entity_domain(&self) -> RecoveredFieldEntityDomain {
        self.entity_domain
    }

    pub fn value_shape(&self) -> &ObservationValueShape {
        &self.value_shape
    }

    pub const fn numeric_representation(&self) -> RecoveredFieldNumericRepresentation {
        self.numeric_representation
    }

    pub(crate) fn logical_component_count(&self) -> Result<usize, RecoveredFieldError> {
        self.value_shape
            .component_count()
            .ok_or(RecoveredFieldError::QuantityShapeMismatch)
    }

    pub(crate) fn storage_component_count(&self) -> Result<usize, RecoveredFieldError> {
        let logical = self.logical_component_count()?;
        match self.numeric_representation {
            RecoveredFieldNumericRepresentation::Real => Ok(logical),
            RecoveredFieldNumericRepresentation::ComplexInterleaved => logical
                .checked_mul(2)
                .ok_or(RecoveredFieldError::StorageShapeMismatch),
        }
    }

    pub(crate) fn storage_shape(
        &self,
        entity_count: usize,
    ) -> Result<Vec<usize>, RecoveredFieldError> {
        let logical = self.logical_component_count()?;
        Ok(match (self.numeric_representation, logical) {
            (RecoveredFieldNumericRepresentation::Real, 1) => vec![entity_count],
            (RecoveredFieldNumericRepresentation::Real, _) => vec![entity_count, logical],
            (RecoveredFieldNumericRepresentation::ComplexInterleaved, 1) => {
                vec![entity_count, 2]
            }
            (RecoveredFieldNumericRepresentation::ComplexInterleaved, _) => {
                vec![entity_count, logical, 2]
            }
        })
    }

    fn derive_identity(&self) -> Result<RecoveredFieldDefinitionIdentity, RecoveredFieldError> {
        encode_contract(
            FIELD_PREFIX,
            "recovered-field-definition",
            &(
                self.quantity,
                self.location,
                self.entity_domain,
                &self.value_shape,
                self.numeric_representation,
            ),
            FIELD_LIMITS,
        )
        .map(Digest::sha256)
        .map(RecoveredFieldDefinitionIdentity)
        .map_err(|error| RecoveredFieldError::Encoding(error.to_string()))
    }

    pub fn validate(&self) -> Result<(), RecoveredFieldError> {
        if self.field_id.trim().is_empty() {
            return Err(RecoveredFieldError::EmptyIdentifier);
        }
        if !matches!(
            (self.location, self.entity_domain),
            (
                FieldLocation::Node,
                RecoveredFieldEntityDomain::StructuralNodes
            ) | (
                FieldLocation::Element,
                RecoveredFieldEntityDomain::SolidElements
            ) | (
                FieldLocation::Element,
                RecoveredFieldEntityDomain::BeamElements
            ) | (
                FieldLocation::Element,
                RecoveredFieldEntityDomain::ShellElements
            ) | (
                FieldLocation::Constraint,
                RecoveredFieldEntityDomain::ConstraintGroups
            )
        ) {
            return Err(RecoveredFieldError::CatalogUnsupportedLocation);
        }
        if self.value_shape.component_count().is_none()
            || !self
                .value_shape
                .is_storage_equivalent_to(&self.quantity.canonical_value_shape())
        {
            return Err(RecoveredFieldError::QuantityShapeMismatch);
        }
        if self.identity != self.derive_identity()? {
            return Err(RecoveredFieldError::IdentityMismatch);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "RecoveredFieldData")]
pub struct RecoveredField {
    identity: RecoveredFieldIdentity,
    definition: RecoveredFieldDefinition,
    sample: RecoveredFieldSample,
    storage: AnalysisField,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RecoveredFieldData {
    identity: RecoveredFieldIdentity,
    definition: RecoveredFieldDefinition,
    sample: RecoveredFieldSample,
    storage: AnalysisField,
}

impl TryFrom<RecoveredFieldData> for RecoveredField {
    type Error = RecoveredFieldError;

    fn try_from(data: RecoveredFieldData) -> Result<Self, Self::Error> {
        let field = Self {
            identity: data.identity,
            definition: data.definition,
            sample: data.sample,
            storage: data.storage,
        };
        field.validate()?;
        Ok(field)
    }
}

impl RecoveredField {
    pub fn new(
        definition: RecoveredFieldDefinition,
        sample: RecoveredFieldSample,
        storage: AnalysisField,
    ) -> Result<Self, RecoveredFieldError> {
        let field = Self {
            identity: RecoveredFieldIdentity(Digest::sha256([])),
            definition,
            sample,
            storage,
        };
        let mut field = field;
        field.identity = field.derive_identity()?;
        field.validate()?;
        Ok(field)
    }

    pub const fn identity(&self) -> RecoveredFieldIdentity {
        self.identity
    }

    pub fn definition(&self) -> &RecoveredFieldDefinition {
        &self.definition
    }

    pub const fn sample(&self) -> RecoveredFieldSample {
        self.sample
    }

    pub fn storage(&self) -> &AnalysisField {
        &self.storage
    }

    pub fn into_storage(self) -> AnalysisField {
        self.storage
    }

    pub fn validate(&self) -> Result<(), RecoveredFieldError> {
        self.definition.validate()?;
        self.sample.validate()?;
        if self.storage.field_id != self.definition.field_id {
            return Err(RecoveredFieldError::StorageIdentifierMismatch);
        }
        let expected = self
            .storage
            .shape
            .iter()
            .try_fold(1usize, |count, extent| count.checked_mul(*extent))
            .ok_or(RecoveredFieldError::StorageShapeMismatch)?;
        let actual = match &self.storage.values {
            AnalysisFieldValues::HostF64(values) => values.len(),
            AnalysisFieldValues::DeviceRef(reference) => reference.element_count,
        };
        let entity_count = self
            .storage
            .shape
            .first()
            .copied()
            .ok_or(RecoveredFieldError::StorageShapeMismatch)?;
        if expected != actual
            || self.storage.shape != self.definition.storage_shape(entity_count)?
        {
            return Err(RecoveredFieldError::StorageShapeMismatch);
        }
        if self.identity != self.derive_identity()? {
            return Err(RecoveredFieldError::IdentityMismatch);
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, RecoveredFieldError> {
        self.validate()?;
        encode_contract(FIELD_PREFIX, "recovered-field", self, FIELD_LIMITS)
            .map_err(|error| RecoveredFieldError::Encoding(error.to_string()))
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, RecoveredFieldError> {
        decode_contract(FIELD_PREFIX, "recovered-field", bytes, FIELD_LIMITS)
            .map_err(|error| RecoveredFieldError::Encoding(error.to_string()))
    }

    fn derive_identity(&self) -> Result<RecoveredFieldIdentity, RecoveredFieldError> {
        encode_contract(
            FIELD_PREFIX,
            "recovered-field-identity",
            &(&self.definition, self.sample, &self.storage),
            FIELD_LIMITS,
        )
        .map(Digest::sha256)
        .map(RecoveredFieldIdentity)
        .map_err(|error| RecoveredFieldError::Encoding(error.to_string()))
    }
}

impl Deref for RecoveredField {
    type Target = AnalysisField;

    fn deref(&self) -> &Self::Target {
        &self.storage
    }
}

/// Canonical, source-bound recovered fields for one topology and numerical result.
///
/// This artifact is the portable boundary between family-owned recovery and solver-neutral
/// observation evaluation. Its identity excludes chunking and placement.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "RecoveredFieldArtifactWire")]
pub struct RecoveredFieldArtifact {
    pub schema_version: u16,
    identity: RecoveredFieldArtifactIdentity,
    pub recovery_request: FieldRecoveryRequestIdentity,
    pub source: FieldRecoverySource,
    pub topology_digest: Digest,
    pub entity_counts: RecoveredFieldEntityCounts,
    fields: Vec<RecoveredField>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecoveredFieldArtifactEntry {
    pub field: RecoveredFieldIdentity,
    pub definition: RecoveredFieldDefinition,
    pub sample: RecoveredFieldSample,
    pub partitions: Vec<RecoveredFieldPartitionDescriptor>,
}

/// Lightweight semantic catalog for a recovered-field artifact.
///
/// The catalog carries enough information to resolve observations and select exact field payloads
/// without loading any field values. Its identity is the identity of the complete artifact.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecoveredFieldArtifactManifest {
    pub schema_version: u16,
    pub identity: RecoveredFieldArtifactIdentity,
    pub recovery_request: FieldRecoveryRequestIdentity,
    pub source: FieldRecoverySource,
    pub topology_digest: Digest,
    pub entity_counts: RecoveredFieldEntityCounts,
    pub fields: Vec<RecoveredFieldArtifactEntry>,
}

impl RecoveredFieldArtifactManifest {
    pub fn validate(&self) -> Result<(), RecoveredFieldError> {
        if self.schema_version != 1 || self.fields.is_empty() {
            return Err(RecoveredFieldError::InvalidArtifact(
                "field manifest requires a supported schema, nonempty topology, and fields".into(),
            ));
        }
        self.entity_counts.validate()?;
        let mut field_ids = std::collections::BTreeSet::new();
        let mut sample_keys = std::collections::BTreeSet::new();
        let mut partition_ids = std::collections::BTreeSet::new();
        for entry in &self.fields {
            entry.definition.validate()?;
            entry.sample.validate()?;
            let expected_entity_count = self
                .entity_counts
                .entity_count(entry.definition.entity_domain());
            let mut next_source_index = 0usize;
            for (ordinal, partition) in entry.partitions.iter().enumerate() {
                let is_last = ordinal + 1 == entry.partitions.len();
                if partition.entity_count == 0
                    || partition.first_source_index != next_source_index
                    || (!is_last
                        && partition.entity_count != RECOVERED_FIELD_PARTITION_ENTITY_COUNT)
                    || partition.entity_count > RECOVERED_FIELD_PARTITION_ENTITY_COUNT
                    || !partition_ids.insert(partition.partition)
                {
                    return Err(RecoveredFieldError::InvalidArtifact(
                        "field storage partitions do not canonically cover their entity domain"
                            .into(),
                    ));
                }
                next_source_index = next_source_index
                    .checked_add(partition.entity_count)
                    .ok_or(RecoveredFieldError::StorageShapeMismatch)?;
            }
            if entry.partitions.is_empty() || next_source_index != expected_entity_count {
                return Err(RecoveredFieldError::InvalidArtifact(
                    "field storage partitions do not cover their complete entity domain".into(),
                ));
            }
            if !field_ids.insert(entry.field)
                || !sample_keys.insert(sample_identity(entry.definition.identity(), entry.sample)?)
            {
                return Err(RecoveredFieldError::CatalogDuplicateSample);
            }
        }
        if self.identity != derive_artifact_identity(self)? {
            return Err(RecoveredFieldError::IdentityMismatch);
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, RecoveredFieldError> {
        self.validate()?;
        encode_contract(
            FIELD_PREFIX,
            "recovered-field-artifact-manifest",
            self,
            FIELD_LIMITS,
        )
        .map_err(|error| RecoveredFieldError::Encoding(error.to_string()))
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, RecoveredFieldError> {
        let manifest: Self = decode_contract(
            FIELD_PREFIX,
            "recovered-field-artifact-manifest",
            bytes,
            FIELD_LIMITS,
        )
        .map_err(|error| RecoveredFieldError::Encoding(error.to_string()))?;
        manifest.validate()?;
        Ok(manifest)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecoveredFieldSelection {
    pub schema_version: u16,
    pub identity: RecoveredFieldSelectionIdentity,
    pub source: RecoveredFieldArtifactManifest,
    pub fields: Vec<RecoveredFieldIdentity>,
}

impl RecoveredFieldSelection {
    pub fn new(
        source: RecoveredFieldArtifactManifest,
        fields: Vec<RecoveredFieldIdentity>,
    ) -> Result<Self, RecoveredFieldError> {
        let mut selection = Self {
            schema_version: 1,
            identity: RecoveredFieldSelectionIdentity(Digest::sha256([])),
            source,
            fields,
        };
        selection.identity = selection.derive_identity()?;
        selection.validate()?;
        Ok(selection)
    }

    pub fn validate(&self) -> Result<(), RecoveredFieldError> {
        self.source.validate()?;
        if self.schema_version != 1 || self.fields.is_empty() {
            return Err(RecoveredFieldError::InvalidArtifact(
                "field selection requires a supported schema and selected fields".into(),
            ));
        }
        let selected = self
            .fields
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>();
        if selected.len() != self.fields.len() {
            return Err(RecoveredFieldError::InvalidArtifact(
                "field selection contains duplicate field identities".into(),
            ));
        }
        let canonical = self
            .source
            .fields
            .iter()
            .filter(|entry| selected.contains(&entry.field))
            .map(|entry| entry.field)
            .collect::<Vec<_>>();
        if canonical != self.fields {
            return Err(RecoveredFieldError::InvalidArtifact(
                "field selection is absent from or reordered relative to its source manifest"
                    .into(),
            ));
        }
        if self.identity != self.derive_identity()? {
            return Err(RecoveredFieldError::IdentityMismatch);
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, RecoveredFieldError> {
        self.validate()?;
        encode_contract(
            FIELD_PREFIX,
            "recovered-field-selection",
            self,
            FIELD_LIMITS,
        )
        .map_err(|error| RecoveredFieldError::Encoding(error.to_string()))
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, RecoveredFieldError> {
        let selection: Self = decode_contract(
            FIELD_PREFIX,
            "recovered-field-selection",
            bytes,
            FIELD_LIMITS,
        )
        .map_err(|error| RecoveredFieldError::Encoding(error.to_string()))?;
        selection.validate()?;
        Ok(selection)
    }

    fn derive_identity(&self) -> Result<RecoveredFieldSelectionIdentity, RecoveredFieldError> {
        encode_contract(
            FIELD_PREFIX,
            "recovered-field-selection-identity",
            &(self.schema_version, self.source.identity, &self.fields),
            FIELD_LIMITS,
        )
        .map(Digest::sha256)
        .map(RecoveredFieldSelectionIdentity)
        .map_err(|error| RecoveredFieldError::Encoding(error.to_string()))
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RecoveredFieldArtifactWire {
    schema_version: u16,
    identity: RecoveredFieldArtifactIdentity,
    recovery_request: FieldRecoveryRequestIdentity,
    source: FieldRecoverySource,
    topology_digest: Digest,
    entity_counts: RecoveredFieldEntityCounts,
    fields: Vec<RecoveredField>,
}

impl TryFrom<RecoveredFieldArtifactWire> for RecoveredFieldArtifact {
    type Error = RecoveredFieldError;

    fn try_from(wire: RecoveredFieldArtifactWire) -> Result<Self, Self::Error> {
        let artifact = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            recovery_request: wire.recovery_request,
            source: wire.source,
            topology_digest: wire.topology_digest,
            entity_counts: wire.entity_counts,
            fields: wire.fields,
        };
        artifact.validate()?;
        Ok(artifact)
    }
}

impl RecoveredFieldArtifact {
    pub fn new(
        recovery_request: FieldRecoveryRequestIdentity,
        source: FieldRecoverySource,
        topology_digest: Digest,
        entity_counts: RecoveredFieldEntityCounts,
        fields: Vec<RecoveredField>,
    ) -> Result<Self, RecoveredFieldError> {
        let mut artifact = Self {
            schema_version: 1,
            identity: RecoveredFieldArtifactIdentity(Digest::sha256([])),
            recovery_request,
            source,
            topology_digest,
            entity_counts,
            fields,
        };
        artifact.identity = artifact.derive_identity()?;
        artifact.validate()?;
        Ok(artifact)
    }

    pub const fn identity(&self) -> RecoveredFieldArtifactIdentity {
        self.identity
    }

    pub fn fields(&self) -> &[RecoveredField] {
        &self.fields
    }

    pub fn into_fields(self) -> Vec<RecoveredField> {
        self.fields
    }

    pub fn manifest(&self) -> Result<RecoveredFieldArtifactManifest, RecoveredFieldError> {
        let manifest = RecoveredFieldArtifactManifest {
            schema_version: self.schema_version,
            identity: self.identity,
            recovery_request: self.recovery_request,
            source: self.source,
            topology_digest: self.topology_digest,
            entity_counts: self.entity_counts,
            fields: self
                .fields
                .iter()
                .map(|field| {
                    Ok(RecoveredFieldArtifactEntry {
                        field: field.identity(),
                        definition: field.definition().clone(),
                        sample: field.sample(),
                        partitions: canonical_field_partition_descriptors(field)?,
                    })
                })
                .collect::<Result<Vec<_>, RecoveredFieldError>>()?,
        };
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn from_manifest_and_fields(
        manifest: RecoveredFieldArtifactManifest,
        fields: Vec<RecoveredField>,
    ) -> Result<Self, RecoveredFieldError> {
        manifest.validate()?;
        let expected_entries = fields
            .iter()
            .map(|field| {
                Ok(RecoveredFieldArtifactEntry {
                    field: field.identity(),
                    definition: field.definition().clone(),
                    sample: field.sample(),
                    partitions: canonical_field_partition_descriptors(field)?,
                })
            })
            .collect::<Result<Vec<_>, RecoveredFieldError>>()?;
        if manifest.fields != expected_entries {
            return Err(RecoveredFieldError::IdentityMismatch);
        }
        let artifact = Self {
            schema_version: manifest.schema_version,
            identity: manifest.identity,
            recovery_request: manifest.recovery_request,
            source: manifest.source,
            topology_digest: manifest.topology_digest,
            entity_counts: manifest.entity_counts,
            fields,
        };
        artifact.validate()?;
        Ok(artifact)
    }

    pub fn validate(&self) -> Result<(), RecoveredFieldError> {
        if self.schema_version != 1 {
            return Err(RecoveredFieldError::InvalidArtifact(
                "artifact requires a supported schema and nonempty topology".into(),
            ));
        }
        self.entity_counts.validate()?;
        if self.fields.is_empty() {
            return Err(RecoveredFieldError::InvalidArtifact(
                "artifact contains no recovered fields".into(),
            ));
        }
        let mut sample_keys = std::collections::BTreeSet::new();
        for field in &self.fields {
            field.validate()?;
            let expected_entities = self
                .entity_counts
                .entity_count(field.definition().entity_domain());
            if expected_entities == 0 {
                return Err(RecoveredFieldError::ArtifactTopologyMismatch);
            }
            let expected_shape = field.definition().storage_shape(expected_entities)?;
            if field.storage().shape != expected_shape {
                return Err(RecoveredFieldError::StorageShapeMismatch);
            }
            let sample_key = sample_identity(field.definition().identity(), field.sample())?;
            if !sample_keys.insert(sample_key) {
                return Err(RecoveredFieldError::CatalogDuplicateSample);
            }
        }
        if self.identity != self.derive_identity()? {
            return Err(RecoveredFieldError::IdentityMismatch);
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, RecoveredFieldError> {
        self.validate()?;
        encode_contract(FIELD_PREFIX, "recovered-field-artifact", self, FIELD_LIMITS)
            .map_err(|error| RecoveredFieldError::Encoding(error.to_string()))
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, RecoveredFieldError> {
        decode_contract(
            FIELD_PREFIX,
            "recovered-field-artifact",
            bytes,
            FIELD_LIMITS,
        )
        .map_err(|error| RecoveredFieldError::Encoding(error.to_string()))
    }

    fn derive_identity(&self) -> Result<RecoveredFieldArtifactIdentity, RecoveredFieldError> {
        derive_artifact_identity(&RecoveredFieldArtifactManifest {
            schema_version: self.schema_version,
            identity: self.identity,
            recovery_request: self.recovery_request,
            source: self.source,
            topology_digest: self.topology_digest,
            entity_counts: self.entity_counts,
            fields: self
                .fields
                .iter()
                .map(|field| {
                    Ok(RecoveredFieldArtifactEntry {
                        field: field.identity(),
                        definition: field.definition().clone(),
                        sample: field.sample(),
                        partitions: canonical_field_partition_descriptors(field)?,
                    })
                })
                .collect::<Result<Vec<_>, RecoveredFieldError>>()?,
        })
    }
}

fn derive_artifact_identity(
    manifest: &RecoveredFieldArtifactManifest,
) -> Result<RecoveredFieldArtifactIdentity, RecoveredFieldError> {
    encode_contract(
        FIELD_PREFIX,
        "recovered-field-artifact-identity",
        &(
            manifest.schema_version,
            manifest.recovery_request,
            manifest.source,
            manifest.topology_digest,
            manifest.entity_counts,
            &manifest.fields,
        ),
        FIELD_LIMITS,
    )
    .map(Digest::sha256)
    .map(RecoveredFieldArtifactIdentity)
    .map_err(|error| RecoveredFieldError::Encoding(error.to_string()))
}

fn sample_identity(
    definition: RecoveredFieldDefinitionIdentity,
    sample: RecoveredFieldSample,
) -> Result<Digest, RecoveredFieldError> {
    encode_contract(
        FIELD_PREFIX,
        "recovered-field-sample",
        &(definition, sample),
        FIELD_LIMITS,
    )
    .map(Digest::sha256)
    .map_err(|error| RecoveredFieldError::Encoding(error.to_string()))
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum RecoveredFieldError {
    #[error("recovered field identifiers must be nonempty")]
    EmptyIdentifier,
    #[error("recovered field quantity and value shape disagree")]
    QuantityShapeMismatch,
    #[error("recovered field sample coordinate is invalid")]
    InvalidSample,
    #[error("recovered field definition identity does not converge")]
    IdentityMismatch,
    #[error("recovered field storage identifier differs from its definition")]
    StorageIdentifierMismatch,
    #[error("recovered field storage shape is invalid")]
    StorageShapeMismatch,
    #[error("recovered field catalog contains a duplicate semantic sample")]
    CatalogDuplicateSample,
    #[error("recovered field location is not supported by the current spatial evaluator")]
    CatalogUnsupportedLocation,
    #[error("recovered field artifact is invalid: {0}")]
    InvalidArtifact(String),
    #[error("recovered field artifact topology differs from its analysis projection")]
    ArtifactTopologyMismatch,
    #[error("recovered field definition identity encoding failed: {0}")]
    Encoding(String),
}
