use std::ops::Deref;

use runmat_analysis_core::{
    AnalysisField, AnalysisFieldValues, FieldLocation, ObservationQuantity, ObservationValueShape,
};
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::execution::{FieldRecoveryRequestIdentity, SolverResultIdentity};

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
#[serde(rename_all = "snake_case")]
pub enum RecoveredFieldNumericRepresentation {
    Real,
    ComplexInterleaved,
    ComplexSplitReal,
    ComplexSplitImaginary,
    Magnitude,
    PhaseRadians,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "domain", content = "coordinate")]
pub enum RecoveredFieldSample {
    Final,
    TimeSeconds(f64),
    ModeIndex(usize),
    FrequencyHertz(f64),
    LoadFactor(f64),
}

impl RecoveredFieldSample {
    pub(super) fn validate(self) -> Result<(), RecoveredFieldError> {
        match self {
            Self::Final | Self::ModeIndex(_) => Ok(()),
            Self::TimeSeconds(value) | Self::FrequencyHertz(value) | Self::LoadFactor(value)
                if value.is_finite() && value.to_bits() != (-0.0_f64).to_bits() =>
            {
                Ok(())
            }
            _ => Err(RecoveredFieldError::InvalidSample),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "RecoveredFieldDefinitionData")]
pub struct RecoveredFieldDefinition {
    identity: RecoveredFieldDefinitionIdentity,
    field_id: String,
    quantity: ObservationQuantity,
    location: FieldLocation,
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
        value_shape: ObservationValueShape,
        numeric_representation: RecoveredFieldNumericRepresentation,
    ) -> Result<Self, RecoveredFieldError> {
        let mut definition = Self {
            identity: RecoveredFieldDefinitionIdentity(Digest::sha256([])),
            field_id: field_id.into(),
            quantity,
            location,
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

    pub fn value_shape(&self) -> &ObservationValueShape {
        &self.value_shape
    }

    pub const fn numeric_representation(&self) -> RecoveredFieldNumericRepresentation {
        self.numeric_representation
    }

    fn derive_identity(&self) -> Result<RecoveredFieldDefinitionIdentity, RecoveredFieldError> {
        encode_contract(
            FIELD_PREFIX,
            "recovered-field-definition",
            &(
                self.quantity,
                self.location,
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
    definition: RecoveredFieldDefinition,
    sample: RecoveredFieldSample,
    storage: AnalysisField,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RecoveredFieldData {
    definition: RecoveredFieldDefinition,
    sample: RecoveredFieldSample,
    storage: AnalysisField,
}

impl TryFrom<RecoveredFieldData> for RecoveredField {
    type Error = RecoveredFieldError;

    fn try_from(data: RecoveredFieldData) -> Result<Self, Self::Error> {
        Self::new(data.definition, data.sample, data.storage)
    }
}

impl RecoveredField {
    pub fn new(
        definition: RecoveredFieldDefinition,
        sample: RecoveredFieldSample,
        storage: AnalysisField,
    ) -> Result<Self, RecoveredFieldError> {
        let field = Self {
            definition,
            sample,
            storage,
        };
        field.validate()?;
        Ok(field)
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
        if expected != actual {
            return Err(RecoveredFieldError::StorageShapeMismatch);
        }
        Ok(())
    }
}

impl Deref for RecoveredField {
    type Target = AnalysisField;

    fn deref(&self) -> &Self::Target {
        &self.storage
    }
}

/// Canonical, source-bound recovered fields for one solver topology and result.
///
/// This artifact is the portable boundary between family-owned recovery and solver-neutral
/// observation evaluation. Its identity excludes chunking and placement.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "RecoveredFieldArtifactWire")]
pub struct RecoveredFieldArtifact {
    pub schema_version: u16,
    identity: RecoveredFieldArtifactIdentity,
    pub recovery_request: FieldRecoveryRequestIdentity,
    pub source_result: SolverResultIdentity,
    pub topology_digest: Digest,
    pub node_count: usize,
    pub element_count: usize,
    fields: Vec<RecoveredField>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RecoveredFieldArtifactWire {
    schema_version: u16,
    identity: RecoveredFieldArtifactIdentity,
    recovery_request: FieldRecoveryRequestIdentity,
    source_result: SolverResultIdentity,
    topology_digest: Digest,
    node_count: usize,
    element_count: usize,
    fields: Vec<RecoveredField>,
}

impl TryFrom<RecoveredFieldArtifactWire> for RecoveredFieldArtifact {
    type Error = RecoveredFieldError;

    fn try_from(wire: RecoveredFieldArtifactWire) -> Result<Self, Self::Error> {
        let artifact = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            recovery_request: wire.recovery_request,
            source_result: wire.source_result,
            topology_digest: wire.topology_digest,
            node_count: wire.node_count,
            element_count: wire.element_count,
            fields: wire.fields,
        };
        artifact.validate()?;
        Ok(artifact)
    }
}

impl RecoveredFieldArtifact {
    pub fn new(
        recovery_request: FieldRecoveryRequestIdentity,
        source_result: SolverResultIdentity,
        topology_digest: Digest,
        node_count: usize,
        element_count: usize,
        fields: Vec<RecoveredField>,
    ) -> Result<Self, RecoveredFieldError> {
        let mut artifact = Self {
            schema_version: 1,
            identity: RecoveredFieldArtifactIdentity(Digest::sha256([])),
            recovery_request,
            source_result,
            topology_digest,
            node_count,
            element_count,
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

    pub fn validate(&self) -> Result<(), RecoveredFieldError> {
        if self.schema_version != 1 || self.node_count == 0 || self.element_count == 0 {
            return Err(RecoveredFieldError::InvalidArtifact(
                "artifact requires a supported schema and nonempty topology".into(),
            ));
        }
        if self.fields.is_empty() {
            return Err(RecoveredFieldError::InvalidArtifact(
                "artifact contains no recovered fields".into(),
            ));
        }
        let mut sample_keys = std::collections::BTreeSet::new();
        for field in &self.fields {
            field.validate()?;
            let expected_entities = match field.definition().location() {
                FieldLocation::Node => self.node_count,
                FieldLocation::Element => self.element_count,
                _ => return Err(RecoveredFieldError::CatalogUnsupportedLocation),
            };
            let components = field
                .definition()
                .value_shape()
                .component_count()
                .ok_or(RecoveredFieldError::QuantityShapeMismatch)?;
            let expected_shape = if components == 1 {
                vec![expected_entities]
            } else {
                vec![expected_entities, components]
            };
            if field.storage().shape != expected_shape {
                return Err(RecoveredFieldError::StorageShapeMismatch);
            }
            let sample_key = encode_contract(
                FIELD_PREFIX,
                "recovered-field-sample",
                &(field.definition().identity(), field.sample()),
                FIELD_LIMITS,
            )
            .map(Digest::sha256)
            .map_err(|error| RecoveredFieldError::Encoding(error.to_string()))?;
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
        encode_contract(
            FIELD_PREFIX,
            "recovered-field-artifact-identity",
            &(
                self.schema_version,
                self.recovery_request,
                self.source_result,
                self.topology_digest,
                self.node_count,
                self.element_count,
                &self.fields,
            ),
            FIELD_LIMITS,
        )
        .map(Digest::sha256)
        .map(RecoveredFieldArtifactIdentity)
        .map_err(|error| RecoveredFieldError::Encoding(error.to_string()))
    }
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
