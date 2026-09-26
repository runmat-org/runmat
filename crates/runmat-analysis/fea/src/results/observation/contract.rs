use runmat_analysis_core::{
    AnalysisField, ComplexRepresentation, FieldLocation, IndependentDomain, ObservationComponent,
    ObservationCostEstimate, ObservationId, ObservationQuantity, ObservationSpec,
    ObservationValueShape, PartialAccessPolicy, ProcedureStepId, SamplingPolicy, SignalPrecision,
    SpatialReduction,
};
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use runmat_meshing_core::StableDigest;
use serde::{Deserialize, Serialize};

use super::distributed::{
    ObservationPartitionCombination, ObservationPartitionEvidence,
    ObservationPartitionEvidenceIdentity,
};
use super::{
    ObservationAccelerationSummary, ObservationFieldSelectionIdentity, ObservationFieldSource,
};
use crate::execution::{AnalysisExecutionError, AnalysisExecutionResult};
use crate::execution::{ObservationEvaluationRequestIdentity, StepIdentity};
use crate::results::field::{
    RecoveredFieldDefinitionIdentity, RecoveredFieldEntityDomain, RecoveredFieldSample,
};

pub(super) const OBSERVATION_PREFIX: &[u8] = b"runmat-fea-observation/v1\0";
// Artifact provenance was replaced, not made backward compatible. Plan/sample codecs remain
// unchanged because their numerical contracts did not change.
const OBSERVATION_ARTIFACT_PREFIX: &[u8] = b"runmat-fea-observation-artifact/v2\0";
pub(super) const OBSERVATION_LIMITS: CanonicalLimits =
    CanonicalLimits::new(1024 * 1024 * 1024, 4_000_000, 65_536, 64);

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ResolvedObservationPlanIdentity(pub(super) Digest);

impl ResolvedObservationPlanIdentity {
    pub const fn digest(self) -> Digest {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ObservationResultIdentity(pub(super) Digest);

impl ObservationResultIdentity {
    pub const fn digest(self) -> Digest {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ObservationArtifactIdentity(pub(super) Digest);

impl ObservationArtifactIdentity {
    pub const fn digest(self) -> Digest {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationLimits {
    pub maximum_observations: usize,
    pub maximum_selected_entities: usize,
    pub maximum_values: usize,
    pub maximum_operation_count: u64,
    pub maximum_peak_resident_bytes: u64,
    pub maximum_stored_bytes: u64,
    pub maximum_transfer_bytes: u64,
}

impl Default for ObservationLimits {
    fn default() -> Self {
        Self {
            maximum_observations: 4096,
            maximum_selected_entities: 100_000_000,
            maximum_values: 400_000_000,
            maximum_operation_count: 10_000_000_000,
            maximum_peak_resident_bytes: 32 * 1024 * 1024 * 1024,
            maximum_stored_bytes: 32 * 1024 * 1024 * 1024,
            maximum_transfer_bytes: 32 * 1024 * 1024 * 1024,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationEntity {
    Node(StableDigest),
    Element(StableDigest),
    IntegrationPoint(StableDigest),
    Connector(StableDigest),
    Point {
        identity: StableDigest,
        ordinal: u64,
        position_m: [f64; 3],
        source_element: StableDigest,
    },
}

pub(super) const fn observation_element_entity(
    domain: RecoveredFieldEntityDomain,
    identity: StableDigest,
) -> ObservationEntity {
    match domain {
        RecoveredFieldEntityDomain::SolidIntegrationPoints => {
            ObservationEntity::IntegrationPoint(identity)
        }
        _ => ObservationEntity::Element(identity),
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResolvedObservationTarget {
    Nodes {
        indices: Vec<usize>,
        identities: Vec<StableDigest>,
    },
    Elements {
        domain: RecoveredFieldEntityDomain,
        indices: Vec<usize>,
        identities: Vec<StableDigest>,
    },
    Points {
        samples: Vec<ResolvedObservationPoint>,
    },
    Volume {
        element_indices: Vec<usize>,
        element_identities: Vec<StableDigest>,
    },
    Boundary {
        face_indices: Vec<usize>,
        face_identities: Vec<StableDigest>,
    },
    Connectors {
        indices: Vec<usize>,
        identities: Vec<StableDigest>,
    },
    ConstraintGroups {
        indices: Vec<usize>,
        identities: Vec<StableDigest>,
    },
}

impl ResolvedObservationTarget {
    pub(crate) fn selected_entity_count(&self) -> usize {
        match self {
            Self::Nodes { indices, .. } | Self::Elements { indices, .. } => indices.len(),
            Self::Points { samples } => samples.len(),
            Self::Volume {
                element_indices, ..
            } => element_indices.len(),
            Self::Boundary { face_indices, .. } => face_indices.len(),
            Self::Connectors { indices, .. } => indices.len(),
            Self::ConstraintGroups { indices, .. } => indices.len(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResolvedObservationPoint {
    pub identity: StableDigest,
    pub ordinal: u64,
    pub position_m: [f64; 3],
    pub source_element_index: usize,
    pub source_element_identity: StableDigest,
    pub source_node_indices: Vec<usize>,
    pub source_node_identities: Vec<StableDigest>,
    pub shape_values: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResolvedObservation {
    pub spec: ObservationSpec,
    pub source_field_definition_identity: RecoveredFieldDefinitionIdentity,
    pub source_entity_domain: RecoveredFieldEntityDomain,
    pub target: ResolvedObservationTarget,
    pub component: ResolvedComponentProjection,
    pub result_value_shape: ObservationValueShape,
    pub sample_bounds: ObservationSampleBounds,
    pub resolved_cost: ObservationCostEstimate,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationSampleBounds {
    pub maximum_produced_samples: usize,
    pub maximum_retained_samples: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResolvedComponentProjection {
    All,
    Cartesian { axis: usize },
    Magnitude3,
    VonMisesSymmetricTensor,
    MaximumPrincipalSymmetricTensor,
    MinimumPrincipalSymmetricTensor,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "ResolvedObservationPlanWire")]
pub struct ResolvedObservationPlan {
    pub schema_version: u16,
    pub identity: ResolvedObservationPlanIdentity,
    pub topology_digest: Digest,
    pub step_id: ProcedureStepId,
    pub independent_domain: IndependentDomain,
    pub observations: Vec<ResolvedObservation>,
}

impl ResolvedObservationPlan {
    pub(super) fn derive_identity(
        &self,
    ) -> Result<ResolvedObservationPlanIdentity, ObservationError> {
        let bytes = encode_contract(
            OBSERVATION_PREFIX,
            "resolved-observation-plan",
            &(
                self.schema_version,
                self.topology_digest,
                &self.step_id,
                &self.independent_domain,
                &self.observations,
            ),
            OBSERVATION_LIMITS,
        )?;
        Ok(ResolvedObservationPlanIdentity(Digest::sha256(bytes)))
    }

    pub fn validate_identity(&self) -> Result<(), ObservationError> {
        if self.schema_version != 1 || self.identity != self.derive_identity()? {
            return Err(ObservationError::Identity(
                "observation plan identity does not converge",
            ));
        }
        let mut observation_ids = std::collections::BTreeSet::new();
        for observation in &self.observations {
            if !observation_ids.insert(observation.spec.observation_id.0.as_str()) {
                return Err(ObservationError::Invalid(
                    "resolved observation identifiers must be unique".into(),
                ));
            }
            if !runmat_analysis_core::observation_sampling_matches_domain(
                &self.independent_domain,
                &observation.spec.sampling,
            ) {
                return Err(ObservationError::Invalid(
                    "resolved observation sampling differs from the step domain".into(),
                ));
            }
            validate_resolved_observation(self.topology_digest, observation)?;
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, ObservationError> {
        self.validate_identity()?;
        Ok(encode_contract(
            OBSERVATION_PREFIX,
            "resolved-observation-plan",
            self,
            OBSERVATION_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, ObservationError> {
        let plan = decode_contract(
            OBSERVATION_PREFIX,
            "resolved-observation-plan",
            bytes,
            OBSERVATION_LIMITS,
        )?;
        Ok(plan)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ResolvedObservationPlanWire {
    schema_version: u16,
    identity: ResolvedObservationPlanIdentity,
    topology_digest: Digest,
    step_id: ProcedureStepId,
    independent_domain: IndependentDomain,
    observations: Vec<ResolvedObservation>,
}

impl TryFrom<ResolvedObservationPlanWire> for ResolvedObservationPlan {
    type Error = ObservationError;

    fn try_from(wire: ResolvedObservationPlanWire) -> Result<Self, Self::Error> {
        let plan = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            topology_digest: wire.topology_digest,
            step_id: wire.step_id,
            independent_domain: wire.independent_domain,
            observations: wire.observations,
        };
        plan.validate_identity()?;
        Ok(plan)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "ObservationResultWire")]
pub struct ObservationResult {
    pub schema_version: u16,
    pub logical_identity: ObservationResultIdentity,
    pub plan_identity: ResolvedObservationPlanIdentity,
    pub topology_digest: Digest,
    pub observation_id: ObservationId,
    pub sample: RecoveredFieldSample,
    pub quantity: ObservationQuantity,
    pub location: FieldLocation,
    pub value_shape: ObservationValueShape,
    pub complex_representation: ComplexRepresentation,
    pub entities: Vec<ObservationEntity>,
    pub field: AnalysisField,
    pub partial_access: PartialAccessPolicy,
    pub partition_evidence: ObservationPartitionEvidence,
}

impl ObservationResult {
    pub(super) fn derive_identity(&self) -> Result<ObservationResultIdentity, ObservationError> {
        let bytes = encode_contract(
            OBSERVATION_PREFIX,
            "observation-result-identity",
            &(
                self.schema_version,
                self.plan_identity,
                self.topology_digest,
                &self.observation_id,
                self.sample,
                self.quantity,
                self.location,
                &self.value_shape,
                self.complex_representation,
                &self.entities,
                &self.field,
                self.partial_access,
            ),
            OBSERVATION_LIMITS,
        )?;
        Ok(ObservationResultIdentity(Digest::sha256(bytes)))
    }

    pub fn validate(&self) -> Result<(), ObservationError> {
        let values = self
            .field
            .as_host_f64()
            .ok_or(ObservationError::Unsupported(
                "observation publication requires host-resident f64 values",
            ))?;
        let expected = self
            .field
            .shape
            .iter()
            .try_fold(1usize, |count, extent| count.checked_mul(*extent));
        let entity_shape_is_valid = self.field.shape
            == observation_result_storage_shape(
                (!self.entities.is_empty()).then_some(self.entities.len()),
                &self.value_shape,
                self.complex_representation,
            )?;
        let mut entity_ids = std::collections::BTreeSet::new();
        let entities_are_valid = self.entities.iter().all(|entity| {
            let (location, identity) = match entity {
                ObservationEntity::Node(identity) => (FieldLocation::Node, identity),
                ObservationEntity::Element(identity) => (FieldLocation::Element, identity),
                ObservationEntity::IntegrationPoint(identity) => {
                    (FieldLocation::IntegrationPoint, identity)
                }
                ObservationEntity::Connector(identity) => (FieldLocation::Connector, identity),
                ObservationEntity::Point {
                    identity,
                    position_m,
                    source_element,
                    ..
                } if position_m.iter().all(|value| value.is_finite())
                    && *source_element != StableDigest::ZERO =>
                {
                    (FieldLocation::Point, identity)
                }
                ObservationEntity::Point { .. } => return false,
            };
            location == self.location && entity_ids.insert(*identity)
        });
        let evidence_shape_is_valid = match self.partition_evidence.combination {
            ObservationPartitionCombination::CanonicalConcatenation => !self.entities.is_empty(),
            ObservationPartitionCombination::CanonicalCompensatedReduction { collective_count } => {
                self.entities.is_empty()
                    && collective_count as usize
                        == self
                            .partition_evidence
                            .contributions
                            .len()
                            .saturating_sub(1)
            }
            ObservationPartitionCombination::CanonicalExtremaReduction { collective_count } => {
                !self.entities.is_empty()
                    && collective_count as usize
                        == self
                            .partition_evidence
                            .contributions
                            .len()
                            .saturating_sub(1)
            }
        };
        self.partition_evidence.validate()?;
        if self.schema_version != 1
            || self.field.field_id != self.observation_id.0
            || expected != Some(values.len())
            || values.is_empty()
            || values.iter().any(|value| !value.is_finite())
            || !entity_shape_is_valid
            || !entities_are_valid
            || !evidence_shape_is_valid
            || self.partition_evidence.sample != self.sample
            || self.logical_identity != self.derive_identity()?
        {
            return Err(ObservationError::Invalid(
                "observation result metadata, values, or identity is inconsistent".into(),
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, ObservationError> {
        self.validate()?;
        Ok(encode_contract(
            OBSERVATION_PREFIX,
            "observation-result",
            self,
            OBSERVATION_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, ObservationError> {
        let result: Self = decode_contract(
            OBSERVATION_PREFIX,
            "observation-result",
            bytes,
            OBSERVATION_LIMITS,
        )?;
        result.validate()?;
        Ok(result)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ObservationResultWire {
    schema_version: u16,
    logical_identity: ObservationResultIdentity,
    plan_identity: ResolvedObservationPlanIdentity,
    topology_digest: Digest,
    observation_id: ObservationId,
    sample: RecoveredFieldSample,
    quantity: ObservationQuantity,
    location: FieldLocation,
    value_shape: ObservationValueShape,
    complex_representation: ComplexRepresentation,
    entities: Vec<ObservationEntity>,
    field: AnalysisField,
    partial_access: PartialAccessPolicy,
    partition_evidence: ObservationPartitionEvidence,
}

impl TryFrom<ObservationResultWire> for ObservationResult {
    type Error = ObservationError;

    fn try_from(wire: ObservationResultWire) -> Result<Self, Self::Error> {
        let result = Self {
            schema_version: wire.schema_version,
            logical_identity: wire.logical_identity,
            plan_identity: wire.plan_identity,
            topology_digest: wire.topology_digest,
            observation_id: wire.observation_id,
            sample: wire.sample,
            quantity: wire.quantity,
            location: wire.location,
            value_shape: wire.value_shape,
            complex_representation: wire.complex_representation,
            entities: wire.entities,
            field: wire.field,
            partial_access: wire.partial_access,
            partition_evidence: wire.partition_evidence,
        };
        result.validate()?;
        Ok(result)
    }
}

/// Canonical observation outputs produced from one recovered-field artifact.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "ObservationArtifactWire")]
pub struct ObservationArtifact {
    pub schema_version: u16,
    identity: ObservationArtifactIdentity,
    pub evaluation_request: ObservationEvaluationRequestIdentity,
    pub producer: StepIdentity,
    pub source_selection: ObservationFieldSelectionIdentity,
    pub source: ObservationFieldSource,
    pub topology_digest: Digest,
    pub plan_identity: ResolvedObservationPlanIdentity,
    results: Vec<ObservationResult>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationArtifactEntry {
    pub result: ObservationResultIdentity,
    pub partition_evidence: ObservationPartitionEvidenceIdentity,
    pub acceleration: Option<ObservationAccelerationSummary>,
    pub observation_id: ObservationId,
    pub sample: RecoveredFieldSample,
    pub quantity: ObservationQuantity,
    pub location: FieldLocation,
    pub value_shape: ObservationValueShape,
    pub partial_access: PartialAccessPolicy,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "ObservationArtifactManifestWire")]
pub struct ObservationArtifactManifest {
    pub schema_version: u16,
    pub identity: ObservationArtifactIdentity,
    pub evaluation_request: ObservationEvaluationRequestIdentity,
    pub producer: StepIdentity,
    pub source_selection: ObservationFieldSelectionIdentity,
    pub source: ObservationFieldSource,
    pub topology_digest: Digest,
    pub plan_identity: ResolvedObservationPlanIdentity,
    pub results: Vec<ObservationArtifactEntry>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ObservationArtifactManifestWire {
    schema_version: u16,
    identity: ObservationArtifactIdentity,
    evaluation_request: ObservationEvaluationRequestIdentity,
    producer: StepIdentity,
    source_selection: ObservationFieldSelectionIdentity,
    source: ObservationFieldSource,
    topology_digest: Digest,
    plan_identity: ResolvedObservationPlanIdentity,
    results: Vec<ObservationArtifactEntry>,
}

impl TryFrom<ObservationArtifactManifestWire> for ObservationArtifactManifest {
    type Error = ObservationError;

    fn try_from(wire: ObservationArtifactManifestWire) -> Result<Self, Self::Error> {
        let manifest = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            evaluation_request: wire.evaluation_request,
            producer: wire.producer,
            source_selection: wire.source_selection,
            source: wire.source,
            topology_digest: wire.topology_digest,
            plan_identity: wire.plan_identity,
            results: wire.results,
        };
        manifest.validate()?;
        Ok(manifest)
    }
}

impl ObservationArtifactManifest {
    pub fn validate(&self) -> Result<(), ObservationError> {
        if self.schema_version != 2
            || self.results.is_empty()
            || self.topology_digest != self.source.observation_topology_digest()
        {
            return Err(ObservationError::Invalid(
                "observation manifest requires a supported schema and results".into(),
            ));
        }
        let mut result_ids = std::collections::BTreeSet::new();
        let mut observation_ids = std::collections::BTreeSet::new();
        let mut acceleration_evidence = std::collections::BTreeSet::new();
        for entry in &self.results {
            if !result_ids.insert(entry.result) || !observation_ids.insert(&entry.observation_id) {
                return Err(ObservationError::Identity(
                    "observation manifest contains duplicate results",
                ));
            }
            if let Some(summary) = &entry.acceleration {
                summary.validate()?;
                if summary
                    .operations
                    .iter()
                    .any(|operation| !acceleration_evidence.insert(operation.evidence))
                {
                    return Err(ObservationError::Identity(
                        "observation manifest reuses acceleration evidence",
                    ));
                }
            }
        }
        if self.identity != derive_artifact_identity(self)? {
            return Err(ObservationError::Identity(
                "observation manifest identity does not converge",
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, ObservationError> {
        self.validate()?;
        Ok(encode_contract(
            OBSERVATION_ARTIFACT_PREFIX,
            "observation-artifact-manifest",
            self,
            OBSERVATION_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, ObservationError> {
        let manifest: Self = decode_contract(
            OBSERVATION_ARTIFACT_PREFIX,
            "observation-artifact-manifest",
            bytes,
            OBSERVATION_LIMITS,
        )?;
        manifest.validate()?;
        Ok(manifest)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ObservationArtifactWire {
    schema_version: u16,
    identity: ObservationArtifactIdentity,
    evaluation_request: ObservationEvaluationRequestIdentity,
    producer: StepIdentity,
    source_selection: ObservationFieldSelectionIdentity,
    source: ObservationFieldSource,
    topology_digest: Digest,
    plan_identity: ResolvedObservationPlanIdentity,
    results: Vec<ObservationResult>,
}

impl TryFrom<ObservationArtifactWire> for ObservationArtifact {
    type Error = ObservationError;

    fn try_from(wire: ObservationArtifactWire) -> Result<Self, Self::Error> {
        let artifact = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            evaluation_request: wire.evaluation_request,
            producer: wire.producer,
            source_selection: wire.source_selection,
            source: wire.source,
            topology_digest: wire.topology_digest,
            plan_identity: wire.plan_identity,
            results: wire.results,
        };
        artifact.validate()?;
        Ok(artifact)
    }
}

pub struct ObservationArtifactInput {
    pub evaluation_request: ObservationEvaluationRequestIdentity,
    pub producer: StepIdentity,
    pub source_selection: ObservationFieldSelectionIdentity,
    pub source: ObservationFieldSource,
    pub topology_digest: Digest,
    pub plan_identity: ResolvedObservationPlanIdentity,
    pub results: Vec<ObservationResult>,
}

impl ObservationArtifact {
    pub fn new(input: ObservationArtifactInput) -> Result<Self, ObservationError> {
        let mut artifact = Self {
            schema_version: 2,
            identity: ObservationArtifactIdentity(Digest::sha256([])),
            evaluation_request: input.evaluation_request,
            producer: input.producer,
            source_selection: input.source_selection,
            source: input.source,
            topology_digest: input.topology_digest,
            plan_identity: input.plan_identity,
            results: input.results,
        };
        artifact.identity = artifact.derive_identity()?;
        artifact.validate()?;
        Ok(artifact)
    }

    pub const fn identity(&self) -> ObservationArtifactIdentity {
        self.identity
    }

    pub fn results(&self) -> &[ObservationResult] {
        &self.results
    }

    pub fn manifest(&self) -> Result<ObservationArtifactManifest, ObservationError> {
        let manifest = ObservationArtifactManifest {
            schema_version: self.schema_version,
            identity: self.identity,
            evaluation_request: self.evaluation_request,
            producer: self.producer,
            source_selection: self.source_selection,
            source: self.source,
            topology_digest: self.topology_digest,
            plan_identity: self.plan_identity,
            results: self
                .results
                .iter()
                .map(observation_artifact_entry)
                .collect::<Result<Vec<_>, _>>()?,
        };
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn from_manifest_and_results(
        manifest: ObservationArtifactManifest,
        results: Vec<ObservationResult>,
    ) -> Result<Self, ObservationError> {
        manifest.validate()?;
        if manifest.results.len() != results.len() {
            return Err(ObservationError::Identity(
                "observation results differ from their manifest",
            ));
        }
        for (entry, result) in manifest.results.iter().zip(&results) {
            let acceleration = ObservationAccelerationSummary::from_evidence(
                &result.partition_evidence.accelerations,
            )?;
            if entry.result != result.logical_identity
                || entry.partition_evidence != result.partition_evidence.identity
                || entry.acceleration != acceleration
                || entry.observation_id != result.observation_id
                || entry.sample != result.sample
                || entry.quantity != result.quantity
                || entry.location != result.location
                || entry.value_shape != result.value_shape
                || entry.partial_access != result.partial_access
            {
                return Err(ObservationError::Identity(
                    "observation results differ from their manifest",
                ));
            }
        }
        let artifact = Self {
            schema_version: manifest.schema_version,
            identity: manifest.identity,
            evaluation_request: manifest.evaluation_request,
            producer: manifest.producer,
            source_selection: manifest.source_selection,
            source: manifest.source,
            topology_digest: manifest.topology_digest,
            plan_identity: manifest.plan_identity,
            results,
        };
        artifact.validate()?;
        Ok(artifact)
    }

    pub fn validate(&self) -> Result<(), ObservationError> {
        if self.schema_version != 2
            || self.results.is_empty()
            || self.topology_digest != self.source.observation_topology_digest()
        {
            return Err(ObservationError::Invalid(
                "observation artifact requires a supported schema and results".into(),
            ));
        }
        let mut result_ids = std::collections::BTreeSet::new();
        let mut observation_ids = std::collections::BTreeSet::new();
        for result in &self.results {
            result.validate()?;
            if result.plan_identity != self.plan_identity
                || result.topology_digest != self.topology_digest
                || !result_ids.insert(result.logical_identity)
                || !observation_ids.insert(&result.observation_id)
            {
                return Err(ObservationError::Identity(
                    "observation artifact contains an unrelated or duplicate result",
                ));
            }
        }
        if self.identity != self.derive_identity()? {
            return Err(ObservationError::Identity(
                "observation artifact identity does not converge",
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, ObservationError> {
        self.validate()?;
        Ok(encode_contract(
            OBSERVATION_ARTIFACT_PREFIX,
            "observation-artifact",
            self,
            OBSERVATION_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, ObservationError> {
        let artifact = decode_contract(
            OBSERVATION_ARTIFACT_PREFIX,
            "observation-artifact",
            bytes,
            OBSERVATION_LIMITS,
        )?;
        Ok(artifact)
    }

    fn derive_identity(&self) -> Result<ObservationArtifactIdentity, ObservationError> {
        derive_artifact_identity(&ObservationArtifactManifest {
            schema_version: self.schema_version,
            identity: self.identity,
            evaluation_request: self.evaluation_request,
            producer: self.producer,
            source_selection: self.source_selection,
            source: self.source,
            topology_digest: self.topology_digest,
            plan_identity: self.plan_identity,
            results: self
                .results
                .iter()
                .map(observation_artifact_entry)
                .collect::<Result<Vec<_>, _>>()?,
        })
    }
}

fn observation_artifact_entry(
    result: &ObservationResult,
) -> Result<ObservationArtifactEntry, ObservationError> {
    Ok(ObservationArtifactEntry {
        result: result.logical_identity,
        partition_evidence: result.partition_evidence.identity,
        acceleration: ObservationAccelerationSummary::from_evidence(
            &result.partition_evidence.accelerations,
        )?,
        observation_id: result.observation_id.clone(),
        sample: result.sample,
        quantity: result.quantity,
        location: result.location,
        value_shape: result.value_shape.clone(),
        partial_access: result.partial_access,
    })
}

fn derive_artifact_identity(
    manifest: &ObservationArtifactManifest,
) -> Result<ObservationArtifactIdentity, ObservationError> {
    let bytes = encode_contract(
        OBSERVATION_ARTIFACT_PREFIX,
        "observation-artifact-identity",
        &(
            manifest.schema_version,
            manifest.evaluation_request,
            manifest.producer,
            manifest.source_selection,
            manifest.source,
            manifest.topology_digest,
            manifest.plan_identity,
            &manifest.results,
        ),
        OBSERVATION_LIMITS,
    )?;
    Ok(ObservationArtifactIdentity(Digest::sha256(bytes)))
}

fn validate_resolved_observation(
    topology_digest: Digest,
    observation: &ResolvedObservation,
) -> Result<(), ObservationError> {
    validate_output_numeric_contract(&observation.spec)?;
    if observation.sample_bounds.maximum_produced_samples == 0
        || observation.sample_bounds.maximum_retained_samples == 0
        || observation.sample_bounds.maximum_retained_samples
            > observation.sample_bounds.maximum_produced_samples
    {
        return Err(ObservationError::Invalid(
            "resolved observation sample bounds are invalid".into(),
        ));
    }
    let expected_sample_bounds = if observation.spec.sampling == SamplingPolicy::Final {
        ObservationSampleBounds {
            maximum_produced_samples: 1,
            maximum_retained_samples: 1,
        }
    } else {
        super::history::resolve_history_sample_bounds(
            &observation.spec.sampling,
            &observation.spec.storage.retention,
            observation.sample_bounds.maximum_produced_samples,
        )?
    };
    if observation.sample_bounds != expected_sample_bounds {
        return Err(ObservationError::Identity(
            "resolved observation sample bounds do not converge",
        ));
    }
    let source_components = shape_component_count(&observation.spec.value_shape)?;
    let spatial_extrema = observation
        .spec
        .reduction
        .as_ref()
        .is_some_and(|reduction| {
            matches!(
                reduction.spatial,
                SpatialReduction::Minimum | SpatialReduction::Maximum
            )
        });
    let has_spatial_reduction = observation
        .spec
        .reduction
        .as_ref()
        .is_some_and(|reduction| reduction.spatial != SpatialReduction::None);
    let has_discrete_reaction_reduction =
        observation
            .spec
            .reduction
            .as_ref()
            .is_some_and(|reduction| {
                matches!(
                    reduction.spatial,
                    SpatialReduction::Resultant | SpatialReduction::Moment { .. }
                ) && matches!(
                    observation.spec.quantity,
                    runmat_analysis_core::ObservationQuantity::ReactionForce
                        | runmat_analysis_core::ObservationQuantity::ReactionMoment
                )
            });
    let component_projection_is_valid = match observation.component {
        ResolvedComponentProjection::All => true,
        ResolvedComponentProjection::Cartesian { axis } => axis < source_components,
        ResolvedComponentProjection::Magnitude3 => source_components == 3,
        ResolvedComponentProjection::VonMisesSymmetricTensor
        | ResolvedComponentProjection::MaximumPrincipalSymmetricTensor
        | ResolvedComponentProjection::MinimumPrincipalSymmetricTensor => source_components == 6,
    };
    if !component_projection_is_valid
        || observation.result_value_shape != observation.spec.result_value_shape()
        || shape_component_count(&observation.result_value_shape)? == 0
    {
        return Err(ObservationError::Invalid(
            "resolved result value shape is inconsistent".into(),
        ));
    }
    match &observation.target {
        ResolvedObservationTarget::Nodes {
            indices,
            identities,
        } => {
            if materialized_field_location(&observation.spec) != FieldLocation::Node
                || observation.source_entity_domain != RecoveredFieldEntityDomain::StructuralNodes
                || (has_spatial_reduction && !spatial_extrema && !has_discrete_reaction_reduction)
                || indices.len() != identities.len()
                || indices.is_empty()
                || !strictly_increasing(indices)
                || !all_unique(identities)
            {
                return Err(ObservationError::Invalid(
                    "resolved node target is empty, unordered, or semantically inconsistent".into(),
                ));
            }
        }
        ResolvedObservationTarget::Elements {
            domain,
            indices,
            identities,
        } => {
            if !matches!(
                materialized_field_location(&observation.spec),
                FieldLocation::Element | FieldLocation::IntegrationPoint
            ) || (materialized_field_location(&observation.spec)
                == FieldLocation::IntegrationPoint)
                != (*domain == RecoveredFieldEntityDomain::SolidIntegrationPoints)
                || observation.source_entity_domain != *domain
                || *domain == RecoveredFieldEntityDomain::StructuralNodes
                || (has_spatial_reduction && !spatial_extrema)
                || indices.len() != identities.len()
                || indices.is_empty()
                || !strictly_increasing(indices)
                || !all_unique(identities)
            {
                return Err(ObservationError::Invalid(
                    "resolved element target is empty, unordered, or semantically inconsistent"
                        .into(),
                ));
            }
        }
        ResolvedObservationTarget::Points { samples } => {
            let valid_ordinals = samples
                .iter()
                .enumerate()
                .all(|(ordinal, sample)| sample.ordinal == ordinal as u64);
            let valid_samples = samples.iter().all(|sample| {
                sample.identity != StableDigest::ZERO
                    && !sample.source_node_indices.is_empty()
                    && sample.source_node_indices.len() == sample.source_node_identities.len()
                    && sample.source_node_indices.len() == sample.shape_values.len()
                    && sample.position_m.iter().all(|value| value.is_finite())
                    && sample.shape_values.iter().all(|value| value.is_finite())
                    && (sample.shape_values.iter().sum::<f64>() - 1.0).abs() <= 1.0e-9
                    && derive_resolved_point_identity(topology_digest, sample)
                        .is_ok_and(|identity| identity == sample.identity)
            });
            if materialized_field_location(&observation.spec) != FieldLocation::Node
                || observation.spec.output_location != FieldLocation::Point
                || observation.source_entity_domain != RecoveredFieldEntityDomain::StructuralNodes
                || observation.spec.recovery
                    != runmat_analysis_core::RecoveryPolicy::ShapeFunctionInterpolation
                || (has_spatial_reduction && !spatial_extrema)
                || samples.is_empty()
                || !valid_ordinals
                || !valid_samples
            {
                return Err(ObservationError::Invalid(
                    "resolved point samples are empty, unordered, or semantically inconsistent"
                        .into(),
                ));
            }
        }
        ResolvedObservationTarget::ConstraintGroups {
            indices,
            identities,
        } => {
            if observation.spec.source_location != FieldLocation::Constraint
                || observation.spec.output_location != FieldLocation::Global
                || observation.source_entity_domain != RecoveredFieldEntityDomain::ConstraintGroups
                || !has_discrete_reaction_reduction
                || indices.len() != 1
                || identities.len() != 1
                || identities[0] == StableDigest::ZERO
            {
                return Err(ObservationError::Invalid(
                    "resolved constraint target is not one canonical reaction-result group".into(),
                ));
            }
        }
        ResolvedObservationTarget::Connectors {
            indices,
            identities,
        } => {
            let reduction_is_valid = observation.spec.reduction.as_ref().is_none_or(|reduction| {
                matches!(
                    (observation.spec.quantity, reduction.spatial),
                    (
                        runmat_analysis_core::ObservationQuantity::ReactionForce,
                        SpatialReduction::Resultant
                    )
                )
            });
            if observation.spec.source_location != FieldLocation::Connector
                || observation.spec.output_location != FieldLocation::Connector
                || observation.source_entity_domain != RecoveredFieldEntityDomain::Connectors
                || !reduction_is_valid
                || indices.len() != 1
                || identities.len() != 1
                || identities[0] == StableDigest::ZERO
            {
                return Err(ObservationError::Invalid(
                    "resolved connector target is not one canonical connector".into(),
                ));
            }
        }
        ResolvedObservationTarget::Volume {
            element_indices,
            element_identities,
        } => {
            if observation.spec.reduction.is_none()
                || !matches!(
                    observation.source_entity_domain,
                    RecoveredFieldEntityDomain::StructuralNodes
                        | RecoveredFieldEntityDomain::SolidElements
                )
                || element_indices.is_empty()
                || !strictly_increasing(element_indices)
                || element_indices.len() != element_identities.len()
                || !all_unique(element_identities)
            {
                return Err(ObservationError::Invalid(
                    "resolved volume target is empty, unordered, or lacks a reduction".into(),
                ));
            }
        }
        ResolvedObservationTarget::Boundary {
            face_indices,
            face_identities,
        } => {
            if materialized_field_location(&observation.spec) != FieldLocation::Node
                || observation.source_entity_domain != RecoveredFieldEntityDomain::StructuralNodes
                || observation.spec.reduction.is_none()
                || face_indices.is_empty()
                || !strictly_increasing(face_indices)
                || face_indices.len() != face_identities.len()
                || !all_unique(face_identities)
            {
                return Err(ObservationError::Invalid(
                    "resolved boundary target is empty, unordered, or semantically inconsistent"
                        .into(),
                ));
            }
        }
    }
    if observation.resolved_cost.operation_count == 0
        || observation.resolved_cost.peak_resident_bytes == 0
        || observation.resolved_cost.stored_bytes == 0
        || observation.resolved_cost.transfer_bytes == 0
        || observation.resolved_cost.collective_count != 0
    {
        return Err(ObservationError::Invalid(
            "resolved observation cost is incomplete or unsupported".into(),
        ));
    }
    Ok(())
}

pub(super) fn materialized_field_location(spec: &ObservationSpec) -> FieldLocation {
    if spec.source_location == FieldLocation::Element
        && spec.output_location == FieldLocation::Node
        && spec.recovery == runmat_analysis_core::RecoveryPolicy::VolumeWeightedAverage
    {
        FieldLocation::Node
    } else {
        spec.source_location
    }
}

pub(super) fn derive_resolved_point_identity(
    topology_digest: Digest,
    sample: &ResolvedObservationPoint,
) -> Result<StableDigest, ObservationError> {
    let bytes = encode_contract(
        OBSERVATION_PREFIX,
        "resolved-observation-point",
        &(
            topology_digest,
            sample.ordinal,
            sample.position_m,
            sample.source_element_identity,
            &sample.source_node_indices,
            &sample.shape_values,
        ),
        OBSERVATION_LIMITS,
    )?;
    Ok(StableDigest::from_bytes(*Digest::sha256(bytes).bytes()))
}

pub(super) fn validate_output_numeric_contract(
    spec: &ObservationSpec,
) -> Result<(), ObservationError> {
    let precision_matches = match spec.complex_representation {
        ComplexRepresentation::Complex => spec.storage.precision == SignalPrecision::Complex128,
        _ => spec.storage.precision == SignalPrecision::Real64,
    };
    if !precision_matches {
        return Err(ObservationError::Invalid(
            "observation precision does not match its requested real or complex representation"
                .into(),
        ));
    }
    if spec.complex_representation == ComplexRepresentation::PhaseRadians
        && spec.reduction.is_some()
    {
        return Err(ObservationError::Unsupported(
            "phase requires a circular reduction rather than a linear spatial reduction",
        ));
    }
    if spec.complex_representation == ComplexRepresentation::Complex
        && spec
            .reduction
            .as_ref()
            .is_some_and(|reduction| reduction.spatial == SpatialReduction::RootMeanSquare)
    {
        return Err(ObservationError::Unsupported(
            "complex root-mean-square reduction requires an explicit magnitude representation",
        ));
    }
    if spec.component == ObservationComponent::Magnitude
        && matches!(
            spec.complex_representation,
            ComplexRepresentation::PhaseRadians | ComplexRepresentation::Complex
        )
    {
        return Err(ObservationError::Unsupported(
            "vector magnitude requires a real, imaginary, or complex-magnitude representation",
        ));
    }
    Ok(())
}

fn strictly_increasing(values: &[usize]) -> bool {
    values.windows(2).all(|pair| pair[0] < pair[1])
}

fn all_unique(values: &[StableDigest]) -> bool {
    values
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>()
        .len()
        == values.len()
}

#[derive(Debug, thiserror::Error)]
pub enum ObservationError {
    #[error("invalid observation contract: {0}")]
    Invalid(String),
    #[error("unsupported observation contract: {0}")]
    Unsupported(&'static str),
    #[error("observation `{observation_id}` cannot resolve target: {reason}")]
    Target {
        observation_id: String,
        reason: String,
    },
    #[error("observation `{observation_id}` cannot bind a recovered field: {reason}")]
    Field {
        observation_id: String,
        reason: String,
    },
    #[error("observation resource limit exceeded: {0}")]
    Limit(&'static str),
    #[error("observation identity mismatch: {0}")]
    Identity(&'static str),
    #[error("observation canonical encoding failed: {0}")]
    Encoding(#[from] runmat_canonical_codec::CanonicalCodecError),
    #[error("observation publication failed: {0}")]
    Publication(#[from] AnalysisExecutionError),
    #[error("analysis topology traversal failed: {0}")]
    Topology(#[from] crate::topology::AnalysisProjectionError),
}

pub(super) fn checked_bytes(values: usize) -> Result<u64, ObservationError> {
    u64::try_from(values)
        .ok()
        .and_then(|count| count.checked_mul(8))
        .ok_or(ObservationError::Limit("value byte count"))
}

pub(super) fn observation_result_storage_component_count(
    logical_component_count: usize,
    representation: ComplexRepresentation,
) -> Result<usize, ObservationError> {
    match representation {
        ComplexRepresentation::Complex => {
            logical_component_count
                .checked_mul(2)
                .ok_or(ObservationError::Limit(
                    "complex observation component count",
                ))
        }
        _ => Ok(logical_component_count),
    }
}

pub(super) fn observation_result_storage_shape(
    entity_count: Option<usize>,
    value_shape: &ObservationValueShape,
    representation: ComplexRepresentation,
) -> Result<Vec<usize>, ObservationError> {
    let logical_components = shape_component_count(value_shape)?;
    Ok(match (entity_count, logical_components, representation) {
        (Some(entities), 1, ComplexRepresentation::Complex) => vec![entities, 2],
        (Some(entities), components, ComplexRepresentation::Complex) => {
            vec![entities, components, 2]
        }
        (Some(entities), 1, _) => vec![entities],
        (Some(entities), components, _) => vec![entities, components],
        (None, _, ComplexRepresentation::Complex) => vec![logical_components, 2],
        (None, _, _) => vec![logical_components],
    })
}

pub(super) fn shape_component_count(
    shape: &ObservationValueShape,
) -> Result<usize, ObservationError> {
    shape.component_count().ok_or_else(|| {
        ObservationError::Invalid(
            "tensor observation shapes require positive nonempty extents".into(),
        )
    })
}

pub(super) fn publication_result<T>(
    value: AnalysisExecutionResult<T>,
) -> Result<T, ObservationError> {
    value.map_err(ObservationError::Publication)
}
