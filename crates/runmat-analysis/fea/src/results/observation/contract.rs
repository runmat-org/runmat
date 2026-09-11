use runmat_analysis_core::{
    AnalysisField, FieldLocation, IndependentDomain, ObservationCostEstimate, ObservationId,
    ObservationQuantity, ObservationSpec, ObservationValueShape, PartialAccessPolicy,
    ProcedureStepId, SamplingPolicy,
};
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use runmat_meshing_core::StableDigest;
use serde::{Deserialize, Serialize};

use crate::execution::{AnalysisExecutionError, AnalysisExecutionResult};
use crate::execution::{
    FieldRecoveryRequestIdentity, ObservationEvaluationRequestIdentity, SolverResultIdentity,
    StepIdentity,
};
use crate::results::field::{
    RecoveredFieldArtifactIdentity, RecoveredFieldDefinitionIdentity, RecoveredFieldEntityDomain,
};

pub(super) const OBSERVATION_PREFIX: &[u8] = b"runmat-fea-observation/v1\0";
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

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationEntity {
    Node(StableDigest),
    Element(StableDigest),
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
    Volume {
        element_indices: Vec<usize>,
    },
    Boundary {
        face_indices: Vec<usize>,
    },
}

impl ResolvedObservationTarget {
    pub(super) fn selected_entity_count(&self) -> usize {
        match self {
            Self::Nodes { indices, .. } | Self::Elements { indices, .. } => indices.len(),
            Self::Volume { element_indices } => element_indices.len(),
            Self::Boundary { face_indices } => face_indices.len(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResolvedObservation {
    pub spec: ObservationSpec,
    pub source_field_definition_identity: RecoveredFieldDefinitionIdentity,
    pub source_entity_domain: RecoveredFieldEntityDomain,
    pub target: ResolvedObservationTarget,
    pub component: ResolvedComponentProjection,
    pub result_component_count: usize,
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
            validate_resolved_observation(observation)?;
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
    pub quantity: ObservationQuantity,
    pub location: FieldLocation,
    pub value_shape: ObservationValueShape,
    pub entities: Vec<ObservationEntity>,
    pub field: AnalysisField,
    pub partial_access: PartialAccessPolicy,
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
                self.quantity,
                self.location,
                &self.value_shape,
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
        let components = shape_component_count(&self.value_shape)?;
        let entity_shape_is_valid = if self.entities.is_empty() {
            self.field.shape == vec![components]
        } else {
            let expected_shape = if components == 1 {
                vec![self.entities.len()]
            } else {
                vec![self.entities.len(), components]
            };
            self.field.shape == expected_shape
        };
        let mut entity_ids = std::collections::BTreeSet::new();
        let entities_are_valid = self.entities.iter().all(|entity| {
            let (location, identity) = match entity {
                ObservationEntity::Node(identity) => (FieldLocation::Node, identity),
                ObservationEntity::Element(identity) => (FieldLocation::Element, identity),
            };
            location == self.location && entity_ids.insert(*identity)
        });
        if self.schema_version != 1
            || self.field.field_id != self.observation_id.0
            || expected != Some(values.len())
            || values.is_empty()
            || values.iter().any(|value| !value.is_finite())
            || !entity_shape_is_valid
            || !entities_are_valid
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
    quantity: ObservationQuantity,
    location: FieldLocation,
    value_shape: ObservationValueShape,
    entities: Vec<ObservationEntity>,
    field: AnalysisField,
    partial_access: PartialAccessPolicy,
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
            quantity: wire.quantity,
            location: wire.location,
            value_shape: wire.value_shape,
            entities: wire.entities,
            field: wire.field,
            partial_access: wire.partial_access,
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
    pub field_recovery_request: FieldRecoveryRequestIdentity,
    pub source_fields: RecoveredFieldArtifactIdentity,
    pub source_result: SolverResultIdentity,
    pub topology_digest: Digest,
    pub plan_identity: ResolvedObservationPlanIdentity,
    results: Vec<ObservationResult>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationArtifactEntry {
    pub result: ObservationResultIdentity,
    pub observation_id: ObservationId,
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
    pub field_recovery_request: FieldRecoveryRequestIdentity,
    pub source_fields: RecoveredFieldArtifactIdentity,
    pub source_result: SolverResultIdentity,
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
    field_recovery_request: FieldRecoveryRequestIdentity,
    source_fields: RecoveredFieldArtifactIdentity,
    source_result: SolverResultIdentity,
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
            field_recovery_request: wire.field_recovery_request,
            source_fields: wire.source_fields,
            source_result: wire.source_result,
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
        if self.schema_version != 1 || self.results.is_empty() {
            return Err(ObservationError::Invalid(
                "observation manifest requires a supported schema and results".into(),
            ));
        }
        let mut result_ids = std::collections::BTreeSet::new();
        let mut observation_ids = std::collections::BTreeSet::new();
        if self.results.iter().any(|entry| {
            !result_ids.insert(entry.result) || !observation_ids.insert(&entry.observation_id)
        }) {
            return Err(ObservationError::Identity(
                "observation manifest contains duplicate results",
            ));
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
            OBSERVATION_PREFIX,
            "observation-artifact-manifest",
            self,
            OBSERVATION_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, ObservationError> {
        let manifest: Self = decode_contract(
            OBSERVATION_PREFIX,
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
    field_recovery_request: FieldRecoveryRequestIdentity,
    source_fields: RecoveredFieldArtifactIdentity,
    source_result: SolverResultIdentity,
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
            field_recovery_request: wire.field_recovery_request,
            source_fields: wire.source_fields,
            source_result: wire.source_result,
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
    pub field_recovery_request: FieldRecoveryRequestIdentity,
    pub source_fields: RecoveredFieldArtifactIdentity,
    pub source_result: SolverResultIdentity,
    pub topology_digest: Digest,
    pub plan_identity: ResolvedObservationPlanIdentity,
    pub results: Vec<ObservationResult>,
}

impl ObservationArtifact {
    pub fn new(input: ObservationArtifactInput) -> Result<Self, ObservationError> {
        let mut artifact = Self {
            schema_version: 1,
            identity: ObservationArtifactIdentity(Digest::sha256([])),
            evaluation_request: input.evaluation_request,
            producer: input.producer,
            field_recovery_request: input.field_recovery_request,
            source_fields: input.source_fields,
            source_result: input.source_result,
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
            field_recovery_request: self.field_recovery_request,
            source_fields: self.source_fields,
            source_result: self.source_result,
            topology_digest: self.topology_digest,
            plan_identity: self.plan_identity,
            results: self
                .results
                .iter()
                .map(|result| ObservationArtifactEntry {
                    result: result.logical_identity,
                    observation_id: result.observation_id.clone(),
                    quantity: result.quantity,
                    location: result.location,
                    value_shape: result.value_shape.clone(),
                    partial_access: result.partial_access,
                })
                .collect(),
        };
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn from_manifest_and_results(
        manifest: ObservationArtifactManifest,
        results: Vec<ObservationResult>,
    ) -> Result<Self, ObservationError> {
        manifest.validate()?;
        if manifest.results.len() != results.len()
            || manifest
                .results
                .iter()
                .zip(&results)
                .any(|(entry, result)| {
                    entry.result != result.logical_identity
                        || entry.observation_id != result.observation_id
                        || entry.quantity != result.quantity
                        || entry.location != result.location
                        || entry.value_shape != result.value_shape
                        || entry.partial_access != result.partial_access
                })
        {
            return Err(ObservationError::Identity(
                "observation results differ from their manifest",
            ));
        }
        let artifact = Self {
            schema_version: manifest.schema_version,
            identity: manifest.identity,
            evaluation_request: manifest.evaluation_request,
            producer: manifest.producer,
            field_recovery_request: manifest.field_recovery_request,
            source_fields: manifest.source_fields,
            source_result: manifest.source_result,
            topology_digest: manifest.topology_digest,
            plan_identity: manifest.plan_identity,
            results,
        };
        artifact.validate()?;
        Ok(artifact)
    }

    pub fn validate(&self) -> Result<(), ObservationError> {
        if self.schema_version != 1 || self.results.is_empty() {
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
            OBSERVATION_PREFIX,
            "observation-artifact",
            self,
            OBSERVATION_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, ObservationError> {
        let artifact = decode_contract(
            OBSERVATION_PREFIX,
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
            field_recovery_request: self.field_recovery_request,
            source_fields: self.source_fields,
            source_result: self.source_result,
            topology_digest: self.topology_digest,
            plan_identity: self.plan_identity,
            results: self
                .results
                .iter()
                .map(|result| ObservationArtifactEntry {
                    result: result.logical_identity,
                    observation_id: result.observation_id.clone(),
                    quantity: result.quantity,
                    location: result.location,
                    value_shape: result.value_shape.clone(),
                    partial_access: result.partial_access,
                })
                .collect(),
        })
    }
}

fn derive_artifact_identity(
    manifest: &ObservationArtifactManifest,
) -> Result<ObservationArtifactIdentity, ObservationError> {
    let bytes = encode_contract(
        OBSERVATION_PREFIX,
        "observation-artifact-identity",
        &(
            manifest.schema_version,
            manifest.evaluation_request,
            manifest.producer,
            manifest.field_recovery_request,
            manifest.source_fields,
            manifest.source_result,
            manifest.topology_digest,
            manifest.plan_identity,
            &manifest.results,
        ),
        OBSERVATION_LIMITS,
    )?;
    Ok(ObservationArtifactIdentity(Digest::sha256(bytes)))
}

fn validate_resolved_observation(
    observation: &ResolvedObservation,
) -> Result<(), ObservationError> {
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
    if observation.result_component_count == 0 {
        return Err(ObservationError::Invalid(
            "resolved source field does not match its authored observation".into(),
        ));
    }
    let source_components = shape_component_count(&observation.spec.value_shape)?;
    let expected_result_components = match observation.component {
        ResolvedComponentProjection::All => source_components,
        ResolvedComponentProjection::Cartesian { axis } if axis < source_components => 1,
        ResolvedComponentProjection::Magnitude3 if source_components == 3 => 1,
        _ => {
            return Err(ObservationError::Invalid(
                "resolved component projection is incompatible with its source field".into(),
            ))
        }
    };
    if observation.result_component_count != expected_result_components {
        return Err(ObservationError::Invalid(
            "resolved result component count is inconsistent".into(),
        ));
    }
    match &observation.target {
        ResolvedObservationTarget::Nodes {
            indices,
            identities,
        } => {
            if observation.spec.source_location != FieldLocation::Node
                || observation.source_entity_domain != RecoveredFieldEntityDomain::StructuralNodes
                || observation.spec.reduction.is_some()
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
            if observation.spec.source_location != FieldLocation::Element
                || observation.source_entity_domain != *domain
                || *domain == RecoveredFieldEntityDomain::StructuralNodes
                || observation.spec.reduction.is_some()
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
        ResolvedObservationTarget::Volume { element_indices } => {
            if observation.spec.reduction.is_none()
                || !matches!(
                    observation.source_entity_domain,
                    RecoveredFieldEntityDomain::StructuralNodes
                        | RecoveredFieldEntityDomain::SolidElements
                )
                || element_indices.is_empty()
                || !strictly_increasing(element_indices)
            {
                return Err(ObservationError::Invalid(
                    "resolved volume target is empty, unordered, or lacks a reduction".into(),
                ));
            }
        }
        ResolvedObservationTarget::Boundary { face_indices } => {
            if observation.spec.source_location != FieldLocation::Node
                || observation.source_entity_domain != RecoveredFieldEntityDomain::StructuralNodes
                || observation.spec.reduction.is_none()
                || face_indices.is_empty()
                || !strictly_increasing(face_indices)
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
