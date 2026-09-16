use std::collections::BTreeSet;

use runmat_analysis_core::{
    AnalysisField, AnalysisFieldValues, AnalysisModel, ProcedureStepId, ReferenceStatePolicy,
    StateCompleteness, StateId, StateKind, StateOutput, StatePersistence, StateSource,
    StateTransition, StateTransitionMethod, StateUse, TopologyTransitionPolicy,
};
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

use super::{
    AnalysisExecutionError, AnalysisExecutionResult, SequenceIdentity, SolverResultIdentity,
    StateIdentity, StateTransitionIdentity, StepIdentity,
};

const STATE_PREFIX: &[u8] = b"runmat-analysis-semantic-state/v1\0";
const STATE_LIMITS: CanonicalLimits =
    CanonicalLimits::new(1024 * 1024 * 1024, 4_000_000, 65_536, 64);

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticStatePublicationRequest {
    pub schema_version: u16,
    pub source: SemanticStatePublicationSource,
    pub producer: StepIdentity,
    pub declaration: StateOutput,
    pub topology_digest: Digest,
    pub source_result: SolverResultIdentity,
}

/// Typed interpretation of the solver result consumed by state publication.
///
/// `SolverResult` is deliberately a generic transport object kind. This discriminator prevents a
/// worker from guessing its schema or accepting one procedure's result as another state kind.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SemanticStatePublicationSource {
    StructuralPrimaryFields,
    StructuralLinearization,
    ModalBasis,
    StructuralResponseEnvelope,
}

impl SemanticStatePublicationRequest {
    pub fn new(
        source: SemanticStatePublicationSource,
        producer: StepIdentity,
        declaration: StateOutput,
        topology_digest: Digest,
        source_result: SolverResultIdentity,
    ) -> AnalysisExecutionResult<Self> {
        let request = Self {
            schema_version: 1,
            source,
            producer,
            declaration,
            topology_digest,
            source_result,
        };
        request.validate()?;
        Ok(request)
    }

    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        if self.schema_version != 1
            || self.declaration.state_id.0.trim().is_empty()
            || self.declaration.completeness != StateCompleteness::Complete
            || self.declaration.persistence == StatePersistence::CheckpointOnly
            || !matches!(
                (self.source, self.declaration.kind),
                (
                    SemanticStatePublicationSource::StructuralPrimaryFields,
                    StateKind::StructuralPrimary
                ) | (
                    SemanticStatePublicationSource::StructuralLinearization,
                    StateKind::GeometricStiffness
                ) | (
                    SemanticStatePublicationSource::ModalBasis,
                    StateKind::ModalBasis
                ) | (
                    SemanticStatePublicationSource::StructuralResponseEnvelope,
                    StateKind::StructuralResponseEnvelope
                )
            )
        {
            return Err(AnalysisExecutionError::Invalid(
                "semantic-state publication source does not match its complete persisted declaration"
                    .into(),
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            STATE_PREFIX,
            "semantic-state-publication-request",
            self,
            STATE_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        let request: Self = decode_contract(
            STATE_PREFIX,
            "semantic-state-publication-request",
            bytes,
            STATE_LIMITS,
        )?;
        request.validate()?;
        Ok(request)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalSemanticState {
    pub schema_version: u16,
    pub identity: StateIdentity,
    pub content_digest: Digest,
    pub producer: StepIdentity,
    pub state_id: StateId,
    pub kind: StateKind,
    pub completeness: StateCompleteness,
    pub persistence: StatePersistence,
    pub topology_digest: Digest,
    pub source_result: SolverResultIdentity,
    pub payload: CanonicalSemanticStatePayload,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", deny_unknown_fields)]
pub enum CanonicalSemanticStatePayload {
    Fields {
        fields: Vec<AnalysisField>,
    },
    StructuralFrequency {
        publication: Box<super::StructuralFrequencyPublicationManifest>,
    },
    ModalBasis {
        basis: Box<crate::pipeline::modal_basis::ModalBasisArtifact>,
    },
    StructuralLinearization {
        linearization:
            Box<crate::pipeline::structural_linearization::StructuralLinearizationArtifact>,
    },
    StructuralResponseEnvelope {
        envelope: Box<super::StructuralResponseEnvelope>,
    },
}

pub struct CanonicalSemanticStateInput {
    pub producer: StepIdentity,
    pub declaration: StateOutput,
    pub topology_digest: Digest,
    pub source_result: SolverResultIdentity,
    pub payload: CanonicalSemanticStatePayload,
}

impl CanonicalSemanticState {
    pub fn from_publication(
        request: &SemanticStatePublicationRequest,
        fields: &[AnalysisField],
    ) -> AnalysisExecutionResult<Self> {
        request.validate()?;
        if request.source != SemanticStatePublicationSource::StructuralPrimaryFields {
            return Err(AnalysisExecutionError::Invalid(
                "field state publication requires a structural-primary source".into(),
            ));
        }
        let mut by_id = std::collections::BTreeMap::new();
        for field in fields {
            if by_id.insert(field.field_id.as_str(), field).is_some() {
                return Err(AnalysisExecutionError::Invalid(format!(
                    "solver result contains duplicate state field `{}`",
                    field.field_id
                )));
            }
        }
        let selected = structural_state_field_ids(request.declaration.kind)?
            .iter()
            .map(|field_id| {
                by_id
                    .get(*field_id)
                    .map(|field| (*field).clone())
                    .ok_or_else(|| {
                        AnalysisExecutionError::Invalid(format!(
                            "solver result does not contain state field `{field_id}`"
                        ))
                    })
            })
            .collect::<AnalysisExecutionResult<Vec<_>>>()?;
        Self::new(CanonicalSemanticStateInput {
            producer: request.producer,
            declaration: request.declaration.clone(),
            topology_digest: request.topology_digest,
            source_result: request.source_result,
            payload: CanonicalSemanticStatePayload::Fields { fields: selected },
        })
    }

    pub fn from_modal_publication(
        request: &SemanticStatePublicationRequest,
        basis: crate::pipeline::modal_basis::ModalBasisArtifact,
    ) -> AnalysisExecutionResult<Self> {
        request.validate()?;
        if request.source != SemanticStatePublicationSource::ModalBasis {
            return Err(AnalysisExecutionError::Invalid(
                "modal-basis state publication requires a modal result source".into(),
            ));
        }
        basis
            .validate()
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        Self::new(CanonicalSemanticStateInput {
            producer: request.producer,
            declaration: request.declaration.clone(),
            topology_digest: request.topology_digest,
            source_result: request.source_result,
            payload: CanonicalSemanticStatePayload::ModalBasis {
                basis: Box::new(basis),
            },
        })
    }

    pub fn from_linearization_publication(
        request: &SemanticStatePublicationRequest,
        linearization: crate::pipeline::structural_linearization::StructuralLinearizationArtifact,
    ) -> AnalysisExecutionResult<Self> {
        request.validate()?;
        if request.source != SemanticStatePublicationSource::StructuralLinearization
            || linearization.topology_digest() != request.topology_digest
        {
            return Err(AnalysisExecutionError::Invalid(
                "geometric-stiffness state publication requires a topology-matched structural linearization"
                    .into(),
            ));
        }
        linearization
            .validate()
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        Self::new(CanonicalSemanticStateInput {
            producer: request.producer,
            declaration: request.declaration.clone(),
            topology_digest: request.topology_digest,
            source_result: request.source_result,
            payload: CanonicalSemanticStatePayload::StructuralLinearization {
                linearization: Box::new(linearization),
            },
        })
    }

    pub fn from_structural_response_envelope_publication(
        request: &SemanticStatePublicationRequest,
        result: &super::StructuralModalResponseSpectrumSolverResult,
    ) -> AnalysisExecutionResult<Self> {
        request.validate()?;
        result.validate()?;
        if request.source != SemanticStatePublicationSource::StructuralResponseEnvelope
            || result.assembly().scope().topology_digest() != request.topology_digest
        {
            return Err(AnalysisExecutionError::Invalid(
                "structural response-envelope state publication requires a topology-matched response-spectrum result"
                    .into(),
            ));
        }
        Self::new(CanonicalSemanticStateInput {
            producer: request.producer,
            declaration: request.declaration.clone(),
            topology_digest: request.topology_digest,
            source_result: request.source_result,
            payload: CanonicalSemanticStatePayload::StructuralResponseEnvelope {
                envelope: Box::new(super::StructuralResponseEnvelope::from_result(result)?),
            },
        })
    }

    pub fn new(input: CanonicalSemanticStateInput) -> AnalysisExecutionResult<Self> {
        let mut state = Self {
            schema_version: 1,
            identity: StateIdentity::new(
                input.producer,
                Digest::sha256([]),
                input.topology_digest,
                input.source_result,
                Digest::sha256([]),
            )?,
            content_digest: Digest::sha256([]),
            producer: input.producer,
            state_id: input.declaration.state_id,
            kind: input.declaration.kind,
            completeness: input.declaration.completeness,
            persistence: input.declaration.persistence,
            topology_digest: input.topology_digest,
            source_result: input.source_result,
            payload: input.payload,
        };
        state.content_digest = state.derive_content_digest()?;
        state.identity = state.derive_identity()?;
        state.validate()?;
        Ok(state)
    }

    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        if self.schema_version != 1 || self.state_id.0.trim().is_empty() {
            return Err(AnalysisExecutionError::Invalid(
                "semantic state metadata or field inventory is incomplete".into(),
            ));
        }
        if self.completeness != StateCompleteness::Complete
            && self.persistence == StatePersistence::Canonical
        {
            return Err(AnalysisExecutionError::Invalid(
                "only complete semantic state may have canonical persistence".into(),
            ));
        }
        match &self.payload {
            CanonicalSemanticStatePayload::Fields { fields } => {
                if matches!(
                    self.kind,
                    StateKind::ModalBasis
                        | StateKind::StructuralComplex
                        | StateKind::StructuralResponseEnvelope
                        | StateKind::StructuralRandomStatistics
                ) || fields.is_empty()
                {
                    return Err(AnalysisExecutionError::Invalid(
                        "field semantic-state payload does not match its declared state kind"
                            .into(),
                    ));
                }
                validate_portable_fields(fields)?;
            }
            CanonicalSemanticStatePayload::StructuralFrequency { publication } => {
                if self.kind != StateKind::StructuralComplex {
                    return Err(AnalysisExecutionError::Invalid(
                        "structural-frequency semantic-state payload does not match its declared state kind"
                            .into(),
                    ));
                }
                publication.validate()?;
                if publication.result().identity() != self.source_result {
                    return Err(AnalysisExecutionError::Identity(
                        "structural-frequency semantic state names another solver result",
                    ));
                }
            }
            CanonicalSemanticStatePayload::ModalBasis { basis } => {
                if self.kind != StateKind::ModalBasis || basis.modes().is_empty() {
                    return Err(AnalysisExecutionError::Invalid(
                        "modal-basis semantic-state payload does not match its declared state kind"
                            .into(),
                    ));
                }
            }
            CanonicalSemanticStatePayload::StructuralLinearization { linearization } => {
                if self.kind != StateKind::GeometricStiffness
                    || linearization.topology_digest() != self.topology_digest
                {
                    return Err(AnalysisExecutionError::Invalid(
                        "structural-linearization payload does not match its declared state kind or topology"
                            .into(),
                    ));
                }
                linearization
                    .validate()
                    .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
            }
            CanonicalSemanticStatePayload::StructuralResponseEnvelope { envelope } => {
                if self.kind != StateKind::StructuralResponseEnvelope
                    || envelope.scope().topology_digest() != self.topology_digest
                {
                    return Err(AnalysisExecutionError::Invalid(
                        "structural response-envelope payload does not match its declared state kind or topology"
                            .into(),
                    ));
                }
                envelope.validate()?;
            }
        }
        let content_digest = self.derive_content_digest()?;
        if self.content_digest != content_digest || self.identity != self.derive_identity()? {
            return Err(AnalysisExecutionError::Identity(
                "semantic state identity does not converge",
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            STATE_PREFIX,
            "semantic-state",
            self,
            STATE_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        let state: Self = decode_contract(STATE_PREFIX, "semantic-state", bytes, STATE_LIMITS)?;
        state.validate()?;
        Ok(state)
    }

    fn derive_content_digest(&self) -> AnalysisExecutionResult<Digest> {
        Ok(Digest::sha256(encode_contract(
            STATE_PREFIX,
            "semantic-state-content",
            &(self.kind, self.topology_digest, &self.payload),
            STATE_LIMITS,
        )?))
    }

    fn derive_identity(&self) -> AnalysisExecutionResult<StateIdentity> {
        let declaration_digest = Digest::sha256(encode_contract(
            STATE_PREFIX,
            "semantic-state-declaration",
            &(
                &self.state_id,
                self.kind,
                self.completeness,
                self.persistence,
            ),
            STATE_LIMITS,
        )?);
        StateIdentity::new(
            self.producer,
            declaration_digest,
            self.topology_digest,
            self.source_result,
            self.content_digest,
        )
    }
}

impl CanonicalSemanticStatePayload {
    pub fn fields(&self) -> Option<&[AnalysisField]> {
        match self {
            Self::Fields { fields } => Some(fields),
            Self::StructuralFrequency { .. }
            | Self::ModalBasis { .. }
            | Self::StructuralLinearization { .. }
            | Self::StructuralResponseEnvelope { .. } => None,
        }
    }

    pub fn structural_frequency(&self) -> Option<&super::StructuralFrequencyPublicationManifest> {
        match self {
            Self::StructuralFrequency { publication } => Some(publication.as_ref()),
            Self::Fields { .. }
            | Self::ModalBasis { .. }
            | Self::StructuralLinearization { .. }
            | Self::StructuralResponseEnvelope { .. } => None,
        }
    }

    pub fn modal_basis(&self) -> Option<&crate::pipeline::modal_basis::ModalBasisArtifact> {
        match self {
            Self::Fields { .. }
            | Self::StructuralFrequency { .. }
            | Self::StructuralLinearization { .. }
            | Self::StructuralResponseEnvelope { .. } => None,
            Self::ModalBasis { basis } => Some(basis.as_ref()),
        }
    }

    pub fn structural_linearization(
        &self,
    ) -> Option<&crate::pipeline::structural_linearization::StructuralLinearizationArtifact> {
        match self {
            Self::StructuralLinearization { linearization } => Some(linearization.as_ref()),
            Self::Fields { .. }
            | Self::StructuralFrequency { .. }
            | Self::ModalBasis { .. }
            | Self::StructuralResponseEnvelope { .. } => None,
        }
    }

    pub fn structural_response_envelope(&self) -> Option<&super::StructuralResponseEnvelope> {
        match self {
            Self::StructuralResponseEnvelope { envelope } => Some(envelope.as_ref()),
            Self::Fields { .. }
            | Self::StructuralFrequency { .. }
            | Self::ModalBasis { .. }
            | Self::StructuralLinearization { .. } => None,
        }
    }
}

fn validate_portable_fields(fields: &[AnalysisField]) -> AnalysisExecutionResult<()> {
    let mut field_ids = BTreeSet::new();
    for field in fields {
        let AnalysisFieldValues::HostF64(values) = &field.values else {
            return Err(AnalysisExecutionError::Invalid(
                "portable semantic state cannot contain a device-local field reference".into(),
            ));
        };
        let expected = field
            .shape
            .iter()
            .try_fold(1_usize, |count, extent| count.checked_mul(*extent));
        if field.field_id.trim().is_empty()
            || !field_ids.insert(field.field_id.as_str())
            || expected != Some(values.len())
            || values.is_empty()
            || values.iter().any(|value| !value.is_finite())
        {
            return Err(AnalysisExecutionError::Invalid(
                "semantic state fields must be unique, finite, non-empty host arrays with exact shapes"
                    .into(),
            ));
        }
    }
    Ok(())
}

fn structural_state_field_ids(kind: StateKind) -> AnalysisExecutionResult<&'static [&'static str]> {
    match kind {
        StateKind::StructuralPrimary => Ok(&[crate::contracts::FEA_FIELD_STRUCTURAL_DISPLACEMENT]),
        _ => Err(AnalysisExecutionError::Invalid(
            "structural linear execution can publish only structural-primary state".into(),
        )),
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalStateTransition {
    pub schema_version: u16,
    pub identity: StateTransitionIdentity,
    pub sequence: SequenceIdentity,
    pub source_state: StateIdentity,
    pub producer: StepIdentity,
    pub consumer: StepIdentity,
    pub producer_step_id: ProcedureStepId,
    pub consumer_step_id: ProcedureStepId,
    pub transition_id: runmat_analysis_core::TransitionId,
    pub state_id: StateId,
    pub method: StateTransitionMethod,
    pub topology: TopologyTransitionPolicy,
    pub reference: ReferenceStatePolicy,
    pub source_topology_digest: Digest,
    pub target_topology_digest: Digest,
}

pub struct CanonicalStateTransitionInput<'a> {
    pub sequence: SequenceIdentity,
    pub producer: StepIdentity,
    pub consumer: StepIdentity,
    pub authored: &'a StateTransition,
    pub source: &'a CanonicalSemanticState,
    pub target_topology_digest: Digest,
}

impl CanonicalStateTransition {
    pub fn bind_upstream(
        model: &AnalysisModel,
        consumer_step_id: &ProcedureStepId,
        source: &CanonicalSemanticState,
        expected_use: StateUse,
        target_topology_digest: Digest,
    ) -> AnalysisExecutionResult<Self> {
        source.validate()?;
        let consumer = model
            .sequence
            .steps
            .iter()
            .find(|step| step.step_id == *consumer_step_id)
            .ok_or(AnalysisExecutionError::Identity(
                "state transition consumer is absent from the authored sequence",
            ))?;
        let mut matching_inputs = consumer.state_inputs.iter().filter(|input| {
            input.use_as == expected_use
                && input.required_kind == source.kind
                && matches!(
                    &input.source,
                    StateSource::Upstream { state_id, .. } if state_id == &source.state_id
                )
        });
        let input = matching_inputs.next().ok_or_else(|| {
            AnalysisExecutionError::Invalid(
                "consumer has no upstream state input matching the published state".into(),
            )
        })?;
        if matching_inputs.next().is_some() {
            return Err(AnalysisExecutionError::Invalid(
                "consumer has multiple upstream inputs matching one published state".into(),
            ));
        }
        let StateSource::Upstream {
            producer_step,
            state_id,
        } = &input.source
        else {
            return Err(AnalysisExecutionError::Invalid(
                "state transition input must name an upstream producer".into(),
            ));
        };
        let producer_contract = model
            .sequence
            .steps
            .iter()
            .find(|step| step.step_id == *producer_step)
            .ok_or(AnalysisExecutionError::Identity(
                "state transition producer is absent from the authored sequence",
            ))?;
        let authored = model
            .sequence
            .transitions
            .iter()
            .find(|transition| {
                transition.producer_step == *producer_step
                    && transition.consumer_step == *consumer_step_id
                    && transition.state_id == *state_id
            })
            .ok_or_else(|| {
                AnalysisExecutionError::Invalid(
                    "state input has no matching authored transition".into(),
                )
            })?;
        let sequence = SequenceIdentity::from_solver_contract(&model.sequence)?;
        let producer = StepIdentity::from_solver_contract(sequence, producer_contract)?;
        let consumer = StepIdentity::from_solver_contract(sequence, consumer)?;
        if source.producer != producer {
            return Err(AnalysisExecutionError::Identity(
                "published state names a different producer step",
            ));
        }
        Self::new(CanonicalStateTransitionInput {
            sequence,
            producer,
            consumer,
            authored,
            source,
            target_topology_digest,
        })
    }

    pub fn new(input: CanonicalStateTransitionInput<'_>) -> AnalysisExecutionResult<Self> {
        input.source.validate()?;
        if input.source.producer != input.producer
            || input.source.state_id != input.authored.state_id
        {
            return Err(AnalysisExecutionError::Identity(
                "state transition source differs from its producer or authored state",
            ));
        }
        let transition_digest = authored_transition_digest(input.authored)?;
        let transition = Self {
            schema_version: 1,
            identity: StateTransitionIdentity::new(
                input.sequence,
                input.producer,
                input.consumer,
                input.source.identity,
                input.source.topology_digest,
                input.target_topology_digest,
                transition_digest,
            )?,
            sequence: input.sequence,
            source_state: input.source.identity,
            producer: input.producer,
            consumer: input.consumer,
            producer_step_id: input.authored.producer_step.clone(),
            consumer_step_id: input.authored.consumer_step.clone(),
            transition_id: input.authored.transition_id.clone(),
            state_id: input.authored.state_id.clone(),
            method: input.authored.method,
            topology: input.authored.topology,
            reference: input.authored.reference,
            source_topology_digest: input.source.topology_digest,
            target_topology_digest: input.target_topology_digest,
        };
        transition.validate()?;
        Ok(transition)
    }

    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        if self.schema_version != 1
            || self.producer == self.consumer
            || self.producer_step_id.0.trim().is_empty()
            || self.consumer_step_id.0.trim().is_empty()
            || self.transition_id.0.trim().is_empty()
            || self.state_id.0.trim().is_empty()
        {
            return Err(AnalysisExecutionError::Invalid(
                "state transition metadata is incomplete or self-referential".into(),
            ));
        }
        if !matches!(
            self.method,
            StateTransitionMethod::Reuse | StateTransitionMethod::InitialGuessOnly
        ) || self.topology != TopologyTransitionPolicy::RequireIdentical
            || self.reference != ReferenceStatePolicy::Preserve
            || self.source_topology_digest != self.target_topology_digest
        {
            return Err(AnalysisExecutionError::Invalid(
                "only byte-preserving state reuse or initial-guess transfer on identical topology is executable"
                    .into(),
            ));
        }
        let authored = StateTransition {
            transition_id: self.transition_id.clone(),
            producer_step: self.producer_step_id.clone(),
            consumer_step: self.consumer_step_id.clone(),
            state_id: self.state_id.clone(),
            method: self.method,
            topology: self.topology,
            reference: self.reference,
        };
        if self.identity
            != StateTransitionIdentity::new(
                self.sequence,
                self.producer,
                self.consumer,
                self.source_state,
                self.source_topology_digest,
                self.target_topology_digest,
                authored_transition_digest(&authored)?,
            )?
        {
            return Err(AnalysisExecutionError::Identity(
                "state transition identity does not converge",
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            STATE_PREFIX,
            "state-transition",
            self,
            STATE_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        let transition: Self =
            decode_contract(STATE_PREFIX, "state-transition", bytes, STATE_LIMITS)?;
        transition.validate()?;
        Ok(transition)
    }
}

fn authored_transition_digest(transition: &StateTransition) -> AnalysisExecutionResult<Digest> {
    Ok(Digest::sha256(encode_contract(
        STATE_PREFIX,
        "authored-state-transition",
        transition,
        STATE_LIMITS,
    )?))
}
