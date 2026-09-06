use std::collections::BTreeSet;

use runmat_analysis_core::{
    AnalysisField, AnalysisFieldValues, ProcedureStepId, ReferenceStatePolicy, StateCompleteness,
    StateId, StateKind, StateOutput, StatePersistence, StateTransition, StateTransitionMethod,
    TopologyTransitionPolicy,
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
    pub producer: StepIdentity,
    pub declaration: StateOutput,
    pub topology_digest: Digest,
    pub source_result: SolverResultIdentity,
}

impl SemanticStatePublicationRequest {
    pub fn structural_linear(
        producer: StepIdentity,
        declaration: StateOutput,
        topology_digest: Digest,
        source_result: SolverResultIdentity,
    ) -> AnalysisExecutionResult<Self> {
        let request = Self {
            schema_version: 1,
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
            || self.declaration.kind != StateKind::StructuralPrimary
        {
            return Err(AnalysisExecutionError::Invalid(
                "structural state publication requires a complete persisted structural-primary declaration"
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
    pub fields: Vec<AnalysisField>,
}

pub struct CanonicalSemanticStateInput {
    pub producer: StepIdentity,
    pub declaration: StateOutput,
    pub topology_digest: Digest,
    pub source_result: SolverResultIdentity,
    pub fields: Vec<AnalysisField>,
}

impl CanonicalSemanticState {
    pub fn from_publication(
        request: &SemanticStatePublicationRequest,
        fields: &[AnalysisField],
    ) -> AnalysisExecutionResult<Self> {
        request.validate()?;
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
            fields: selected,
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
            fields: input.fields,
        };
        state.content_digest = state.derive_content_digest()?;
        state.identity = state.derive_identity()?;
        state.validate()?;
        Ok(state)
    }

    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        if self.schema_version != 1 || self.state_id.0.trim().is_empty() || self.fields.is_empty() {
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
        let mut field_ids = BTreeSet::new();
        for field in &self.fields {
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
            &(self.kind, self.topology_digest, &self.fields),
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
