//! Governed transitions bind original semantic states to authored consumers.
use super::*;

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
