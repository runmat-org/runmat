use runmat_analysis_core::{
    validate_model_sequence, AnalysisModel, AnalysisProcedure, IndependentDomain, PhysicsFamily,
    ProcedureStep, ProcedureStepId, RandomExcitationSpectrum, SolutionSpace, StateKind,
    StateSource, StateUse,
};
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use serde::{Deserialize, Serialize};

use crate::{
    contracts::ComputeBackend, results::observation::ObservationExecutionPolicy,
    solve::dynamics::RandomResponseIntegrationPolicy,
};

use super::{
    AnalysisExecutionError, AnalysisExecutionResult, CanonicalStateTransition, SequenceIdentity,
    StepIdentity, StructuralRandomResponseWorkloadIdentity,
};

const WORKLOAD_PREFIX: &[u8] = b"runmat-structural-random-response-workload\0";
const WORKLOAD_LIMITS: CanonicalLimits =
    crate::contracts::canonical_digest::wire_bounded(CanonicalLimits::new(0, 0, 65_536, 64));

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "StructuralRandomResponseWorkloadWire")]
pub struct StructuralRandomResponseWorkload {
    schema_version: u16,
    identity: StructuralRandomResponseWorkloadIdentity,
    model: AnalysisModel,
    step_id: ProcedureStepId,
    backend: ComputeBackend,
    observation_execution: ObservationExecutionPolicy,
    modal_basis_transition: CanonicalStateTransition,
    integration_policy: RandomResponseIntegrationPolicy,
    maximum_intervals_per_task: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StructuralRandomResponseWorkloadWire {
    schema_version: u16,
    identity: StructuralRandomResponseWorkloadIdentity,
    model: AnalysisModel,
    step_id: ProcedureStepId,
    backend: ComputeBackend,
    observation_execution: ObservationExecutionPolicy,
    modal_basis_transition: CanonicalStateTransition,
    integration_policy: RandomResponseIntegrationPolicy,
    maximum_intervals_per_task: usize,
}

pub struct StructuralRandomResponseWorkloadInput {
    pub model: AnalysisModel,
    pub step_id: ProcedureStepId,
    pub backend: ComputeBackend,
    pub observation_execution: ObservationExecutionPolicy,
    pub modal_basis_transition: CanonicalStateTransition,
    pub integration_policy: RandomResponseIntegrationPolicy,
    pub maximum_intervals_per_task: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralRandomResponseTemplate {
    pub schema_version: u16,
    pub model: AnalysisModel,
    pub step_id: ProcedureStepId,
    pub backend: ComputeBackend,
    pub observation_execution: ObservationExecutionPolicy,
    pub integration_policy: RandomResponseIntegrationPolicy,
    pub maximum_intervals_per_task: usize,
}

impl StructuralRandomResponseTemplate {
    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        validate_common(
            self.schema_version,
            &self.model,
            &self.step_id,
            self.backend,
        )?;
        validate_execution_policy(self.integration_policy, self.maximum_intervals_per_task)
    }

    pub fn bind(
        self,
        modal_basis_transition: CanonicalStateTransition,
    ) -> AnalysisExecutionResult<StructuralRandomResponseWorkload> {
        self.validate()?;
        StructuralRandomResponseWorkload::new(StructuralRandomResponseWorkloadInput {
            model: self.model,
            step_id: self.step_id,
            backend: self.backend,
            observation_execution: self.observation_execution,
            modal_basis_transition,
            integration_policy: self.integration_policy,
            maximum_intervals_per_task: self.maximum_intervals_per_task,
        })
    }
}

impl TryFrom<StructuralRandomResponseWorkloadWire> for StructuralRandomResponseWorkload {
    type Error = AnalysisExecutionError;

    fn try_from(wire: StructuralRandomResponseWorkloadWire) -> Result<Self, Self::Error> {
        let workload = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            model: wire.model,
            step_id: wire.step_id,
            backend: wire.backend,
            observation_execution: wire.observation_execution,
            modal_basis_transition: wire.modal_basis_transition,
            integration_policy: wire.integration_policy,
            maximum_intervals_per_task: wire.maximum_intervals_per_task,
        };
        workload.validate()?;
        Ok(workload)
    }
}

impl StructuralRandomResponseWorkload {
    pub fn new(input: StructuralRandomResponseWorkloadInput) -> AnalysisExecutionResult<Self> {
        let mut workload = Self {
            schema_version: 1,
            identity: StructuralRandomResponseWorkloadIdentity::derive(
                "structural-random-response-workload-placeholder",
                &1_u8,
            )?,
            model: input.model,
            step_id: input.step_id,
            backend: input.backend,
            observation_execution: input.observation_execution,
            modal_basis_transition: input.modal_basis_transition,
            integration_policy: input.integration_policy,
            maximum_intervals_per_task: input.maximum_intervals_per_task,
        };
        workload.identity = workload.derive_identity()?;
        workload.validate()?;
        Ok(workload)
    }

    pub const fn identity(&self) -> StructuralRandomResponseWorkloadIdentity {
        self.identity
    }

    pub fn model(&self) -> &AnalysisModel {
        &self.model
    }

    pub fn step_id(&self) -> &ProcedureStepId {
        &self.step_id
    }

    pub const fn backend(&self) -> ComputeBackend {
        self.backend
    }

    pub const fn observation_execution(&self) -> ObservationExecutionPolicy {
        self.observation_execution
    }

    pub const fn modal_basis_transition(&self) -> &CanonicalStateTransition {
        &self.modal_basis_transition
    }

    pub const fn integration_policy(&self) -> RandomResponseIntegrationPolicy {
        self.integration_policy
    }

    pub const fn maximum_intervals_per_task(&self) -> usize {
        self.maximum_intervals_per_task
    }

    pub fn procedure_step(&self) -> AnalysisExecutionResult<&ProcedureStep> {
        validate_common(
            self.schema_version,
            &self.model,
            &self.step_id,
            self.backend,
        )
    }

    pub fn excitation(&self) -> AnalysisExecutionResult<&RandomExcitationSpectrum> {
        let IndependentDomain::RandomExcitation { excitation_id } =
            &self.procedure_step()?.independent_domain
        else {
            return Err(invalid(
                "random-response workload has an incompatible domain",
            ));
        };
        self.model
            .random_excitations
            .iter()
            .find(|excitation| excitation.excitation_id == *excitation_id)
            .ok_or_else(|| invalid("random-response step references an absent excitation"))
    }

    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        let step = self.procedure_step()?;
        validate_execution_policy(self.integration_policy, self.maximum_intervals_per_task)?;
        self.modal_basis_transition.validate()?;
        let sequence = SequenceIdentity::from_solver_contract(&self.model.sequence)?;
        let consumer = StepIdentity::from_solver_contract(sequence, step)?;
        if self.modal_basis_transition.consumer != consumer
            || self.modal_basis_transition.consumer_step_id != self.step_id
            || self.excitation()?.validate_intrinsic().is_err()
            || self.identity != self.derive_identity()?
        {
            return Err(AnalysisExecutionError::Identity(
                "random-response workload differs from its step, excitation, or modal-basis transition",
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            WORKLOAD_PREFIX,
            "structural-random-response-workload",
            self,
            WORKLOAD_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        Ok(decode_contract(
            WORKLOAD_PREFIX,
            "structural-random-response-workload",
            bytes,
            WORKLOAD_LIMITS,
        )?)
    }

    fn derive_identity(&self) -> AnalysisExecutionResult<StructuralRandomResponseWorkloadIdentity> {
        StructuralRandomResponseWorkloadIdentity::derive(
            "structural-random-response-workload",
            &(
                self.schema_version,
                &self.model,
                &self.step_id,
                self.backend,
                self.observation_execution,
                &self.modal_basis_transition,
                self.integration_policy,
                self.maximum_intervals_per_task,
            ),
        )
    }
}

fn validate_common<'a>(
    schema_version: u16,
    model: &'a AnalysisModel,
    step_id: &ProcedureStepId,
    backend: ComputeBackend,
) -> AnalysisExecutionResult<&'a ProcedureStep> {
    if schema_version != 1 || backend != ComputeBackend::Cpu {
        return Err(invalid(
            "random response currently requires schema one and the qualified CPU backend",
        ));
    }
    validate_model_sequence(model).map_err(|error| invalid(error.to_string()))?;
    let mut matches = model
        .sequence
        .steps
        .iter()
        .filter(|step| step.step_id == *step_id);
    let step = matches
        .next()
        .ok_or_else(|| invalid("random-response workload references an absent step"))?;
    if matches.next().is_some() {
        return Err(invalid(
            "random-response workload references a duplicate step",
        ));
    }
    if step.family != PhysicsFamily::Structural
        || step.procedure != AnalysisProcedure::RandomResponse
        || step.solution_space != SolutionSpace::ModalReduced
        || !matches!(
            step.independent_domain,
            IndependentDomain::RandomExcitation { .. }
        )
        || !step.activations.is_empty()
        || !matches!(
            step.state_inputs.as_slice(),
            [input]
                if input.use_as == StateUse::ModalBasis
                    && input.required_kind == StateKind::ModalBasis
                    && matches!(input.source, StateSource::Upstream { .. })
        )
        || !matches!(
            step.state_outputs.as_slice(),
            [output] if output.kind == StateKind::StructuralRandomStatistics
        )
    {
        return Err(invalid(
            "random response requires one modal-reduced excitation step, one modal-basis input, and one statistics output",
        ));
    }
    Ok(step)
}

fn validate_execution_policy(
    policy: RandomResponseIntegrationPolicy,
    maximum_intervals_per_task: usize,
) -> AnalysisExecutionResult<()> {
    policy
        .validate()
        .map_err(|error| invalid(error.to_string()))?;
    if !(1..=4096).contains(&maximum_intervals_per_task) {
        return Err(invalid(
            "random-response interval tasks must contain between one and 4096 intervals",
        ));
    }
    Ok(())
}

fn invalid(message: impl Into<String>) -> AnalysisExecutionError {
    AnalysisExecutionError::Invalid(message.into())
}
