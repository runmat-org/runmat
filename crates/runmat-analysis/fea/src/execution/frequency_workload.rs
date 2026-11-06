use runmat_analysis_core::{
    validate_model_sequence, AnalysisModel, AnalysisProcedure, IndependentDomain, PhysicsFamily,
    ProcedureStep, ProcedureStepId, SolutionSpace, StateKind, StateSource, StateUse,
};
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use serde::{Deserialize, Serialize};

use crate::{
    contracts::ComputeBackend, frequency::FrequencyResponsePlan,
    results::observation::ObservationExecutionPolicy, solve::dynamics::DirectHarmonicPolicy,
};

use super::{
    AnalysisExecutionError, AnalysisExecutionResult, CanonicalStateTransition, SequenceIdentity,
    StateIdentity, StepIdentity, StructuralFrequencyWorkloadIdentity,
};

const FREQUENCY_WORKLOAD_PREFIX: &[u8] = b"runmat-structural-frequency-workload/v1\0";
const FREQUENCY_WORKLOAD_LIMITS: CanonicalLimits =
    crate::contracts::canonical_digest::wire_bounded(CanonicalLimits::new(0, 0, 65_536, 64));

/// Numerical method and typed upstream state for a structural harmonic response.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", deny_unknown_fields)]
pub enum StructuralFrequencySolverPlan {
    Direct { policy: DirectHarmonicPolicy },
    ModalReduced { basis_state: StateIdentity },
}

#[derive(Clone, Copy, Debug, Default)]
pub struct StructuralFrequencyStateInputs<'a> {
    pub prestress: Option<&'a super::CanonicalSemanticState>,
    pub modal_basis: Option<&'a super::CanonicalSemanticState>,
}

impl StructuralFrequencySolverPlan {
    pub const fn solution_space(&self) -> SolutionSpace {
        match self {
            Self::Direct { .. } => SolutionSpace::Direct,
            Self::ModalReduced { .. } => SolutionSpace::ModalReduced,
        }
    }

    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        match self {
            Self::Direct { policy } => policy
                .validate()
                .map_err(|error| AnalysisExecutionError::Invalid(error.to_string())),
            Self::ModalReduced { .. } => Ok(()),
        }
    }
}

/// Numerical intent for a prestressed harmonic step before produced state identities exist.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", deny_unknown_fields)]
pub enum StructuralPrestressedFrequencySolverTemplate {
    Direct { policy: DirectHarmonicPolicy },
    ModalReduced,
}

impl StructuralPrestressedFrequencySolverTemplate {
    const fn solution_space(&self) -> SolutionSpace {
        match self {
            Self::Direct { .. } => SolutionSpace::Direct,
            Self::ModalReduced => SolutionSpace::ModalReduced,
        }
    }

    fn validate(&self) -> AnalysisExecutionResult<()> {
        match self {
            Self::Direct { policy } => policy
                .validate()
                .map_err(|error| AnalysisExecutionError::Invalid(error.to_string())),
            Self::ModalReduced => Ok(()),
        }
    }
}

/// Authored modal harmonic intent before its upstream modal-basis identity has been published.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralModalFrequencyTemplate {
    pub schema_version: u16,
    pub model: AnalysisModel,
    pub step_id: ProcedureStepId,
    pub backend: ComputeBackend,
    pub observation_execution: ObservationExecutionPolicy,
    pub plan: FrequencyResponsePlan,
}

impl StructuralModalFrequencyTemplate {
    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        if self.schema_version != 1 || self.backend != ComputeBackend::Cpu {
            return Err(AnalysisExecutionError::Invalid(
                "modal harmonic execution currently requires the CPU backend".into(),
            ));
        }
        validate_model_sequence(&self.model)
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        self.plan
            .validate()
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        let step = unique_frequency_step(&self.model, &self.step_id)?;
        validate_frequency_step(step, &self.plan, SolutionSpace::ModalReduced, false, 1)
    }

    pub fn bind(
        self,
        basis_state: StateIdentity,
    ) -> AnalysisExecutionResult<StructuralFrequencyWorkload> {
        self.validate()?;
        StructuralFrequencyWorkload::new(StructuralFrequencyWorkloadInput {
            model: self.model,
            step_id: self.step_id,
            backend: self.backend,
            observation_execution: self.observation_execution,
            plan: self.plan,
            solver: StructuralFrequencySolverPlan::ModalReduced { basis_state },
            prestress_transition: None,
        })
    }
}

/// Authored prestressed harmonic intent before its preload and optional modal-basis states have
/// been published by the sequence coordinator.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralPrestressedFrequencyTemplate {
    pub schema_version: u16,
    pub model: AnalysisModel,
    pub step_id: ProcedureStepId,
    pub backend: ComputeBackend,
    pub observation_execution: ObservationExecutionPolicy,
    pub plan: FrequencyResponsePlan,
    pub solver: StructuralPrestressedFrequencySolverTemplate,
}

impl StructuralPrestressedFrequencyTemplate {
    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        if self.schema_version != 1 || self.backend != ComputeBackend::Cpu {
            return Err(AnalysisExecutionError::Invalid(
                "prestressed harmonic execution currently requires the CPU backend".into(),
            ));
        }
        validate_model_sequence(&self.model)
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        self.plan
            .validate()
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        self.solver.validate()?;
        let step = unique_frequency_step(&self.model, &self.step_id)?;
        validate_frequency_step(
            step,
            &self.plan,
            self.solver.solution_space(),
            true,
            usize::from(matches!(
                self.solver,
                StructuralPrestressedFrequencySolverTemplate::ModalReduced
            )),
        )
    }

    pub fn bind(
        self,
        prestress_transition: CanonicalStateTransition,
        modal_basis_state: Option<StateIdentity>,
    ) -> AnalysisExecutionResult<StructuralFrequencyWorkload> {
        self.validate()?;
        let solver = match (self.solver, modal_basis_state) {
            (StructuralPrestressedFrequencySolverTemplate::Direct { policy }, None) => {
                StructuralFrequencySolverPlan::Direct { policy }
            }
            (StructuralPrestressedFrequencySolverTemplate::ModalReduced, Some(basis_state)) => {
                StructuralFrequencySolverPlan::ModalReduced { basis_state }
            }
            _ => {
                return Err(AnalysisExecutionError::Invalid(
                    "prestressed harmonic solver and modal-basis state presence differ".into(),
                ))
            }
        };
        StructuralFrequencyWorkload::new(StructuralFrequencyWorkloadInput {
            model: self.model,
            step_id: self.step_id,
            backend: self.backend,
            observation_execution: self.observation_execution,
            plan: self.plan,
            solver,
            prestress_transition: Some(prestress_transition),
        })
    }
}

/// Path-free authored input shared by every independently scheduled frequency entry.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "StructuralFrequencyWorkloadWire")]
pub struct StructuralFrequencyWorkload {
    schema_version: u16,
    identity: StructuralFrequencyWorkloadIdentity,
    model: AnalysisModel,
    step_id: ProcedureStepId,
    backend: ComputeBackend,
    observation_execution: ObservationExecutionPolicy,
    plan: FrequencyResponsePlan,
    solver: StructuralFrequencySolverPlan,
    prestress_transition: Option<CanonicalStateTransition>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StructuralFrequencyWorkloadWire {
    schema_version: u16,
    identity: StructuralFrequencyWorkloadIdentity,
    model: AnalysisModel,
    step_id: ProcedureStepId,
    backend: ComputeBackend,
    observation_execution: ObservationExecutionPolicy,
    plan: FrequencyResponsePlan,
    solver: StructuralFrequencySolverPlan,
    prestress_transition: Option<CanonicalStateTransition>,
}

impl TryFrom<StructuralFrequencyWorkloadWire> for StructuralFrequencyWorkload {
    type Error = AnalysisExecutionError;

    fn try_from(wire: StructuralFrequencyWorkloadWire) -> Result<Self, Self::Error> {
        let workload = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            model: wire.model,
            step_id: wire.step_id,
            backend: wire.backend,
            observation_execution: wire.observation_execution,
            plan: wire.plan,
            solver: wire.solver,
            prestress_transition: wire.prestress_transition,
        };
        workload.validate()?;
        Ok(workload)
    }
}

pub struct StructuralFrequencyWorkloadInput {
    pub model: AnalysisModel,
    pub step_id: ProcedureStepId,
    pub backend: ComputeBackend,
    pub observation_execution: ObservationExecutionPolicy,
    pub plan: FrequencyResponsePlan,
    pub solver: StructuralFrequencySolverPlan,
    pub prestress_transition: Option<CanonicalStateTransition>,
}

impl StructuralFrequencyWorkload {
    pub fn new(input: StructuralFrequencyWorkloadInput) -> AnalysisExecutionResult<Self> {
        let mut workload = Self {
            schema_version: 1,
            identity: StructuralFrequencyWorkloadIdentity::derive(
                "structural-frequency-workload-placeholder",
                &1_u8,
            )?,
            model: input.model,
            step_id: input.step_id,
            backend: input.backend,
            observation_execution: input.observation_execution,
            plan: input.plan,
            solver: input.solver,
            prestress_transition: input.prestress_transition,
        };
        workload.identity = workload.derive_identity()?;
        workload.validate()?;
        Ok(workload)
    }

    pub const fn identity(&self) -> StructuralFrequencyWorkloadIdentity {
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

    pub fn plan(&self) -> &FrequencyResponsePlan {
        &self.plan
    }

    pub fn solver(&self) -> &StructuralFrequencySolverPlan {
        &self.solver
    }

    pub const fn prestress_transition(&self) -> Option<&CanonicalStateTransition> {
        self.prestress_transition.as_ref()
    }

    /// Rebinds the workload to a semantically equivalent resolved model.
    ///
    /// Boundary-region resolution happens after portable study preparation. The workload remains
    /// the single numerical authority, so replacing its authored model must also regenerate its
    /// content identity and revalidate every step, plan, and state-input relationship.
    pub fn replace_model(&mut self, model: AnalysisModel) -> AnalysisExecutionResult<()> {
        self.model = model;
        self.identity = self.derive_identity()?;
        self.validate()
    }

    pub fn procedure_step(&self) -> AnalysisExecutionResult<&ProcedureStep> {
        unique_frequency_step(&self.model, &self.step_id)
    }

    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        if self.schema_version != 1 || self.backend != ComputeBackend::Cpu {
            return Err(AnalysisExecutionError::Invalid(
                "structural frequency execution currently requires the CPU backend".into(),
            ));
        }
        validate_model_sequence(&self.model)
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        self.plan
            .validate()
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        self.solver.validate()?;
        let expected_modal_inputs = usize::from(matches!(
            &self.solver,
            StructuralFrequencySolverPlan::ModalReduced { .. }
        ));
        let step = self.procedure_step()?;
        let expects_prestress = step.procedure == AnalysisProcedure::PrestressedHarmonicResponse;
        validate_frequency_step(
            step,
            &self.plan,
            self.solver.solution_space(),
            expects_prestress,
            expected_modal_inputs,
        )?;
        match (expects_prestress, self.prestress_transition.as_ref()) {
            (false, None) => {}
            (true, Some(transition)) => {
                let sequence = SequenceIdentity::from_solver_contract(&self.model.sequence)?;
                let consumer = StepIdentity::from_solver_contract(sequence, step)?;
                transition.validate()?;
                if transition.consumer != consumer || transition.consumer_step_id != self.step_id {
                    return Err(AnalysisExecutionError::Identity(
                        "structural frequency prestress transition names another consumer",
                    ));
                }
            }
            _ => {
                return Err(AnalysisExecutionError::Invalid(
                    "prestressed harmonic procedure and state transition presence differ".into(),
                ));
            }
        }
        if self.identity != self.derive_identity()? {
            return Err(AnalysisExecutionError::Identity(
                "structural frequency workload identity does not converge",
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            FREQUENCY_WORKLOAD_PREFIX,
            "structural-frequency-workload",
            self,
            FREQUENCY_WORKLOAD_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        Ok(decode_contract(
            FREQUENCY_WORKLOAD_PREFIX,
            "structural-frequency-workload",
            bytes,
            FREQUENCY_WORKLOAD_LIMITS,
        )?)
    }

    fn derive_identity(&self) -> AnalysisExecutionResult<StructuralFrequencyWorkloadIdentity> {
        StructuralFrequencyWorkloadIdentity::derive(
            "structural-frequency-workload",
            &(
                self.schema_version,
                &self.model,
                &self.step_id,
                self.backend,
                self.observation_execution,
                &self.plan,
                &self.solver,
                &self.prestress_transition,
            ),
        )
    }
}

fn unique_frequency_step<'a>(
    model: &'a AnalysisModel,
    step_id: &ProcedureStepId,
) -> AnalysisExecutionResult<&'a ProcedureStep> {
    let mut matches = model
        .sequence
        .steps
        .iter()
        .filter(|step| step.step_id == *step_id);
    let step = matches.next().ok_or_else(|| {
        AnalysisExecutionError::Invalid(format!(
            "structural frequency workload references absent step `{}`",
            step_id.as_str()
        ))
    })?;
    if matches.next().is_some() {
        return Err(AnalysisExecutionError::Invalid(format!(
            "structural frequency workload references duplicate step `{}`",
            step_id.as_str()
        )));
    }
    Ok(step)
}

fn validate_frequency_step(
    step: &ProcedureStep,
    plan: &FrequencyResponsePlan,
    solution_space: SolutionSpace,
    expects_prestress: bool,
    expected_modal_inputs: usize,
) -> AnalysisExecutionResult<()> {
    if step.family != PhysicsFamily::Structural
        || step.procedure
            != if expects_prestress {
                AnalysisProcedure::PrestressedHarmonicResponse
            } else {
                AnalysisProcedure::HarmonicResponse
            }
        || step.solution_space != solution_space
    {
        return Err(AnalysisExecutionError::Invalid(
            "structural frequency workload requires a matching structural harmonic procedure"
                .into(),
        ));
    }
    let IndependentDomain::FrequencyHertz(authored) = &step.independent_domain else {
        return Err(AnalysisExecutionError::Invalid(
            "structural harmonic procedure requires a frequency domain".into(),
        ));
    };
    if authored.iter().any(|frequency| {
        plan.frequencies_hz()
            .binary_search_by(|candidate| candidate.total_cmp(frequency))
            .is_err()
    }) {
        return Err(AnalysisExecutionError::Invalid(
            "structural frequency plan omits an authored procedure frequency".into(),
        ));
    }
    let prestress_inputs = step
        .state_inputs
        .iter()
        .filter(|input| {
            input.use_as == StateUse::Prestress
                && input.required_kind == StateKind::GeometricStiffness
                && matches!(input.source, StateSource::Upstream { .. })
        })
        .count();
    let modal_inputs = step
        .state_inputs
        .iter()
        .filter(|input| {
            input.use_as == StateUse::ModalBasis
                && input.required_kind == StateKind::ModalBasis
                && matches!(input.source, StateSource::Upstream { .. })
        })
        .count();
    if step.state_inputs.len() != usize::from(expects_prestress) + expected_modal_inputs
        || prestress_inputs != usize::from(expects_prestress)
        || modal_inputs != expected_modal_inputs
    {
        return Err(AnalysisExecutionError::Invalid(
            "structural frequency solver plan differs from its authored state inputs".into(),
        ));
    }
    Ok(())
}
