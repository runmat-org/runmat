use runmat_analysis_core::{
    validate_model_sequence, AnalysisModel, AnalysisProcedure, FormulationRequirement,
    IndependentDomain, PhysicsFamily, ProcedureStep, ProcedureStepId, SolutionSpace, StateKind,
    StateSource, StateUse,
};
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use serde::{Deserialize, Serialize};

use crate::{
    contracts::ComputeBackend, frequency::FrequencyResponsePlan,
    results::observation::ObservationExecutionPolicy, solve::dynamics::DirectHarmonicPolicy,
};

use super::{
    AnalysisExecutionError, AnalysisExecutionResult, StateIdentity,
    StructuralFrequencyWorkloadIdentity,
};

const FREQUENCY_WORKLOAD_PREFIX: &[u8] = b"runmat-structural-frequency-workload/v1\0";
const FREQUENCY_WORKLOAD_LIMITS: CanonicalLimits =
    CanonicalLimits::new(64 * 1024 * 1024, 1_000_000, 65_536, 64);

/// Numerical method and typed upstream state for a structural harmonic response.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", deny_unknown_fields)]
pub enum StructuralFrequencySolverPlan {
    Direct { policy: DirectHarmonicPolicy },
    ModalReduced { basis_state: StateIdentity },
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

/// Path-free authored input shared by every independently scheduled frequency entry.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "StructuralFrequencyWorkloadWire")]
pub struct StructuralFrequencyWorkload {
    schema_version: u16,
    identity: StructuralFrequencyWorkloadIdentity,
    model: AnalysisModel,
    step_id: ProcedureStepId,
    backend: ComputeBackend,
    formulation_requirement: FormulationRequirement,
    observation_execution: ObservationExecutionPolicy,
    plan: FrequencyResponsePlan,
    solver: StructuralFrequencySolverPlan,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StructuralFrequencyWorkloadWire {
    schema_version: u16,
    identity: StructuralFrequencyWorkloadIdentity,
    model: AnalysisModel,
    step_id: ProcedureStepId,
    backend: ComputeBackend,
    formulation_requirement: FormulationRequirement,
    observation_execution: ObservationExecutionPolicy,
    plan: FrequencyResponsePlan,
    solver: StructuralFrequencySolverPlan,
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
            formulation_requirement: wire.formulation_requirement,
            observation_execution: wire.observation_execution,
            plan: wire.plan,
            solver: wire.solver,
        };
        workload.validate()?;
        Ok(workload)
    }
}

pub struct StructuralFrequencyWorkloadInput {
    pub model: AnalysisModel,
    pub step_id: ProcedureStepId,
    pub backend: ComputeBackend,
    pub formulation_requirement: FormulationRequirement,
    pub observation_execution: ObservationExecutionPolicy,
    pub plan: FrequencyResponsePlan,
    pub solver: StructuralFrequencySolverPlan,
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
            formulation_requirement: input.formulation_requirement,
            observation_execution: input.observation_execution,
            plan: input.plan,
            solver: input.solver,
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

    pub const fn formulation_requirement(&self) -> FormulationRequirement {
        self.formulation_requirement
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

    pub fn procedure_step(&self) -> AnalysisExecutionResult<&ProcedureStep> {
        let mut matches = self
            .model
            .sequence
            .steps
            .iter()
            .filter(|step| step.step_id == self.step_id);
        let step = matches.next().ok_or_else(|| {
            AnalysisExecutionError::Invalid(format!(
                "structural frequency workload references absent step `{}`",
                self.step_id.as_str()
            ))
        })?;
        if matches.next().is_some() {
            return Err(AnalysisExecutionError::Invalid(format!(
                "structural frequency workload references duplicate step `{}`",
                self.step_id.as_str()
            )));
        }
        Ok(step)
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
        let step = self.procedure_step()?;
        if step.family != PhysicsFamily::Structural
            || step.procedure != AnalysisProcedure::HarmonicResponse
            || step.solution_space != self.solver.solution_space()
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
            self.plan
                .frequencies_hz()
                .binary_search_by(|candidate| candidate.total_cmp(frequency))
                .is_err()
        }) {
            return Err(AnalysisExecutionError::Invalid(
                "structural frequency plan omits an authored procedure frequency".into(),
            ));
        }
        match (&self.solver, step.state_inputs.as_slice()) {
            (StructuralFrequencySolverPlan::Direct { .. }, []) => {}
            (StructuralFrequencySolverPlan::ModalReduced { .. }, [input])
                if input.use_as == StateUse::ModalBasis
                    && input.required_kind == StateKind::ModalBasis
                    && matches!(input.source, StateSource::Upstream { .. }) => {}
            _ => {
                return Err(AnalysisExecutionError::Invalid(
                    "structural frequency solver plan differs from its authored state inputs"
                        .into(),
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
                self.formulation_requirement,
                self.observation_execution,
                &self.plan,
                &self.solver,
            ),
        )
    }
}
