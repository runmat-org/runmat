use runmat_analysis_core::{
    validate_model_sequence, AnalysisModel, AnalysisProcedure, IndependentDomain, PhysicsFamily,
    ProcedureStep, ProcedureStepId, SolutionSpace, StateKind, StateSource, StateUse,
};
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use serde::{Deserialize, Serialize};

use crate::contracts::ComputeBackend;
use crate::results::observation::ObservationExecutionPolicy;

use super::{
    AnalysisExecutionError, AnalysisExecutionResult, CanonicalStateTransition, SequenceIdentity,
    StepIdentity, StructuralBucklingWorkloadIdentity,
};

const BUCKLING_WORKLOAD_PREFIX: &[u8] = b"runmat-structural-buckling-workload/v1\0";
const BUCKLING_WORKLOAD_LIMITS: CanonicalLimits =
    crate::contracts::canonical_digest::wire_bounded(CanonicalLimits::new(0, 0, 65_536, 64));
const MAXIMUM_REQUESTED_MODES: usize = 4096;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralBucklingSolverPolicy {
    requested_mode_count: usize,
    relative_tolerance: f64,
    maximum_iterations: usize,
}

impl StructuralBucklingSolverPolicy {
    pub fn new(
        requested_mode_count: usize,
        relative_tolerance: f64,
        maximum_iterations: usize,
    ) -> AnalysisExecutionResult<Self> {
        let policy = Self {
            requested_mode_count,
            relative_tolerance,
            maximum_iterations,
        };
        policy.validate()?;
        Ok(policy)
    }

    pub const fn requested_mode_count(self) -> usize {
        self.requested_mode_count
    }

    pub const fn relative_tolerance(self) -> f64 {
        self.relative_tolerance
    }

    pub const fn maximum_iterations(self) -> usize {
        self.maximum_iterations
    }

    pub fn validate(self) -> AnalysisExecutionResult<()> {
        if self.requested_mode_count == 0
            || self.requested_mode_count > MAXIMUM_REQUESTED_MODES
            || !self.relative_tolerance.is_finite()
            || self.relative_tolerance <= 0.0
            || self.relative_tolerance >= 1.0
            || self.maximum_iterations == 0
        {
            return Err(invalid(
                "structural buckling solver policy is outside its finite bounds",
            ));
        }
        Ok(())
    }
}

impl Default for StructuralBucklingSolverPolicy {
    fn default() -> Self {
        Self {
            requested_mode_count: 3,
            relative_tolerance: 1.0e-8,
            maximum_iterations: 256,
        }
    }
}

/// Authored buckling intent before the exact preload state has been published.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralBucklingTemplate {
    pub schema_version: u16,
    pub model: AnalysisModel,
    pub step_id: ProcedureStepId,
    pub backend: ComputeBackend,
    pub observation_execution: ObservationExecutionPolicy,
    pub solver: StructuralBucklingSolverPolicy,
}

impl StructuralBucklingTemplate {
    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        validate_common(
            self.schema_version,
            &self.model,
            &self.step_id,
            self.backend,
            self.solver,
        )
        .map(|_| ())
    }

    pub fn bind(
        self,
        prestress_transition: CanonicalStateTransition,
    ) -> AnalysisExecutionResult<StructuralBucklingWorkload> {
        self.validate()?;
        StructuralBucklingWorkload::new(StructuralBucklingWorkloadInput {
            model: self.model,
            step_id: self.step_id,
            backend: self.backend,
            observation_execution: self.observation_execution,
            solver: self.solver,
            prestress_transition,
        })
    }
}

/// Placement-neutral buckling solve bound to one immutable preload transition.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "StructuralBucklingWorkloadWire")]
pub struct StructuralBucklingWorkload {
    schema_version: u16,
    identity: StructuralBucklingWorkloadIdentity,
    model: AnalysisModel,
    step_id: ProcedureStepId,
    backend: ComputeBackend,
    observation_execution: ObservationExecutionPolicy,
    solver: StructuralBucklingSolverPolicy,
    prestress_transition: CanonicalStateTransition,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StructuralBucklingWorkloadWire {
    schema_version: u16,
    identity: StructuralBucklingWorkloadIdentity,
    model: AnalysisModel,
    step_id: ProcedureStepId,
    backend: ComputeBackend,
    observation_execution: ObservationExecutionPolicy,
    solver: StructuralBucklingSolverPolicy,
    prestress_transition: CanonicalStateTransition,
}

pub struct StructuralBucklingWorkloadInput {
    pub model: AnalysisModel,
    pub step_id: ProcedureStepId,
    pub backend: ComputeBackend,
    pub observation_execution: ObservationExecutionPolicy,
    pub solver: StructuralBucklingSolverPolicy,
    pub prestress_transition: CanonicalStateTransition,
}

impl TryFrom<StructuralBucklingWorkloadWire> for StructuralBucklingWorkload {
    type Error = AnalysisExecutionError;

    fn try_from(wire: StructuralBucklingWorkloadWire) -> Result<Self, Self::Error> {
        let workload = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            model: wire.model,
            step_id: wire.step_id,
            backend: wire.backend,
            observation_execution: wire.observation_execution,
            solver: wire.solver,
            prestress_transition: wire.prestress_transition,
        };
        workload.validate()?;
        Ok(workload)
    }
}

impl StructuralBucklingWorkload {
    pub fn new(input: StructuralBucklingWorkloadInput) -> AnalysisExecutionResult<Self> {
        let mut workload = Self {
            schema_version: 1,
            identity: StructuralBucklingWorkloadIdentity::derive(
                "structural-buckling-workload-placeholder",
                &1_u8,
            )?,
            model: input.model,
            step_id: input.step_id,
            backend: input.backend,
            observation_execution: input.observation_execution,
            solver: input.solver,
            prestress_transition: input.prestress_transition,
        };
        workload.identity = workload.derive_identity()?;
        workload.validate()?;
        Ok(workload)
    }

    pub const fn identity(&self) -> StructuralBucklingWorkloadIdentity {
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

    pub const fn solver(&self) -> StructuralBucklingSolverPolicy {
        self.solver
    }

    pub const fn prestress_transition(&self) -> &CanonicalStateTransition {
        &self.prestress_transition
    }

    pub fn procedure_step(&self) -> AnalysisExecutionResult<&ProcedureStep> {
        validate_common(
            self.schema_version,
            &self.model,
            &self.step_id,
            self.backend,
            self.solver,
        )
    }

    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        let step = self.procedure_step()?;
        self.prestress_transition.validate()?;
        let sequence = SequenceIdentity::from_solver_contract(&self.model.sequence)?;
        let consumer = StepIdentity::from_solver_contract(sequence, step)?;
        if self.prestress_transition.consumer != consumer
            || self.prestress_transition.consumer_step_id != self.step_id
        {
            return Err(AnalysisExecutionError::Identity(
                "buckling prestress transition names a different consumer",
            ));
        }
        if self.identity != self.derive_identity()? {
            return Err(AnalysisExecutionError::Identity(
                "structural buckling workload identity does not match its contents",
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            BUCKLING_WORKLOAD_PREFIX,
            "structural-buckling-workload",
            self,
            BUCKLING_WORKLOAD_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        Ok(decode_contract(
            BUCKLING_WORKLOAD_PREFIX,
            "structural-buckling-workload",
            bytes,
            BUCKLING_WORKLOAD_LIMITS,
        )?)
    }

    fn derive_identity(&self) -> AnalysisExecutionResult<StructuralBucklingWorkloadIdentity> {
        StructuralBucklingWorkloadIdentity::derive(
            "structural-buckling-workload",
            &(
                self.schema_version,
                &self.model,
                &self.step_id,
                self.backend,
                self.observation_execution,
                self.solver,
                &self.prestress_transition,
            ),
        )
    }
}

fn validate_common<'a>(
    schema_version: u16,
    model: &'a AnalysisModel,
    step_id: &ProcedureStepId,
    backend: ComputeBackend,
    solver: StructuralBucklingSolverPolicy,
) -> AnalysisExecutionResult<&'a ProcedureStep> {
    if schema_version != 1 || backend != ComputeBackend::Cpu {
        return Err(invalid(
            "structural buckling currently requires schema one and the qualified CPU backend",
        ));
    }
    solver.validate()?;
    validate_model_sequence(model).map_err(|error| invalid(error.to_string()))?;
    let mut matches = model
        .sequence
        .steps
        .iter()
        .filter(|step| step.step_id == *step_id);
    let step = matches
        .next()
        .ok_or_else(|| invalid("structural buckling workload references an absent step"))?;
    if matches.next().is_some() {
        return Err(invalid(
            "structural buckling workload references a duplicate step",
        ));
    }
    let IndependentDomain::Eigenvalue { requested_count } = step.independent_domain else {
        return Err(invalid("structural buckling requires an eigenvalue domain"));
    };
    if step.family != PhysicsFamily::Structural
        || step.procedure != AnalysisProcedure::EigenvalueBuckling
        || step.solution_space != SolutionSpace::Direct
        || requested_count != solver.requested_mode_count()
        || !step.activations.is_empty()
        || !matches!(
            step.state_inputs.as_slice(),
            [input]
                if input.use_as == StateUse::Prestress
                    && input.required_kind == StateKind::GeometricStiffness
                    && matches!(input.source, StateSource::Upstream { .. })
        )
    {
        return Err(invalid(
            "structural buckling requires one direct eigenvalue step and one upstream geometric-stiffness prestress input",
        ));
    }
    Ok(step)
}

fn invalid(message: impl Into<String>) -> AnalysisExecutionError {
    AnalysisExecutionError::Invalid(message.into())
}
