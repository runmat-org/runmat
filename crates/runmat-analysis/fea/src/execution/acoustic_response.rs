//! Portable canonical acoustic cavity-mode and pressure-transient intent.
//!
//! One solve produces a sequence of real nodal pressure samples: mass-normalized mode shapes
//! (mode samples) or the time-domain pressure history (time samples). Numerical execution only;
//! scheduling and resource admission are separate gates.
use super::{
    AnalysisExecutionError, AnalysisExecutionResult, AnalysisWorkloadIdentity, SequenceIdentity,
    StepIdentity,
};
use crate::{
    admission::AnalysisRunIdentity,
    contracts::{validate_solver_formulation_compatibility, SolverMethod, SolverPreconditioner},
    physics::acoustic::{
        solve_canonical_modal, solve_canonical_transient, AcousticModalPolicy,
        AcousticTransientPolicy,
    },
};
use runmat_analysis_core::{
    validate_model, AnalysisModel, AnalysisProcedure, BoundaryTransferImplementation,
    ComplexRepresentation, FieldLocation, ObservationComponent, ObservationQuantity,
    ObservationRequest, ObservationValueShape, PhysicsFamily, ProcedureStep, ProcedureStepId,
    RecoveryPolicy, RestartPolicy, SamplingPolicy, SolutionSpace, SourceImplementation,
    StateCompleteness, StateKind, StatePersistence,
};
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use runmat_meshing_core::SolverMeshArtifact;
use serde::{Deserialize, Serialize};

mod fields;
pub(crate) use fields::pressure_definition as acoustic_response_pressure_definition;
mod publication;
mod receipt;
mod result;
pub use fields::ACOUSTIC_PRESSURE_FIELD_ID;
pub use receipt::{
    AcousticResponseNumericalReceipt, AcousticResponseNumericalReceiptArtifact,
    ACOUSTIC_RESPONSE_RECEIPT_MAX_ENCODED_BYTES,
};
pub use result::{AcousticResponseDiagnostics, AcousticResponseSolverResult};
#[cfg(test)]
mod tests;

const PREFIX: &[u8] = b"runmat-acoustic-response-workload/v1\0";
const LIMITS: CanonicalLimits =
    crate::contracts::canonical_digest::wire_bounded(CanonicalLimits::new(0, 0, 65_536, 64));

/// The procedure-specific numerical controls of one acoustic response solve.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "procedure", deny_unknown_fields)]
pub enum AcousticResponseSolver {
    Modal { policy: AcousticModalPolicy },
    Transient { policy: AcousticTransientPolicy },
}

impl AcousticResponseSolver {
    pub const fn procedure(self) -> AnalysisProcedure {
        match self {
            Self::Modal { .. } => AnalysisProcedure::ModalExtraction,
            Self::Transient { .. } => AnalysisProcedure::TransientResponse,
        }
    }
    pub const fn method(self) -> SolverMethod {
        match self {
            Self::Modal { .. } => SolverMethod::AcousticCanonicalShiftInvertBlockKrylov,
            Self::Transient { .. } => SolverMethod::AcousticCanonicalNewmarkPcg,
        }
    }
    pub const fn state_kind(self) -> StateKind {
        match self {
            Self::Modal { .. } => StateKind::AcousticModes,
            Self::Transient { .. } => StateKind::AcousticPressure,
        }
    }
    /// The accepted bound on each solve's own physical residual.
    pub fn relative_tolerance(self) -> f64 {
        match self {
            Self::Modal { policy } => policy.relative_tolerance,
            Self::Transient { policy } => policy.linear_relative_tolerance,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AcousticResponseWorkload {
    pub model: AnalysisModel,
    pub step_id: ProcedureStepId,
    pub solver: AcousticResponseSolver,
}

impl AcousticResponseWorkload {
    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        self.step().map(|_| ())
    }

    pub(crate) fn step(&self) -> AnalysisExecutionResult<&ProcedureStep> {
        validate_model(&self.model).map_err(invalid)?;
        let [step] = self.model.sequence.steps.as_slice() else {
            return Err(invalid(
                "acoustic response workload requires one standalone authored step",
            ));
        };
        let policy_valid = match self.solver {
            AcousticResponseSolver::Modal { policy } => policy.validate().is_ok(),
            AcousticResponseSolver::Transient { policy } => policy.validate().is_ok(),
        };
        if step.step_id != self.step_id
            || step.family != PhysicsFamily::Acoustic
            || step.procedure != self.solver.procedure()
            || step.solution_space != SolutionSpace::Direct
            || step.restart != RestartPolicy::Fresh
            || !step.state_inputs.is_empty()
            || !step.activations.is_empty()
            || !self.model.sequence.dependencies.is_empty()
            || !self.model.sequence.transitions.is_empty()
            || !self.model.sequence.branches.is_empty()
            || !policy_valid
        {
            return Err(invalid(
                "acoustic response workload requires a fresh direct standalone modal or transient step with valid numerical controls",
            ));
        }
        validate_observations(self.solver, &step.observations)?;
        if matches!(self.solver, AcousticResponseSolver::Transient { .. }) {
            let runmat_analysis_core::IndependentDomain::TimeSeconds {
                start,
                end,
                interval_count,
                ..
            } = &step.independent_domain
            else {
                return Err(invalid("acoustic transient requires an authored time grid"));
            };
            crate::physics::acoustic::TransientRetention::for_step(step)
                .map_err(invalid)?
                .validate_for_domain(*start, *end, *interval_count)
                .map_err(invalid)?;
        }
        if !matches!(
            step.state_outputs.as_slice(),
            [output] if output.kind == self.solver.state_kind()
                && output.completeness == StateCompleteness::Complete
                && output.persistence != StatePersistence::CheckpointOnly
        ) {
            return Err(invalid(
                "acoustic response step must declare exactly its complete pressure state",
            ));
        }
        Ok(step)
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            PREFIX,
            "acoustic-response-workload",
            self,
            LIMITS,
        )?)
    }
    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        let value: Self = decode_contract(PREFIX, "acoustic-response-workload", bytes, LIMITS)?;
        value.validate()?;
        Ok(value)
    }
    pub fn identity(&self) -> AnalysisExecutionResult<AnalysisWorkloadIdentity> {
        AnalysisWorkloadIdentity::derive(
            "acoustic-response-workload",
            &Digest::sha256(self.canonical_bytes()?),
        )
    }
    pub fn producer_identity(&self) -> AnalysisExecutionResult<StepIdentity> {
        StepIdentity::from_solver_contract(
            SequenceIdentity::from_solver_contract(&self.model.sequence)?,
            self.step()?,
        )
    }
    pub fn formulation(
        &self,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<AnalysisRunIdentity> {
        self.validate()?;
        let identity = crate::admission::admit_nonstructural(
            PhysicsFamily::Acoustic,
            self.solver.procedure(),
            Some(mesh),
            SourceImplementation::CanonicalBoundaryProjection,
            BoundaryTransferImplementation::CanonicalBoundaryProjection,
        )
        .map_err(invalid)?;
        validate_solver_formulation_compatibility(
            &identity,
            self.solver.method(),
            SolverPreconditioner::Ilu0,
        )
        .map_err(invalid)?;
        Ok(identity)
    }
    /// Serial numerical execution, not scheduler/resource or publication admission.
    pub fn execute(
        &self,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<AcousticResponseSolverResult> {
        let formulation = self.formulation(mesh)?;
        let response = match self.solver {
            AcousticResponseSolver::Modal { policy } => result::SolvedResponse::Modal(
                solve_canonical_modal(&self.model, mesh, self.step_id.as_str(), policy)
                    .map_err(invalid)?,
            ),
            AcousticResponseSolver::Transient { policy } => result::SolvedResponse::Transient(
                solve_canonical_transient(
                    &self.model,
                    mesh,
                    self.step_id.as_str(),
                    policy,
                    &crate::physics::acoustic::TransientRetention::for_step(self.step()?)
                        .map_err(invalid)?,
                )
                .map_err(invalid)?,
            ),
        };
        AcousticResponseSolverResult::new(self, mesh, formulation, response)
    }
}

fn validate_observations(
    solver: AcousticResponseSolver,
    observations: &[ObservationRequest],
) -> AnalysisExecutionResult<()> {
    if matches!(solver, AcousticResponseSolver::Modal { .. }) && !observations.is_empty() {
        return Err(invalid(
            "acoustic cavity modes do not admit authored observations",
        ));
    }
    for observation in observations {
        let ObservationRequest::History(spec) = observation else {
            return Err(invalid(
                "acoustic pressure transients admit authored history outputs only",
            ));
        };
        if spec.quantity != ObservationQuantity::AcousticPressure
            || spec.value_shape != ObservationValueShape::Scalar
            || spec.source_location != FieldLocation::Node
            || spec.output_location != FieldLocation::Node
            || spec.recovery != RecoveryPolicy::Native
            || spec.component != ObservationComponent::All
            || spec.complex_representation != ComplexRepresentation::Real
            || !matches!(
                &spec.sampling,
                SamplingPolicy::EveryAcceptedIncrement
                    | SamplingPolicy::TimeSeconds(_)
                    | SamplingPolicy::TimeIntervalSeconds { .. }
            )
        {
            return Err(invalid(
                "acoustic transient history requires native nodal scalar pressure sampled over time",
            ));
        }
    }
    Ok(())
}

impl AcousticResponseWorkload {
    /// Rederive the admitted plan from this workload and mesh before any numerical allocation.
    pub fn execute_with_resource_plan(
        &self,
        mesh: &SolverMeshArtifact,
        plan: &crate::planning::resource::FeaResourcePlan,
    ) -> AnalysisExecutionResult<AcousticResponseSolverResult> {
        use crate::planning::resource::{
            build_acoustic_response_resource_plan, FeaResourceDemand,
            PreparedAcousticResponseResourceFacts,
        };
        plan.validate_identity().map_err(invalid)?;
        let FeaResourceDemand::AcousticResponse(demand) = &plan.input.demand else {
            return Err(invalid(
                "acoustic response requires its own admitted resource demand",
            ));
        };
        let output_set = plan
            .input
            .output_set
            .clone()
            .ok_or_else(|| invalid("acoustic response requires its admitted output set"))?;
        let expected = build_acoustic_response_resource_plan(
            self,
            mesh,
            PreparedAcousticResponseResourceFacts {
                local_execution:
                    crate::planning::resource::LocalExecutionPlan::from_cpu_millicores(
                        plan.request.cpu_millicores,
                    )
                    .map_err(invalid)?,
                inputs: demand.inputs.clone(),
                result_output: demand.result_output.clone(),
                receipt_output: demand.receipt_output.clone(),
                state_profiles_output: demand.state_profiles_output.clone(),
                output_set,
            },
        )
        .map_err(invalid)?;
        if &expected != plan {
            return Err(invalid(
                "acoustic response resource plan is not derived from this workload and mesh",
            ));
        }
        self.execute(mesh)
    }
}

fn invalid(error: impl std::fmt::Display) -> AnalysisExecutionError {
    AnalysisExecutionError::Invalid(error.to_string())
}
