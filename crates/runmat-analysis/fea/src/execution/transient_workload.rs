use runmat_analysis_core::{
    validate_model_sequence, AnalysisModel, AnalysisProcedure, PhysicsFamily, ProcedureStep,
    ProcedureStepId, SolutionSpace, StateKind, StateSource, StateUse,
};
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};
use serde::{Deserialize, Serialize};

use crate::{
    assembly::AssemblySummary,
    contracts::ComputeBackend,
    solve::{
        dynamics::ModalReducedSystem,
        transient::{StructuralTimeStepPolicy, TransientSolveOptions},
    },
};

use super::{
    AnalysisExecutionError, AnalysisExecutionResult, AnalysisWorkKind, AnalysisWorkloadIdentity,
    CanonicalSemanticState, CanonicalStateTransition, ConstraintPlanIdentity,
    DiscretizationIdentity, ExecutionPlanIdentity, NumericalPlanIdentity, OperatorIdentity,
    ProblemIdentity, SequenceIdentity, StateIdentity, StepIdentity,
    StructuralDampingOperatorIdentity, StructuralDofLayoutIdentity,
    StructuralExcitationSetIdentity, StructuralFormulationIdentity, StructuralKinematicsIdentity,
    StructuralMassOperatorIdentity, StructuralStiffnessOperatorIdentity,
    StructuralTransientCheckpointScope, StructuralTransientCoordinateSpace,
    StructuralTransientIntegrator, StructuralTransientPrecision, StructuralTransientSchedule,
    StructuralTransientSolverOptionsIdentity, StudyIdentity,
};

const TRANSIENT_WORKLOAD_PREFIX: &[u8] = b"runmat-structural-transient-workload/v2\0";
const TRANSIENT_WORKLOAD_LIMITS: CanonicalLimits =
    crate::contracts::canonical_digest::wire_bounded(CanonicalLimits::new(0, 0, 65_536, 64));
const TRANSIENT_IDENTITY_PREFIX: &[u8] = b"runmat-structural-transient-identity/v1\0";
const TRANSIENT_IDENTITY_LIMITS: CanonicalLimits =
    crate::contracts::canonical_digest::wire_bounded(CanonicalLimits::new(0, 0, 65_536, 64));

/// Numerical coordinate space and typed upstream state for structural transient response.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", deny_unknown_fields)]
pub enum StructuralTransientSolverPlan {
    Direct,
    ModalReduced {
        basis_transition: Box<CanonicalStateTransition>,
    },
}

impl StructuralTransientSolverPlan {
    pub const fn solution_space(&self) -> SolutionSpace {
        match self {
            Self::Direct => SolutionSpace::Direct,
            Self::ModalReduced { .. } => SolutionSpace::ModalReduced,
        }
    }

    pub const fn basis_state(&self) -> Option<StateIdentity> {
        match self {
            Self::Direct => None,
            Self::ModalReduced { basis_transition } => Some(basis_transition.source_state),
        }
    }

    pub const fn basis_transition(&self) -> Option<&CanonicalStateTransition> {
        match self {
            Self::Direct => None,
            Self::ModalReduced { basis_transition } => Some(basis_transition),
        }
    }
}

/// Portable authored input shared by every temporal block in one structural transient solve.
///
/// The solver mesh remains a separate numerical object so workers can cache and transfer it
/// independently. Block bounds and predecessor state remain in the block request/checkpoint
/// objects; changing either therefore never changes this root workload.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralTransientWorkload {
    pub schema_version: u16,
    pub model: AnalysisModel,
    pub step_id: ProcedureStepId,
    pub backend: ComputeBackend,
    pub solver: StructuralTransientSolverPlan,
    pub time_step_policy: StructuralTimeStepPolicy,
    pub residual_target: f64,
    pub initial_fields: Vec<super::InitialFieldBinding>,
}

impl StructuralTransientWorkload {
    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        if self.schema_version != 2
            || !self.residual_target.is_finite()
            || self.residual_target <= 0.0
            || self.time_step_policy.interval_count() == 0
        {
            return Err(AnalysisExecutionError::Invalid(
                "structural transient workload has invalid schema or solver options".into(),
            ));
        }
        if matches!(
            self.solver,
            StructuralTransientSolverPlan::ModalReduced { .. }
        ) && self.backend != ComputeBackend::Cpu
        {
            return Err(AnalysisExecutionError::Invalid(
                "modal transient execution currently requires the CPU backend".into(),
            ));
        }
        validate_model_sequence(&self.model)
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        let step = self.procedure_step()?;
        crate::planning::resource::require_initial_field_ceiling(
            &self.initial_fields,
            TRANSIENT_WORKLOAD_LIMITS,
            step.accuracy.resources.resident_byte_budget(),
        )
        .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        super::validate_initial_field_declarations(&self.model, step, &self.initial_fields)?;

        if step.procedure != AnalysisProcedure::TransientResponse
            || step.solution_space != self.solver.solution_space()
            || step.family != PhysicsFamily::Structural
        {
            return Err(AnalysisExecutionError::Invalid(
                "structural transient workload requires a matching structural transient-response step"
                    .into(),
            ));
        }
        match (&self.solver, step.state_inputs.as_slice()) {
            (StructuralTransientSolverPlan::Direct, []) => {}
            (StructuralTransientSolverPlan::ModalReduced { basis_transition }, [input])
                if input.use_as == StateUse::ModalBasis
                    && input.required_kind == StateKind::ModalBasis
                    && matches!(input.source, StateSource::Upstream { .. }) =>
            {
                basis_transition.validate()?;
                let StateSource::Upstream {
                    producer_step,
                    state_id,
                } = &input.source
                else {
                    unreachable!()
                };
                if basis_transition.producer_step_id != *producer_step
                    || basis_transition.consumer_step_id != step.step_id
                    || basis_transition.state_id != *state_id
                    || basis_transition.method != runmat_analysis_core::StateTransitionMethod::Reuse
                {
                    return Err(AnalysisExecutionError::Identity(
                        "modal transient basis transition differs from its authored state input",
                    ));
                }
            }
            _ => {
                return Err(AnalysisExecutionError::Invalid(
                    "structural transient solver plan differs from its authored state inputs"
                        .into(),
                ));
            }
        }
        Ok(())
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
                "structural transient workload references absent step `{}`",
                self.step_id.as_str()
            ))
        })?;
        if matches.next().is_some() {
            return Err(AnalysisExecutionError::Invalid(format!(
                "structural transient workload references duplicate step `{}`",
                self.step_id.as_str()
            )));
        }
        Ok(step)
    }

    pub fn solve_options(&self, solver_mesh: SolverMeshArtifact) -> TransientSolveOptions {
        TransientSolveOptions {
            time_step_policy: self.time_step_policy,
            residual_target: self.residual_target,
            progress_operation: "fea.solve_structural_transient_block".into(),
            solver_mesh: Some(solver_mesh),
        }
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            TRANSIENT_WORKLOAD_PREFIX,
            "structural-transient-workload",
            self,
            TRANSIENT_WORKLOAD_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        let workload: Self = decode_contract(
            TRANSIENT_WORKLOAD_PREFIX,
            "structural-transient-workload",
            bytes,
            TRANSIENT_WORKLOAD_LIMITS,
        )?;
        workload.validate()?;
        Ok(workload)
    }
}

pub(crate) struct StructuralTransientCheckpointScopeInput<'a> {
    pub workload: &'a StructuralTransientWorkload,
    pub solver_mesh: &'a SolverMeshArtifact,
    pub initial_assembly: &'a AssemblySummary,
    pub modal_state: Option<&'a CanonicalSemanticState>,
    pub schedule: StructuralTransientSchedule,
}

impl StructuralTransientCheckpointScope {
    pub(crate) fn from_prepared_workload(
        input: StructuralTransientCheckpointScopeInput<'_>,
    ) -> AnalysisExecutionResult<Self> {
        input.workload.validate()?;
        input
            .solver_mesh
            .validate_canonical()
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        input.schedule.validate()?;
        if input.initial_assembly.dof_count == 0 {
            return Err(AnalysisExecutionError::Invalid(
                "structural transient checkpoint scope requires physical assembled DOFs".into(),
            ));
        }
        let model = &input.workload.model;
        let step_contract = input.workload.procedure_step()?;
        let study = StudyIdentity::from_authored_contract(model, None)?;
        let sequence = SequenceIdentity::from_solver_contract(&model.sequence)?;
        let step = StepIdentity::from_solver_contract(sequence, step_contract)?;
        let mesh_digest = Digest::from_bytes(*input.solver_mesh.canonical_digest.bytes());
        let (coordinate_space, modal_dependency) = match &input.workload.solver {
            StructuralTransientSolverPlan::Direct => {
                if input.modal_state.is_some() {
                    return Err(AnalysisExecutionError::Invalid(
                        "direct structural transient scope received a modal-basis state".into(),
                    ));
                }
                (StructuralTransientCoordinateSpace::Direct, None)
            }
            StructuralTransientSolverPlan::ModalReduced { basis_transition } => {
                let basis_state = basis_transition.source_state;
                let state = input.modal_state.ok_or_else(|| {
                    AnalysisExecutionError::Invalid(
                        "modal transient scope is missing its exact modal-basis state".into(),
                    )
                })?;
                state.validate()?;
                if state.identity != basis_state
                    || state.kind != StateKind::ModalBasis
                    || state.topology_digest != mesh_digest
                    || basis_transition.target_topology_digest != mesh_digest
                {
                    return Err(AnalysisExecutionError::Identity(
                        "modal transient basis state differs from its workload or solver mesh",
                    ));
                }
                let basis = state.payload.modal_basis().ok_or_else(|| {
                    AnalysisExecutionError::Invalid(
                        "modal transient state does not contain a modal basis".into(),
                    )
                })?;
                let reduced = ModalReducedSystem::project(input.initial_assembly, basis).map_err(
                    |error| match error {
                        crate::solve::dynamics::ModalProjectionError::Usage(error) => {
                            AnalysisExecutionError::Usage(error)
                        }
                        other => AnalysisExecutionError::Invalid(other.to_string()),
                    },
                )?;
                (
                    StructuralTransientCoordinateSpace::ModalReduced {
                        basis_state,
                        basis: basis.identity(),
                        projection: reduced.evidence().clone(),
                        coordinate_count: reduced.mode_count(),
                    },
                    Some([
                        basis_transition.source_state.digest(),
                        basis_transition.identity.digest(),
                    ]),
                )
            }
        };
        let constraint_digest = contract_digest(
            "constraint-plan",
            &(
                &input.initial_assembly.constraint_matrix,
                &input.initial_assembly.affine_constraint_reduction,
                input.initial_assembly.operator.constrained_dofs(),
            ),
        )?;
        let constraint_plan = ConstraintPlanIdentity::new(step, mesh_digest, constraint_digest)?;
        let excitations = StructuralExcitationSetIdentity::from_authored_contract(
            step,
            &(
                &model.signals,
                &model.excitations,
                &model.excitation_cases,
                &input.schedule,
            ),
        )?;
        let model_digest = contract_digest("model", model)?;
        let problem = ProblemIdentity::from_structural_contract(
            step,
            model_digest,
            None,
            constraint_plan,
            excitations,
        )?;
        let property_identity = input
            .initial_assembly
            .structural_property_map_identity
            .as_ref()
            .ok_or_else(|| {
                AnalysisExecutionError::Invalid(
                    "transient structural assembly has no compiled property-map identity".into(),
                )
            })?;
        let formulation_digest = contract_digest(
            "formulation-and-properties",
            &(
                &input.initial_assembly.structural_formulations,
                property_identity,
            ),
        )?;
        let discretization = DiscretizationIdentity::new(problem, mesh_digest, formulation_digest)?;
        let operators = input.initial_assembly.operator.operators();
        let stiffness_operator = OperatorIdentity::new(
            discretization,
            operators
                .stiffness()
                .numerical_identity()
                .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?,
        )?;
        let mass_operator = OperatorIdentity::new(
            discretization,
            operators
                .mass()
                .numerical_identity()
                .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?,
        )?;
        let damping_operator = OperatorIdentity::new(
            discretization,
            operators
                .damping()
                .numerical_identity()
                .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?,
        )?;
        let solver_options_digest = contract_digest(
            "solver-options",
            &(
                &input.workload.solver,
                input.workload.time_step_policy,
                input.workload.residual_target,
            ),
        )?;
        let combined_operator = OperatorIdentity::new(
            discretization,
            contract_digest(
                "second-order-operators",
                &(stiffness_operator, mass_operator, damping_operator),
            )?,
        )?;
        let numerical_plan = NumericalPlanIdentity::new(combined_operator, solver_options_digest)?;
        let execution_plan = ExecutionPlanIdentity::new(
            numerical_plan,
            contract_digest("temporal-block-execution", &(1_u16, "accepted-boundary"))?,
        )?;
        let workload_bytes = input.workload.canonical_bytes()?;
        let mut dependencies = vec![mesh_digest];
        dependencies.extend(modal_dependency.into_iter().flatten());
        let source_workload = AnalysisWorkloadIdentity::new(
            study,
            step,
            execution_plan,
            AnalysisWorkKind::SolveStructuralTransientBlock,
            Digest::sha256(&workload_bytes),
            dependencies,
        )?;
        let dof_layout = StructuralDofLayoutIdentity::new(
            discretization,
            contract_digest(
                "structural-dof-layout",
                &input.initial_assembly.structural_dof_layout,
            )?,
        )?;
        let formulation = StructuralFormulationIdentity::new(discretization, formulation_digest)?;
        let kinematics = StructuralKinematicsIdentity::new(
            constraint_plan,
            excitations,
            contract_digest(
                "structural-kinematics",
                &(&model.constraints, &input.schedule),
            )?,
        )?;
        Ok(Self {
            study,
            step,
            execution_plan,
            source_workload,
            numerical_plan,
            discretization,
            constraint_plan,
            stiffness_operator: StructuralStiffnessOperatorIdentity::new(stiffness_operator)?,
            mass_operator: StructuralMassOperatorIdentity::new(mass_operator)?,
            damping_operator: StructuralDampingOperatorIdentity::new(damping_operator)?,
            excitations,
            dof_layout,
            formulation,
            kinematics,
            dof_count: input.initial_assembly.dof_count,
            coordinate_space,
            integrator: StructuralTransientIntegrator::NewmarkAverageAcceleration,
            precision: StructuralTransientPrecision::Real64,
            solver_options: StructuralTransientSolverOptionsIdentity::new(solver_options_digest)?,
            schedule: input.schedule,
        })
    }
}

fn contract_digest<T: Serialize>(
    domain: &'static str,
    value: &T,
) -> AnalysisExecutionResult<Digest> {
    Ok(Digest::sha256(encode_contract(
        TRANSIENT_IDENTITY_PREFIX,
        domain,
        value,
        TRANSIENT_IDENTITY_LIMITS,
    )?))
}

#[cfg(test)]
mod tests;
