use runmat_analysis_core::{
    validate_model_sequence, AnalysisModel, AnalysisProcedure, FormulationRequirement,
    PhysicsFamily, ProcedureStep, ProcedureStepId, SolutionSpace,
};
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};
use serde::{Deserialize, Serialize};

use crate::{
    assembly::AssemblySummary,
    contracts::ComputeBackend,
    solve::transient::{StructuralTimeStepPolicy, TransientSolveOptions},
};

use super::{
    AnalysisExecutionError, AnalysisExecutionResult, AnalysisWorkKind, AnalysisWorkloadIdentity,
    ConstraintPlanIdentity, DiscretizationIdentity, ExecutionPlanIdentity, NumericalPlanIdentity,
    OperatorIdentity, ProblemIdentity, SequenceIdentity, StepIdentity,
    StructuralDampingOperatorIdentity, StructuralDofLayoutIdentity,
    StructuralExcitationSetIdentity, StructuralFormulationIdentity, StructuralKinematicsIdentity,
    StructuralMassOperatorIdentity, StructuralStiffnessOperatorIdentity,
    StructuralTransientCheckpointScope, StructuralTransientIntegrator,
    StructuralTransientPrecision, StructuralTransientSchedule,
    StructuralTransientSolverOptionsIdentity, StudyIdentity,
};

const TRANSIENT_WORKLOAD_PREFIX: &[u8] = b"runmat-structural-transient-workload/v1\0";
const TRANSIENT_WORKLOAD_LIMITS: CanonicalLimits =
    CanonicalLimits::new(64 * 1024 * 1024, 1_000_000, 65_536, 64);
const TRANSIENT_IDENTITY_PREFIX: &[u8] = b"runmat-structural-transient-identity/v1\0";
const TRANSIENT_IDENTITY_LIMITS: CanonicalLimits =
    CanonicalLimits::new(64 * 1024 * 1024, 1_000_000, 65_536, 64);

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
    pub formulation_requirement: FormulationRequirement,
    pub time_step_policy: StructuralTimeStepPolicy,
    pub residual_target: f64,
}

impl StructuralTransientWorkload {
    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        if self.schema_version != 1
            || !self.residual_target.is_finite()
            || self.residual_target <= 0.0
            || self.time_step_policy.interval_count() == 0
        {
            return Err(AnalysisExecutionError::Invalid(
                "structural transient workload has invalid schema or solver options".into(),
            ));
        }
        validate_model_sequence(&self.model)
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        let step = self.procedure_step()?;
        if step.procedure != AnalysisProcedure::TransientResponse
            || step.solution_space != SolutionSpace::Direct
            || step.family != PhysicsFamily::Structural
        {
            return Err(AnalysisExecutionError::Invalid(
                "structural transient workload requires a direct structural transient-response step"
                    .into(),
            ));
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
            formulation_requirement: self.formulation_requirement,
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

pub struct StructuralTransientCheckpointScopeInput<'a> {
    pub workload: &'a StructuralTransientWorkload,
    pub solver_mesh: &'a SolverMeshArtifact,
    pub initial_assembly: &'a AssemblySummary,
    pub schedule: StructuralTransientSchedule,
}

impl StructuralTransientCheckpointScope {
    pub fn from_prepared_workload(
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
        let formulation_digest = contract_digest(
            "formulation",
            &input.initial_assembly.structural_formulations,
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
                input.workload.formulation_requirement,
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
        let source_workload = AnalysisWorkloadIdentity::new(
            study,
            step,
            execution_plan,
            AnalysisWorkKind::SolveStructuralTransientBlock,
            Digest::sha256(&workload_bytes),
            vec![mesh_digest],
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
