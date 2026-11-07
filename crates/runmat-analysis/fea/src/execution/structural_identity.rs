use runmat_analysis_core::{AnalysisModel, ProcedureStep};
use runmat_canonical_codec::{encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use runmat_meshing_core::SolverMeshArtifact;
use serde::Serialize;

use crate::assembly::AssemblySummary;

use super::{
    AnalysisExecutionResult, ConstraintPlanIdentity, DiscretizationIdentity, ExecutionPlanIdentity,
    NumericalPlanIdentity, OperatorIdentity, ProblemIdentity, SequenceIdentity,
    StateTransitionIdentity, StepIdentity, StructuralDampingOperatorIdentity,
    StructuralDofLayoutIdentity, StructuralExcitationSetIdentity, StructuralFormulationIdentity,
    StructuralKinematicsIdentity, StructuralMassOperatorIdentity,
    StructuralStiffnessOperatorIdentity, StudyIdentity,
};

const STRUCTURAL_IDENTITY_PREFIX: &[u8] = b"runmat-structural-prepared-identities/v1\0";
const STRUCTURAL_IDENTITY_LIMITS: CanonicalLimits =
    crate::contracts::canonical_digest::wire_bounded(CanonicalLimits::new(0, 0, 65_536, 64));

pub(crate) struct StructuralPreparedIdentities {
    pub study: StudyIdentity,
    pub step: StepIdentity,
    pub discretization: DiscretizationIdentity,
    pub constraint_plan: ConstraintPlanIdentity,
    pub stiffness_operator: StructuralStiffnessOperatorIdentity,
    pub mass_operator: StructuralMassOperatorIdentity,
    pub damping_operator: StructuralDampingOperatorIdentity,
    pub excitations: StructuralExcitationSetIdentity,
    pub dof_layout: StructuralDofLayoutIdentity,
    pub formulation: StructuralFormulationIdentity,
    pub kinematics: StructuralKinematicsIdentity,
    pub execution_plan: ExecutionPlanIdentity,
}

pub(crate) struct StructuralPreparedIdentityInput<'a, E, S, X, K> {
    pub model: &'a AnalysisModel,
    pub step: &'a ProcedureStep,
    pub solver_mesh: &'a SolverMeshArtifact,
    pub assembly: &'a AssemblySummary,
    pub input_state: Option<StateTransitionIdentity>,
    pub excitation_contract: &'a E,
    pub solver_contract: &'a S,
    pub execution_contract: &'a X,
    pub kinematics_contract: &'a K,
}

pub(crate) fn derive_structural_prepared_identities<E, S, X, K>(
    input: StructuralPreparedIdentityInput<'_, E, S, X, K>,
) -> AnalysisExecutionResult<StructuralPreparedIdentities>
where
    E: Serialize,
    S: Serialize,
    X: Serialize,
    K: Serialize,
{
    let study = StudyIdentity::from_authored_contract(input.model, None)?;
    let sequence = SequenceIdentity::from_solver_contract(&input.model.sequence)?;
    let step = StepIdentity::from_solver_contract(sequence, input.step)?;
    let mesh_digest = Digest::from_bytes(*input.solver_mesh.canonical_digest.bytes());
    let constraint_digest = structural_contract_digest(
        "constraint-plan",
        &(
            &input.assembly.constraint_matrix,
            &input.assembly.affine_constraint_reduction,
            input.assembly.operator.constrained_dofs(),
        ),
    )?;
    let constraint_plan = ConstraintPlanIdentity::new(step, mesh_digest, constraint_digest)?;
    let excitations =
        StructuralExcitationSetIdentity::from_authored_contract(step, input.excitation_contract)?;
    let problem = ProblemIdentity::from_structural_contract(
        step,
        structural_contract_digest("model", input.model)?,
        input.input_state,
        constraint_plan,
        excitations,
    )?;
    let property_identity = input
        .assembly
        .structural_property_map_identity
        .as_ref()
        .ok_or_else(|| {
            super::AnalysisExecutionError::Invalid(
                "structural assembly has no compiled property-map identity".into(),
            )
        })?;
    let formulation_digest = structural_contract_digest(
        "formulation-and-properties",
        &(&input.assembly.structural_formulations, property_identity),
    )?;
    let discretization = DiscretizationIdentity::new(problem, mesh_digest, formulation_digest)?;
    let operators = input.assembly.operator.operators();
    let stiffness = OperatorIdentity::new(
        discretization,
        operators
            .stiffness()
            .numerical_identity()
            .map_err(|error| super::AnalysisExecutionError::Invalid(error.to_string()))?,
    )?;
    let mass = OperatorIdentity::new(
        discretization,
        operators
            .mass()
            .numerical_identity()
            .map_err(|error| super::AnalysisExecutionError::Invalid(error.to_string()))?,
    )?;
    let damping = OperatorIdentity::new(
        discretization,
        operators
            .damping()
            .numerical_identity()
            .map_err(|error| super::AnalysisExecutionError::Invalid(error.to_string()))?,
    )?;
    let combined_operator = OperatorIdentity::new(
        discretization,
        structural_contract_digest("second-order-operators", &(stiffness, mass, damping))?,
    )?;
    let numerical_plan = NumericalPlanIdentity::new(
        combined_operator,
        structural_contract_digest("solver", input.solver_contract)?,
    )?;
    let execution_plan = ExecutionPlanIdentity::new(
        numerical_plan,
        structural_contract_digest("execution", input.execution_contract)?,
    )?;
    Ok(StructuralPreparedIdentities {
        study,
        step,
        discretization,
        constraint_plan,
        stiffness_operator: StructuralStiffnessOperatorIdentity::new(stiffness)?,
        mass_operator: StructuralMassOperatorIdentity::new(mass)?,
        damping_operator: StructuralDampingOperatorIdentity::new(damping)?,
        excitations,
        dof_layout: StructuralDofLayoutIdentity::new(
            discretization,
            structural_contract_digest(
                "structural-dof-layout",
                &input.assembly.structural_dof_layout,
            )?,
        )?,
        formulation: StructuralFormulationIdentity::new(discretization, formulation_digest)?,
        kinematics: StructuralKinematicsIdentity::new(
            constraint_plan,
            excitations,
            structural_contract_digest("structural-kinematics", input.kinematics_contract)?,
        )?,
        execution_plan,
    })
}

pub(crate) fn structural_contract_digest<T: Serialize>(
    domain: &'static str,
    value: &T,
) -> AnalysisExecutionResult<Digest> {
    Ok(Digest::sha256(encode_contract(
        STRUCTURAL_IDENTITY_PREFIX,
        domain,
        value,
        STRUCTURAL_IDENTITY_LIMITS,
    )?))
}
