//! Prepared static execution retains the portable worker's exact admission through assembly.

use runmat_analysis_core::AnalysisModel;
use runmat_execution::Digest;

use crate::{
    constraints::CanonicalStructuralConstraintPlan,
    contracts::{ComputeBackend, FeaRunError, FeaRunResult, LinearStaticSolveOptions},
    planning::resource::{FeaResourcePlan, PreparedInputResourceFacts},
    results::observation::ObservationExecutionPolicy,
};

use super::{
    run_linear_static_with_constraint_plan, run_linear_static_with_constraint_plan_owned,
    StaticPreparedInputs,
};

pub async fn run_linear_static_with_prepared_constraint_plan_tet4_owned<O, F, A>(
    model: &AnalysisModel,
    backend: ComputeBackend,
    options: LinearStaticSolveOptions,
    constraint_plan: &CanonicalStructuralConstraintPlan,
    admitted: Option<StaticResourceAdmission<'_>>,
    maximum_block_elements: usize,
    before_block: O,
) -> Result<FeaRunResult, A>
where
    O: FnMut(crate::algebra::NumericalCapacityPhase, runmat_simulation_usage::OperationCount) -> F,
    F: std::future::Future<Output = Result<(), A>>,
    A: From<FeaRunError>
        + From<crate::assembly::LinearAssemblyError>
        + From<crate::assembly::StructuralAssemblyError>
        + From<crate::assembly::solver_solid::SolverSolidAssemblyError>
        + Send,
{
    run_linear_static_with_constraint_plan_owned(
        model,
        backend,
        options,
        StaticPreparedInputs {
            constraint_plan: Some(constraint_plan),
            admitted,
            thermal_source: None,
        },
        maximum_block_elements,
        true,
        before_block,
    )
    .await
    .map(|result| result.run)
}

pub async fn run_linear_static_with_prepared_constraint_plan_and_linearization_tet4_owned<O, F, A>(
    model: &AnalysisModel,
    backend: ComputeBackend,
    options: LinearStaticSolveOptions,
    input: StaticLinearizationInput<'_>,
    admitted: Option<StaticResourceAdmission<'_>>,
    maximum_block_elements: usize,
    before_block: O,
) -> Result<LinearStaticExecution, A>
where
    O: FnMut(crate::algebra::NumericalCapacityPhase, runmat_simulation_usage::OperationCount) -> F,
    F: std::future::Future<Output = Result<(), A>>,
    A: From<FeaRunError>
        + From<crate::assembly::LinearAssemblyError>
        + From<crate::assembly::StructuralAssemblyError>
        + From<crate::assembly::solver_solid::SolverSolidAssemblyError>
        + Send,
{
    let StaticLinearizationInput {
        constraint_plan,
        topology_digest,
    } = input;
    let computation = run_linear_static_with_constraint_plan_owned(
        model,
        backend,
        options,
        StaticPreparedInputs {
            constraint_plan: Some(constraint_plan),
            admitted,
            thermal_source: None,
        },
        maximum_block_elements,
        true,
        before_block,
    )
    .await?;
    let linearization =
        super::super::structural_linearization::StructuralLinearizationArtifact::from_solution(
            &computation.summary,
            computation.run.identity.clone(),
            topology_digest,
            &computation.solution,
        )
        .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
    Ok(LinearStaticExecution {
        run: computation.run,
        linearization,
    })
}

#[derive(Clone, Copy)]
pub struct StaticResourceAdmission<'a> {
    pub plan: &'a FeaResourcePlan,
    /// Worker-derived facts, never extracted from the submitted resource plan.
    pub prepared_inputs: Option<&'a PreparedInputResourceFacts>,
    pub observation_execution: ObservationExecutionPolicy,
}

/// Constraint preparation and topology identity retained for the emitted linearization.
pub struct StaticLinearizationInput<'a> {
    pub constraint_plan: &'a CanonicalStructuralConstraintPlan,
    pub topology_digest: Digest,
}

pub struct LinearStaticExecution {
    pub run: FeaRunResult,
    pub linearization: super::super::structural_linearization::StructuralLinearizationArtifact,
}

/// Executes a linear-static solve with a topology-resolved constraint plan supplied by the
/// portable execution boundary. Inline callers use [`super::run_linear_static_with_options`],
/// which compiles the same plan before assembly.
pub fn run_linear_static_with_prepared_constraint_plan(
    model: &AnalysisModel,
    backend: ComputeBackend,
    options: LinearStaticSolveOptions,
    constraint_plan: &CanonicalStructuralConstraintPlan,
    admitted: Option<StaticResourceAdmission<'_>>,
) -> Result<FeaRunResult, FeaRunError> {
    run_linear_static_with_constraint_plan(
        model,
        backend,
        options,
        Some(constraint_plan),
        admitted,
        None,
    )
    .map(|result| result.run)
}

/// Executes the preload solve and publishes the exact initial-stress tangent from the same
/// assembly and solution. This prevents downstream stability procedures from reconstructing or
/// relabeling a base state.
pub fn run_linear_static_with_prepared_constraint_plan_and_linearization(
    model: &AnalysisModel,
    backend: ComputeBackend,
    options: LinearStaticSolveOptions,
    constraint_plan: &CanonicalStructuralConstraintPlan,
    topology_digest: Digest,
    admitted: Option<StaticResourceAdmission<'_>>,
) -> Result<LinearStaticExecution, FeaRunError> {
    let computation = run_linear_static_with_constraint_plan(
        model,
        backend,
        options,
        Some(constraint_plan),
        admitted,
        None,
    )?;
    let linearization =
        super::super::structural_linearization::StructuralLinearizationArtifact::from_solution(
            &computation.summary,
            computation.run.identity.clone(),
            topology_digest,
            &computation.solution,
        )
        .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
    Ok(LinearStaticExecution {
        run: computation.run,
        linearization,
    })
}
