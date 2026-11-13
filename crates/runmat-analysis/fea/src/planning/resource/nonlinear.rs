//! Whole direct nonlinear pipeline preflight. Physical point state is additional
//! persistent storage, not an alternate spelling for a linear recovery reservation.
use super::*;
pub(super) mod block;
mod output;
use crate::{
    assembly::nonlinear_solid::resources::NonlinearSolidStorageDemand,
    solve::nonlinear::NonlinearSolveOptions,
    topology::{CanonicalSolidIntegrationPlan, CanonicalSolidIntegrationRule},
};
pub use block::{
    build_nonlinear_solid_block_resource_plan, nonlinear_solid_block_output_construction_bytes,
    nonlinear_solid_block_payload_bytes_upper_bound,
};

pub(crate) fn build_nonlinear_resource_plan_with_output_limit(
    input: StaticResourcePreflight<'_>,
    options: &NonlinearSolveOptions,
    output_limit: usize,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    if input.prepared_inputs.is_some() {
        return Err(ResourcePlanningError::Invalid(
            "portable nonlinear transport requires its block planner".into(),
        ));
    }
    build_resource_plan(derive_nonlinear_resource_input(
        input,
        options,
        output_limit,
    )?)
}

pub(crate) fn build_nonlinear_recovery_resource_plan(
    input: StaticResourcePreflight<'_>,
    options: &NonlinearSolveOptions,
    output_limit: usize,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    build_resource_plan(derive_resource_input(input, options, output_limit, true)?)
}

fn derive_nonlinear_resource_input(
    input: StaticResourcePreflight<'_>,
    options: &NonlinearSolveOptions,
    output_limit: usize,
) -> Result<FeaResourcePlanningInput, ResourcePlanningError> {
    derive_resource_input(input, options, output_limit, false)
}

fn derive_resource_input(
    input: StaticResourcePreflight<'_>,
    options: &NonlinearSolveOptions,
    output_limit: usize,
    recovery: bool,
) -> Result<FeaResourcePlanningInput, ResourcePlanningError> {
    if output_limit == 0 || output_limit > options.max_accepted_increments {
        return Err(ResourcePlanningError::Invalid(
            "nonlinear retained-output limit exceeds original global cap".into(),
        ));
    }
    let invalid = |e: &dyn std::fmt::Display| ResourcePlanningError::Invalid(e.to_string());
    let mesh = input.mesh.ok_or_else(|| {
        ResourcePlanningError::Invalid("nonlinear preflight requires its canonical mesh".into())
    })?;
    if input.backend != crate::ComputeBackend::Cpu || !input.accelerator_requests.is_empty() {
        return Err(ResourcePlanningError::Invalid(
            "nonlinear execution envelope is accelerator-free CPU".into(),
        ));
    }
    let integration_storage = CanonicalSolidIntegrationPlan::derivation_resident_bytes_estimate(
        mesh,
        CanonicalSolidIntegrationRule::Keast15,
    )
    .map_err(|e| invalid(&e))?;
    let integration =
        CanonicalSolidIntegrationPlan::derive(mesh, CanonicalSolidIntegrationRule::Keast15)
            .map_err(|e| invalid(&e))?;
    let properties = crate::properties::compile_structural_property_map(input.model, Some(mesh))
        .map_err(|e| invalid(&e))?
        .identity;
    let mut input = super::preflight::build_static_resource_input(
        input,
        super::StaticSolvePath::ProvingFactorization,
    )?;
    let FeaResourceDemand::StructuralLinearPipeline(mut baseline) = input.demand else {
        return Err(ResourcePlanningError::Invalid(
            "structural preflight omitted its operator demand".into(),
        ));
    };
    let dofs = usize::try_from(baseline.operator.sparse.rows.0)
        .map_err(|_| ResourcePlanningError::Overflow("nonlinear DOFs"))?;
    let nonzeros = usize::try_from(baseline.operator.sparse.maximum_nonzeros())
        .map_err(|_| ResourcePlanningError::Overflow("nonlinear sparse support"))?;
    let storage = NonlinearSolidStorageDemand::derive(
        &integration,
        &properties,
        dofs,
        nonzeros,
        output_limit,
        options.line_search,
    )
    .map_err(|e| invalid(&e))?;
    // Raw output buffers and the complete carrier coexist with canonical content
    // identity construction. Reserve its materialization, not just f64 payloads.
    let output = output::projection(
        dofs,
        integration.elements().len(),
        output_limit,
        storage.point_history_bytes,
    )?;
    if recovery {
        // Preserve affine-reduction scratch needed by physical validation, but
        // remove Krylov/preconditioner/factorization work: no solve is executed.
        let transpose = baseline.operator.solve.transpose_index_bytes_upper_bound();
        baseline.operator.solve = SolveResourcePolicy::NoSolve {
            workspace_bytes_upper_bound: transpose,
        };
        // Field construction/observations remain in the shared operator policy.
        // Recovery makes no new solver artifact or state-profile catalog.
        baseline.operator.publication.result_artifact_bytes = ByteCount(0);
        baseline.operator.publication.result_egress_bytes = ByteCount(0);
        input.phase = FeaResourcePhase::StructuralNonlinearRecovery;
        input.demand = FeaResourceDemand::StructuralNonlinearRecovery(
            StructuralNonlinearRecoveryResourceDemand {
                constraint_compilation_peak_bytes: baseline.constraint_compilation_peak_bytes,
                operator: baseline.operator,
                integration_identity: integration.identity(),
                point_count: Count(integration.total_point_count() as u64),
                retained_output_increments: Count(output_limit as u64),
                physical_evaluation_limit: Count(2),
                additional_persistent_bytes: ByteCount(
                    storage
                        .recovery_peak_bytes
                        .checked_add(integration_storage)
                        .ok_or(ResourcePlanningError::Overflow(
                            "nonlinear recovery storage",
                        ))?,
                ),
                history_limits: output.history_limits,
            },
        );
        return Ok(input);
    }
    let (output_bound, output_construction) = (output.wire, output.resident);
    baseline.operator.publication.result_artifact_bytes = ByteCount(output_bound);
    baseline.operator.publication.result_egress_bytes = ByteCount(output_bound);
    let limits =
        crate::solve::nonlinear::NonlinearWorkLimits::derive(options).map_err(|e| invalid(&e))?;
    let accepted = u64::try_from(limits.accepted)
        .map_err(|_| ResourcePlanningError::Overflow("nonlinear accepted cap"))?;
    let evaluations = u64::try_from(limits.total_evaluations().map_err(|e| invalid(&e))?)
        .map_err(|_| ResourcePlanningError::Overflow("nonlinear evaluation cap"))?;
    input.phase = FeaResourcePhase::StructuralNonlinearPipeline;
    input.demand =
        FeaResourceDemand::StructuralNonlinearPipeline(StructuralNonlinearPipelineResourceDemand {
            constraint_compilation_peak_bytes: baseline.constraint_compilation_peak_bytes,
            operator: baseline.operator,
            integration_identity: integration.identity(),
            point_count: Count(integration.total_point_count() as u64),
            maximum_accepted_increments: Count(accepted),
            maximum_retained_output_increments: Count(output_limit as u64),
            tangent_evaluation_limit: Count(evaluations),
            linear_solve_limit: Count(
                u64::try_from(limits.linear_solves)
                    .map_err(|_| ResourcePlanningError::Overflow("nonlinear tangent solve cap"))?,
            ),
            additional_persistent_bytes: ByteCount(
                storage
                    .nonlinear_peak_bytes
                    .checked_add(integration_storage)
                    .and_then(|v| v.checked_add(output_construction))
                    .ok_or(ResourcePlanningError::Overflow(
                        "nonlinear persistent storage",
                    ))?,
            ),
            state_profiles_output: baseline.state_profiles_output,
            block_transport: None,
            history_limits: output.history_limits,
        });
    Ok(input)
}
