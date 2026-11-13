use super::*;
use crate::{
    planning::resource::{
        build_nonlinear_recovery_resource_plan, build_nonlinear_resource_plan_with_output_limit,
        FeaResourcePlan, LocalExecutionPlan, StaticResourcePreflight,
    },
    post::fields::structural_static_observable_field_definitions,
    results::observation::{
        resolve_structural_observation_plan, ObservationExecutionPolicy, ObservationLimits,
    },
    solve::preconditioner::SpdPreconditionerKind,
};
use runmat_analysis_core::ProcedureStep;
#[cfg(test)]
mod tests;

pub(super) fn tangent_admission(
    plan: &FeaResourcePlan,
) -> Result<&crate::planning::resource::SolverAdmissionResourcePolicy, FeaRunError> {
    plan.input
        .structural_operator()
        .and_then(|operator| match &operator.solve {
            crate::planning::resource::SolveResourcePolicy::Iterative { admission, .. } => {
                Some(admission)
            }
            _ => None,
        })
        .ok_or_else(|| {
            FeaRunError::InvalidModel("nonlinear plan omitted tangent solver admission".into())
        })
}

pub(super) fn prepare(
    model: &AnalysisModel,
    step: &ProcedureStep,
    backend: ComputeBackend,
    options: &NonlinearSolveOptions,
) -> Result<FeaResourcePlan, FeaRunError> {
    prepare_with_output_limit(
        model,
        step,
        backend,
        options,
        options.max_accepted_increments,
    )
}

pub(super) fn prepare_with_output_limit(
    model: &AnalysisModel,
    step: &ProcedureStep,
    backend: ComputeBackend,
    options: &NonlinearSolveOptions,
    output_limit: usize,
) -> Result<FeaResourcePlan, FeaRunError> {
    prepare_for(model, step, backend, options, output_limit, false)
}

pub(super) fn prepare_recovery(
    model: &AnalysisModel,
    step: &ProcedureStep,
    backend: ComputeBackend,
    options: &NonlinearSolveOptions,
    output_limit: usize,
) -> Result<FeaResourcePlan, FeaRunError> {
    prepare_for(model, step, backend, options, output_limit, true)
}

fn prepare_for(
    model: &AnalysisModel,
    step: &ProcedureStep,
    backend: ComputeBackend,
    options: &NonlinearSolveOptions,
    output_limit: usize,
    recovery: bool,
) -> Result<FeaResourcePlan, FeaRunError> {
    crate::solve::nonlinear::require_fixed_structural_nonlinear(model)?;
    let compilation = crate::pipeline::constraint_preparation::compile_inventory(
        model,
        options.solver_mesh.as_ref(),
        step,
    )?;
    let constraints = compilation.plan();
    let fields = structural_static_observable_field_definitions()
        .map_err(|e| FeaRunError::Assembly(e.to_string()))?;
    let observations = resolve_structural_observation_plan(
        step,
        model,
        options.solver_mesh.as_ref(),
        constraints,
        &fields,
        ObservationLimits::default(),
    )
    .map_err(|e| FeaRunError::Observation(e.to_string()))?;
    let build = if recovery {
        build_nonlinear_recovery_resource_plan
    } else {
        build_nonlinear_resource_plan_with_output_limit
    };
    Ok(build(
        StaticResourcePreflight {
            compilation: Some(&compilation),
            model,
            step,
            backend,
            preconditioner: SpdPreconditionerKind::Auto,
            observation_execution: ObservationExecutionPolicy::serial(),
            mesh: options.solver_mesh.as_ref(),
            constraints,
            observations: &observations,
            accelerator_requests: &[],
            local_execution: LocalExecutionPlan::serial(),
            prepared_inputs: None,
        },
        options,
        output_limit,
    )?)
}
