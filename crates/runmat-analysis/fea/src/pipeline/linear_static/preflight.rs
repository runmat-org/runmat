use runmat_analysis_core::{AnalysisModel, FieldLocation, ObservationTarget, ProcedureStep};

use crate::{
    constraints::CanonicalStructuralConstraintPlan,
    contracts::{ComputeBackend, FeaRunError, LinearStaticSolveOptions},
    planning::resource::{build_static_resource_plan, StaticResourcePreflight},
    post::fields::structural_observable_field_definitions,
    results::observation::{
        resolve_structural_observation_plan, ObservationExecutionPolicy, ObservationLimits,
    },
};

/// Direct CPU physical solves have no portable scheduler to admit their resource envelope.
/// Derive topology, constraints, observations, and recovery demand before matrix assembly,
/// then pass the same compiled constraint plan into assembly and result recovery. Portable
/// solves have already revalidated their bound resource plan at the runner boundary. Direct
/// GPU lacks a bound accelerator request; connector-scoped observations need a connector-aware
/// preassembly compiler. These remain outside this admission envelope.
pub(super) fn prepare_direct_static_constraints(
    model: &AnalysisModel,
    step: &ProcedureStep,
    backend: ComputeBackend,
    options: &LinearStaticSolveOptions,
    prepared_plan_present: bool,
) -> Result<Option<CanonicalStructuralConstraintPlan>, FeaRunError> {
    if prepared_plan_present
        || backend != ComputeBackend::Cpu
        || (!model.connectors.is_empty() && has_connector_observation(step))
        || (options.solver_mesh.is_none() && !super::model_has_explicit_structural_elements(model))
    {
        return Ok(None);
    }
    let constraints =
        CanonicalStructuralConstraintPlan::compile_for_model(model, options.solver_mesh.as_ref())
            .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
    let _properties = crate::properties::compile_property_map(model, options.solver_mesh.as_ref())
        .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
    let fields = structural_observable_field_definitions()
        .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
    let observations = resolve_structural_observation_plan(
        step,
        model,
        options.solver_mesh.as_ref(),
        &constraints,
        &fields,
        ObservationLimits::default(),
    )
    .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
    build_static_resource_plan(StaticResourcePreflight {
        model,
        step,
        backend,
        preconditioner: options.preconditioner_kind,
        observation_execution: ObservationExecutionPolicy::serial(),
        mesh: options.solver_mesh.as_ref(),
        constraints: &constraints,
        observations: &observations,
        accelerator_requests: &[],
        local_execution: options.local_execution,
    })?;
    Ok(Some(constraints))
}

fn has_connector_observation(step: &ProcedureStep) -> bool {
    step.observations.iter().any(|request| {
        request.spec().is_some_and(|spec| {
            matches!(spec.target, ObservationTarget::Connector { .. })
                || spec.source_location == FieldLocation::Connector
                || spec.output_location == FieldLocation::Connector
        })
    })
}
