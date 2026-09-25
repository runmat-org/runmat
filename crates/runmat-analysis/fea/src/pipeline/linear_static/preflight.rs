use runmat_analysis_core::{AnalysisModel, ProcedureStep};

use crate::{
    assembly::{explicit_connectors::compile_model_connector_plan, AssemblySummary},
    connectors::CanonicalStructuralConnectorPlan,
    constraints::CanonicalStructuralConstraintPlan,
    contracts::{ComputeBackend, FeaRunError, LinearStaticSolveOptions},
    planning::resource::{build_static_resource_plan, StaticResourcePreflight},
    post::fields::structural_static_observable_field_definitions,
    results::observation::{
        resolve_structural_observation_plan_with_connectors, ObservationExecutionPolicy,
        ObservationLimits,
    },
};

pub(super) struct DirectStaticPreflight {
    pub constraints: CanonicalStructuralConstraintPlan,
    pub connectors: CanonicalStructuralConnectorPlan,
}

/// Direct CPU physical solves have no portable scheduler to admit their resource envelope.
/// Derive topology, constraints, observations, and recovery demand before matrix assembly,
/// then pass the same compiled constraint plan into assembly and result recovery. Portable
/// solves have already revalidated their bound resource plan at the runner boundary. Direct
/// GPU lacks a bound accelerator request and remains outside this admission envelope.
pub(super) fn prepare_direct_static_constraints(
    model: &AnalysisModel,
    step: &ProcedureStep,
    backend: ComputeBackend,
    options: &LinearStaticSolveOptions,
    prepared_plan_present: bool,
) -> Result<Option<DirectStaticPreflight>, FeaRunError> {
    if prepared_plan_present
        || backend != ComputeBackend::Cpu
        || (options.solver_mesh.is_none() && !model_has_explicit_structural_elements(model))
    {
        return Ok(None);
    }
    let constraints =
        CanonicalStructuralConstraintPlan::compile_for_model(model, options.solver_mesh.as_ref())
            .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
    let _properties = crate::properties::compile_property_map(model, options.solver_mesh.as_ref())
        .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
    let connectors = compile_model_connector_plan(
        model,
        options.solver_mesh.as_ref(),
        constraints.reference_points.len(),
    )
    .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
    let fields = structural_static_observable_field_definitions()
        .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
    let observations = resolve_structural_observation_plan_with_connectors(
        step,
        model,
        options.solver_mesh.as_ref(),
        &constraints,
        &connectors,
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
    Ok(Some(DirectStaticPreflight {
        constraints,
        connectors,
    }))
}

pub(super) fn verify_executed_connector_plan(
    direct: Option<&DirectStaticPreflight>,
    portable: Option<&CanonicalStructuralConnectorPlan>,
    summary: &AssemblySummary,
) -> Result<(), FeaRunError> {
    let preassembly_plan = direct.map(|plans| &plans.connectors).or(portable);
    if let Some(plan) = preassembly_plan.filter(|plan| !plan.entries.is_empty()) {
        if summary.structural_connector_plan.as_ref() != Some(plan) {
            return Err(FeaRunError::Assembly(
                "executed connector plan differs from the admitted preassembly plan".into(),
            ));
        }
    }
    Ok(())
}

/// Portable execution already admitted its resource plan. Compile the same connector identity
/// before assembly so a changed numerical target resolution cannot pass on admission alone.
pub(super) fn prepare_portable_static_connectors(
    model: &AnalysisModel,
    options: &LinearStaticSolveOptions,
    prepared_constraints: Option<&CanonicalStructuralConstraintPlan>,
) -> Result<Option<CanonicalStructuralConnectorPlan>, FeaRunError> {
    prepared_constraints
        .filter(|_| !model.connectors.is_empty())
        .map(|plan| {
            compile_model_connector_plan(
                model,
                options.solver_mesh.as_ref(),
                plan.reference_points.len(),
            )
            .map_err(|error| FeaRunError::Assembly(error.to_string()))
        })
        .transpose()
}

pub(super) fn model_has_explicit_structural_elements(model: &AnalysisModel) -> bool {
    model
        .structural
        .as_ref()
        .is_some_and(|structural| !structural.elements.is_empty())
}
