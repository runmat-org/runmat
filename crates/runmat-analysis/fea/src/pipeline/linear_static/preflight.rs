use runmat_analysis_core::{AnalysisModel, ProcedureStep};

use crate::{
    assembly::{explicit_connectors::compile_model_connector_plan, AssemblySummary},
    connectors::CanonicalStructuralConnectorPlan,
    constraints::CanonicalStructuralConstraintPlan,
    contracts::{ComputeBackend, FeaRunError, LinearStaticSolveOptions},
    planning::resource::{
        build_static_resource_plan, verify_assembled_structural_storage, StaticResourcePreflight,
    },
    post::fields::structural_static_observable_field_definitions,
    results::observation::{
        resolve_structural_observation_plan_with_connectors, ObservationExecutionPolicy,
        ObservationLimits,
    },
};

pub(super) struct StaticPreassembly {
    pub constraints: Option<CanonicalStructuralConstraintPlan>,
    connectors: Option<CanonicalStructuralConnectorPlan>,
}

/// Direct CPU physical solves derive constraints, observations, and recovery demand before
/// assembly. Portable solves have already revalidated their bound resource plan at the worker
/// boundary; both paths retain connector identity for post-assembly reconciliation. Direct GPU
/// lacks a bound accelerator request and remains outside this admission envelope.
pub(super) fn prepare_static_preassembly(
    model: &AnalysisModel,
    step: &ProcedureStep,
    backend: ComputeBackend,
    options: &LinearStaticSolveOptions,
    prepared_constraints: Option<&CanonicalStructuralConstraintPlan>,
) -> Result<StaticPreassembly, FeaRunError> {
    if let Some(constraints) = prepared_constraints {
        let connectors = (!model.connectors.is_empty())
            .then(|| {
                compile_model_connector_plan(
                    model,
                    options.solver_mesh.as_ref(),
                    constraints.reference_points.len(),
                )
                .map_err(|error| FeaRunError::Assembly(error.to_string()))
            })
            .transpose()?;
        return Ok(StaticPreassembly {
            constraints: None,
            connectors,
        });
    }
    if backend != ComputeBackend::Cpu
        || (options.solver_mesh.is_none() && !model_has_explicit_structural_elements(model))
    {
        return Ok(StaticPreassembly {
            constraints: None,
            connectors: None,
        });
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
    Ok(StaticPreassembly {
        constraints: Some(constraints),
        connectors: Some(connectors),
    })
}

pub(super) fn verify_executed_preassembly(
    model: &AnalysisModel,
    options: &LinearStaticSolveOptions,
    constraints: Option<&CanonicalStructuralConstraintPlan>,
    preassembly: &StaticPreassembly,
    summary: &AssemblySummary,
) -> Result<(), FeaRunError> {
    if let Some(plan) = preassembly
        .connectors
        .as_ref()
        .filter(|plan| !plan.entries.is_empty())
    {
        if summary.structural_connector_plan.as_ref() != Some(plan) {
            return Err(FeaRunError::Assembly(
                "executed connector plan differs from the admitted preassembly plan".into(),
            ));
        }
    }
    if let Some(constraints) = constraints {
        verify_assembled_structural_storage(
            model,
            options.solver_mesh.as_ref(),
            constraints,
            summary,
        )?;
    }
    Ok(())
}

pub(super) fn model_has_explicit_structural_elements(model: &AnalysisModel) -> bool {
    model
        .structural
        .as_ref()
        .is_some_and(|structural| !structural.elements.is_empty())
}
