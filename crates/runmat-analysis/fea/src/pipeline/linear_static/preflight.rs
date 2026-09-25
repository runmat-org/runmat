use runmat_analysis_core::{AnalysisModel, ProcedureStep};

use crate::{
    assembly::{explicit_connectors::compile_model_connector_plan, AssemblySummary},
    connectors::CanonicalStructuralConnectorPlan,
    constraints::{CanonicalStructuralConstraintPlan, ConstraintEnforcement},
    contracts::{ComputeBackend, FeaRunError, LinearStaticSolveOptions},
    planning::resource::{
        build_static_resource_plan, verify_assembled_structural_storage, FeaResourceDemand,
        SolveResourcePolicy, SolverAdmissionResourcePolicy, StaticResourcePreflight,
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
    pub solver_admission: Option<SolverAdmissionResourcePolicy>,
}

/// Direct and prepared CPU physical solves derive observations and recovery demand before
/// assembly. Prepared execution retains its worker-verified constraint identity and independently
/// derives the same numerical solver envelope. Direct GPU lacks a bound accelerator request and
/// remains outside this admission envelope.
pub(super) fn prepare_static_preassembly(
    model: &AnalysisModel,
    step: &ProcedureStep,
    backend: ComputeBackend,
    options: &LinearStaticSolveOptions,
    prepared_constraints: Option<&CanonicalStructuralConstraintPlan>,
) -> Result<StaticPreassembly, FeaRunError> {
    if backend != ComputeBackend::Cpu
        || (options.solver_mesh.is_none() && !model_has_explicit_structural_elements(model))
    {
        let connectors = prepared_constraints
            .filter(|_| !model.connectors.is_empty())
            .map(|constraints| {
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
            solver_admission: None,
        });
    }
    let compiled_constraints = prepared_constraints
        .is_none()
        .then(|| {
            CanonicalStructuralConstraintPlan::compile_for_model(
                model,
                options.solver_mesh.as_ref(),
            )
            .map_err(|error| FeaRunError::Assembly(error.to_string()))
        })
        .transpose()?;
    let constraints = prepared_constraints
        .or(compiled_constraints.as_ref())
        .expect("CPU physical preassembly has compiled or prepared constraints");
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
        constraints,
        &connectors,
        &fields,
        ObservationLimits::default(),
    )
    .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
    let resource_plan = build_static_resource_plan(StaticResourcePreflight {
        model,
        step,
        backend,
        preconditioner: options.preconditioner_kind,
        observation_execution: ObservationExecutionPolicy::serial(),
        mesh: options.solver_mesh.as_ref(),
        constraints,
        observations: &observations,
        accelerator_requests: &[],
        local_execution: options.local_execution,
    })?;
    // The multiplier branch uses a saddle-point GMRES solve rather than an SPD coercivity
    // certificate. Its independent workspace reconciliation is still a separate gate.
    let solver_admission = match &resource_plan.input.demand {
        FeaResourceDemand::StructuralLinearPipeline(demand) => match &demand.solve {
            SolveResourcePolicy::Iterative { admission, .. }
                if constraints.enforcement != ConstraintEnforcement::LagrangeMultiplier =>
            {
                Some(admission.clone())
            }
            _ => None,
        },
        _ => None,
    };
    Ok(StaticPreassembly {
        constraints: compiled_constraints,
        connectors: Some(connectors),
        solver_admission,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::{fixture_model, FixtureId};

    #[test]
    fn prepared_and_direct_physical_static_solve_admit_the_same_exact_work_envelope() {
        let model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
        let step = &model.sequence.steps[0];
        let options = LinearStaticSolveOptions::default();
        let constraints = CanonicalStructuralConstraintPlan::compile_for_model(&model, None)
            .expect("physical beam constraint plan");
        let direct = prepare_static_preassembly(&model, step, ComputeBackend::Cpu, &options, None)
            .expect("direct resource admission");
        let prepared = prepare_static_preassembly(
            &model,
            step,
            ComputeBackend::Cpu,
            &options,
            Some(&constraints),
        )
        .expect("prepared resource admission");
        assert_eq!(direct.constraints.as_ref(), Some(&constraints));
        assert!(prepared.constraints.is_none());
        assert_eq!(direct.solver_admission, prepared.solver_admission);
        assert!(prepared.solver_admission.is_some());
    }

    #[test]
    fn prepared_physical_static_rejects_memory_ceiling_before_assembly() {
        let mut model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
        model.sequence.steps[0]
            .accuracy
            .resources
            .maximum_resident_bytes = 1;
        let constraints = CanonicalStructuralConstraintPlan::compile_for_model(&model, None)
            .expect("physical beam constraint plan");
        let error = prepare_static_preassembly(
            &model,
            &model.sequence.steps[0],
            ComputeBackend::Cpu,
            &LinearStaticSolveOptions::default(),
            Some(&constraints),
        )
        .err()
        .expect("prepared physical solve must admit its resources before assembly");
        assert!(matches!(
            error,
            FeaRunError::Resource(
                crate::planning::resource::ResourcePlanningError::AccuracyCeiling {
                    resource: "resident memory bytes",
                    maximum: 1,
                    ..
                }
            )
        ));
    }
}
