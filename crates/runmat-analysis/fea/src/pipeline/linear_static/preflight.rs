use runmat_analysis_core::{AnalysisModel, ProcedureStep};

use crate::{
    assembly::{explicit_connectors::compile_model_connector_plan, AssemblySummary},
    connectors::CanonicalStructuralConnectorPlan,
    constraints::CanonicalStructuralConstraintPlan,
    contracts::{ComputeBackend, FeaRunError, LinearStaticSolveOptions},
    planning::resource::{
        build_static_resource_plan, verify_assembled_structural_storage, FeaResourceDemand,
        ResourcePlanningError, SolveResourcePolicy, SolverAdmissionResourcePolicy,
        StaticResourcePreflight,
    },
    post::fields::structural_static_observable_field_definitions,
    results::observation::{
        resolve_structural_observation_plan_with_connectors, ObservationExecutionPolicy,
        ObservationLimits,
    },
};

use super::StaticResourceAdmission;

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
    admitted: Option<StaticResourceAdmission<'_>>,
) -> Result<StaticPreassembly, FeaRunError> {
    if backend != ComputeBackend::Cpu
        || (options.solver_mesh.is_none() && !model_has_explicit_structural_elements(model))
    {
        if admitted.is_some() {
            return Err(ResourcePlanningError::Invalid(
                "admitted static plan cannot bypass CPU physical preassembly".into(),
            )
            .into());
        }
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
        observation_execution: admitted
            .as_ref()
            .map_or(ObservationExecutionPolicy::serial(), |admission| {
                admission.observation_execution
            }),
        mesh: options.solver_mesh.as_ref(),
        constraints,
        observations: &observations,
        accelerator_requests: &[],
        local_execution: options.local_execution,
    })?;
    if admitted.is_some_and(|admission| admission.plan != &resource_plan) {
        return Err(ResourcePlanningError::Invalid(
            "numerical static preassembly differs from the admitted resource plan".into(),
        )
        .into());
    }
    // Both PCG and saddle-point GMRES prove a prepared SPD operator (reduced where needed).
    // The multiplier's Krylov workspace still needs measured reconciliation.
    let solver_admission = match &resource_plan.input.demand {
        FeaResourceDemand::StructuralLinearPipeline(demand) => match &demand.solve {
            SolveResourcePolicy::Iterative { admission, .. }
            | SolveResourcePolicy::SaddlePointGmres { admission, .. } => Some(admission.clone()),
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
    fn prepared_static_rejects_a_substituted_worker_plan_before_assembly() {
        let model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
        let step = &model.sequence.steps[0];
        let options = LinearStaticSolveOptions::default();
        let constraints = CanonicalStructuralConstraintPlan::compile_for_model(&model, None)
            .expect("physical beam constraint plan");
        let connectors =
            compile_model_connector_plan(&model, None, constraints.reference_points.len())
                .expect("canonical connector plan");
        let fields = structural_static_observable_field_definitions().unwrap();
        let observations = resolve_structural_observation_plan_with_connectors(
            step,
            &model,
            None,
            &constraints,
            &connectors,
            &fields,
            ObservationLimits::default(),
        )
        .unwrap();
        let plan = build_static_resource_plan(StaticResourcePreflight {
            model: &model,
            step,
            backend: ComputeBackend::Cpu,
            preconditioner: options.preconditioner_kind,
            observation_execution: ObservationExecutionPolicy::serial(),
            mesh: None,
            constraints: &constraints,
            observations: &observations,
            accelerator_requests: &[],
            local_execution: options.local_execution,
        })
        .unwrap();
        let admitted = StaticResourceAdmission {
            plan: &plan,
            observation_execution: ObservationExecutionPolicy::serial(),
        };
        prepare_static_preassembly(
            &model,
            step,
            ComputeBackend::Cpu,
            &options,
            Some(&constraints),
            Some(admitted),
        )
        .expect("exact worker plan must converge at numerical preassembly");

        let mut substituted_input = plan.input.clone();
        substituted_input.worker.fixed_memory_bytes.0 += 1;
        let substituted =
            crate::planning::resource::build_resource_plan(substituted_input).unwrap();
        assert_ne!(substituted.identity, plan.identity);
        let error = prepare_static_preassembly(
            &model,
            step,
            ComputeBackend::Cpu,
            &options,
            Some(&constraints),
            Some(StaticResourceAdmission {
                plan: &substituted,
                observation_execution: ObservationExecutionPolicy::serial(),
            }),
        )
        .err()
        .expect("substituted worker plan must fail before assembly");
        assert!(matches!(
            error,
            FeaRunError::Resource(ResourcePlanningError::Invalid(_))
        ));
    }

    #[test]
    fn prepared_and_direct_physical_static_solve_admit_the_same_exact_work_envelope() {
        let model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
        let step = &model.sequence.steps[0];
        let options = LinearStaticSolveOptions::default();
        let constraints = CanonicalStructuralConstraintPlan::compile_for_model(&model, None)
            .expect("physical beam constraint plan");
        let direct =
            prepare_static_preassembly(&model, step, ComputeBackend::Cpu, &options, None, None)
                .expect("direct resource admission");
        let prepared = prepare_static_preassembly(
            &model,
            step,
            ComputeBackend::Cpu,
            &options,
            Some(&constraints),
            None,
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
            None,
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
