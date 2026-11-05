use runmat_analysis_core::{AnalysisModel, CoupledPhysicsFamily, PhysicsFamily, ProcedureStep};

use crate::{
    assembly::{explicit_connectors::compile_model_connector_plan, AssemblySummary},
    connectors::CanonicalStructuralConnectorPlan,
    constraints::CanonicalStructuralConstraintPlan,
    contracts::{ComputeBackend, FeaRunError, LinearStaticSolveOptions},
    planning::resource::{
        build_static_resource_plan_selecting, structural_resource_derivation_work_upper_bound,
        verify_assembled_structural_storage, FeaResourceDemand, ResourcePlanningError,
        SolveResourcePolicy, SolverAdmissionResourcePolicy, StaticResourcePreflight,
        StaticSolvePath, StaticSolveSelection,
    },
    post::fields::structural_static_observable_field_definitions,
    results::observation::{
        resolve_structural_observation_plan_with_connectors,
        structural_observation_derivation_work_upper_bound, ObservationError,
        ObservationExecutionPolicy, ObservationLimits,
    },
};
use runmat_meshing_core::SolverMeshArtifact;
use runmat_simulation_usage::{LedgerError, OperationCount};

use super::StaticResourceAdmission;

pub(super) fn resource_projection_work_upper_bound(
    mesh: Option<&SolverMeshArtifact>,
) -> Result<OperationCount, FeaRunError> {
    structural_resource_derivation_work_upper_bound(mesh).map_err(FeaRunError::from)
}

pub(super) fn structural_linear_preparation_work_upper_bound(
    mesh: Option<&SolverMeshArtifact>,
) -> Result<OperationCount, FeaRunError> {
    let observation =
        structural_observation_derivation_work_upper_bound(mesh).map_err(|error| match error {
            ObservationError::Accounting(cause) => FeaRunError::Usage(cause),
            other => FeaRunError::Assembly(other.to_string()),
        })?;
    observation
        .checked_add(resource_projection_work_upper_bound(mesh)?)
        .map_err(|overflow| FeaRunError::Usage(LedgerError::from(overflow)))
}

pub(super) struct StaticPreassembly {
    pub constraints: Option<CanonicalStructuralConstraintPlan>,
    connectors: Option<CanonicalStructuralConnectorPlan>,
    pub solver_admission: Option<SolverAdmissionResourcePolicy>,
    /// The planned solve path (`None` for synthetic operators, which have no plan).
    pub solve_path: Option<StaticSolvePath>,
}

/// Direct and prepared CPU physical solves derive observations and recovery demand before
/// assembly. Prepared execution retains its worker-verified constraint identity and independently
/// derives the same numerical solver envelope. The caller admits accelerator placement first
/// (`accelerator_admission`), so this boundary only sees the host backend.
pub(super) fn prepare_static_preassembly(
    model: &AnalysisModel,
    step: &ProcedureStep,
    backend: ComputeBackend,
    options: &LinearStaticSolveOptions,
    prepared_constraints: Option<&CanonicalStructuralConstraintPlan>,
    admitted: Option<StaticResourceAdmission<'_>>,
) -> Result<StaticPreassembly, FeaRunError> {
    debug_assert_eq!(
        backend,
        ComputeBackend::Cpu,
        "static preassembly runs after accelerator admission resolved the host backend"
    );
    if options.solver_mesh.is_none() && !model_has_explicit_structural_elements(model) {
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
            solve_path: None,
        });
    }
    let compilation = crate::pipeline::constraint_preparation::compile_inventory(
        model,
        options.solver_mesh.as_ref(),
        step,
    )?;
    if prepared_constraints.is_some_and(|plan| plan != compilation.plan()) {
        return Err(FeaRunError::Assembly(
            "prepared constraints differ from original compilation".into(),
        ));
    }
    let constraints = compilation.plan();
    let properties =
        crate::properties::compile_structural_property_map(model, options.solver_mesh.as_ref())
            .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
    if step.family == PhysicsFamily::Coupled(CoupledPhysicsFamily::Thermomechanical) {
        properties
            .validate_thermal_solid_capabilities()
            .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
    }
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
    .map_err(|error| match error {
        ObservationError::Accounting(cause) => FeaRunError::Usage(cause),
        other => FeaRunError::Assembly(other.to_string()),
    })?;
    // An admitted plan fixes the path; a direct run selects it within the authored ceiling and
    // the memory a single solve may use on this host.
    let selection = match admitted.as_ref() {
        Some(admission) => {
            match crate::planning::resource::recorded_static_solve_path(admission.plan) {
                Some(path) => StaticSolveSelection::Recorded(path),
                None => {
                    return Err(ResourcePlanningError::Invalid(
                        "admitted static plan records no solve path".into(),
                    )
                    .into())
                }
            }
        }
        None => StaticSolveSelection::WithinCeiling {
            machine_memory_bytes: crate::solve::shared_sparse::default_memory_budget(),
        },
    };
    let resource_plan = build_static_resource_plan_selecting(
        StaticResourcePreflight {
            compilation: Some(&compilation),
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
            prepared_inputs: admitted.and_then(|admission| admission.prepared_inputs),
        },
        selection,
    )?;
    if admitted.is_some_and(|admission| admission.plan != &resource_plan) {
        return Err(ResourcePlanningError::Invalid(
            "numerical static preassembly differs from the admitted resource plan".into(),
        )
        .into());
    }
    // Both PCG and saddle-point GMRES prove a prepared SPD operator (reduced where needed).
    // The multiplier's Krylov workspace still needs measured reconciliation.
    let solver_admission = match &resource_plan.input.demand {
        FeaResourceDemand::StructuralLinearPipeline(demand) => match &demand.operator.solve {
            SolveResourcePolicy::Iterative { admission, .. }
            | SolveResourcePolicy::SaddlePointGmres { admission, .. } => Some(admission.clone()),
            _ => None,
        },
        _ => None,
    };
    Ok(StaticPreassembly {
        constraints: if prepared_constraints.is_none() {
            Some(compilation.into_plan())
        } else {
            None
        },
        connectors: Some(connectors),
        solver_admission,
        solve_path: crate::planning::resource::recorded_static_solve_path(&resource_plan),
    })
}

pub(super) fn verify_executed_preassembly(
    model: &AnalysisModel,
    options: &LinearStaticSolveOptions,
    constraints: Option<&CanonicalStructuralConstraintPlan>,
    preassembly: &StaticPreassembly,
    summary: &AssemblySummary,
) -> Result<Option<crate::algebra::SparsePattern>, FeaRunError> {
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
    constraints
        .map(|constraints| {
            verify_assembled_structural_storage(
                model,
                options.solver_mesh.as_ref(),
                constraints,
                summary,
            )
        })
        .transpose()
        .map(Option::flatten)
        .map_err(FeaRunError::from)
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
    use crate::planning::resource::build_static_resource_plan;
    use runmat_analysis_core::{AnalysisProcedure, AnalysisSequence};

    /// Diagnostic: the static resource plan of a structured Tet10 cantilever as it refines.
    #[test]
    #[ignore]
    fn report_static_plan_growth_for_refined_tet10_cantilever() {
        let model = fixture_model(FixtureId::CantileverLinearStatic);
        let step = &model.sequence.steps[0];
        for divisions in [2usize, 4, 6, 8, 10] {
            let mesh = crate::adaptation::structural_recovery::qualification_tests::mesh::structured_cantilever(
                divisions,
                runmat_meshing_core::ElementOrder::Tet10,
            );
            let compilation = crate::pipeline::constraint_preparation::compile_inventory(
                &model,
                Some(&mesh),
                step,
            )
            .unwrap();
            let constraints = compilation.plan();
            let connectors = compile_model_connector_plan(
                &model,
                Some(&mesh),
                constraints.reference_points.len(),
            )
            .unwrap();
            let fields = structural_static_observable_field_definitions().unwrap();
            let observations = resolve_structural_observation_plan_with_connectors(
                step,
                &model,
                Some(&mesh),
                constraints,
                &connectors,
                &fields,
                ObservationLimits::default(),
            )
            .unwrap();
            let plan = build_static_resource_plan(StaticResourcePreflight {
                compilation: Some(&compilation),
                model: &model,
                step,
                backend: ComputeBackend::Cpu,
                preconditioner: LinearStaticSolveOptions::default().preconditioner_kind,
                observation_execution: ObservationExecutionPolicy::serial(),
                mesh: Some(&mesh),
                constraints,
                observations: &observations,
                accelerator_requests: &[],
                local_execution: LinearStaticSolveOptions::default().local_execution,
                prepared_inputs: None,
            })
            .unwrap();
            eprintln!(
                "divisions={divisions} nodes={} elements={} memory={} scratch={}\n{:#?}",
                mesh.topology.nodes.len(),
                mesh.topology.volume_elements.len(),
                plan.request.memory_bytes,
                plan.request.scratch_bytes,
                plan.breakdown
            );
        }
    }

    #[test]
    fn physical_static_rejects_missing_solid_capabilities_before_resource_planning() {
        let model = fixture_model(FixtureId::CantileverLinearStatic);
        let options = LinearStaticSolveOptions {
            solver_mesh: Some(crate::tests::single_tetrahedron_solver_mesh_for_model(
                &model,
            )),
            ..LinearStaticSolveOptions::default()
        };
        prepare_static_preassembly(
            &model,
            &model.sequence.steps[0],
            ComputeBackend::Cpu,
            &options,
            None,
            None,
        )
        .expect("mechanical solid is valid before assembly");

        let mut thermal_only = model.clone();
        thermal_only.materials[0].elasticity = None;
        let error = prepare_static_preassembly(
            &thermal_only,
            &thermal_only.sequence.steps[0],
            ComputeBackend::Cpu,
            &options,
            None,
            None,
        )
        .err()
        .expect("thermal-only solid must fail before assembly");
        assert!(matches!(
            error,
            FeaRunError::Assembly(message)
                if message.contains("structural solid")
                    && message.contains("IsotropicLinearElasticity")
        ));

        let mut massless = model;
        massless.materials[0].mass_density_kg_per_m3 =
            (None).map(runmat_types::physical::MassDensity::new);
        let error = prepare_static_preassembly(
            &massless,
            &massless.sequence.steps[0],
            ComputeBackend::Cpu,
            &options,
            None,
            None,
        )
        .err()
        .expect("massless solid must fail before assembly");
        assert!(matches!(
            error,
            FeaRunError::Assembly(message)
                if message.contains("structural solid")
                    && message.contains("MassDensity")
        ));
    }

    #[test]
    fn coupled_static_rejects_missing_thermal_capability_before_resource_planning() {
        let mut model = fixture_model(FixtureId::CantileverLinearStatic);
        model.sequence = AnalysisSequence::from_steps(
            "thermomechanical",
            vec![ProcedureStep::coupled(
                "static",
                CoupledPhysicsFamily::Thermomechanical,
                AnalysisProcedure::Static,
            )],
        );
        let options = LinearStaticSolveOptions {
            solver_mesh: Some(crate::tests::single_tetrahedron_solver_mesh_for_model(
                &model,
            )),
            ..LinearStaticSolveOptions::default()
        };
        prepare_static_preassembly(
            &model,
            &model.sequence.steps[0],
            ComputeBackend::Cpu,
            &options,
            None,
            None,
        )
        .expect("capable coupled solid passes preassembly");
        model.materials[0].thermal = None;
        let error = prepare_static_preassembly(
            &model,
            &model.sequence.steps[0],
            ComputeBackend::Cpu,
            &options,
            None,
            None,
        )
        .err()
        .expect("missing thermal capability must fail before planning");
        assert!(matches!(
            error,
            FeaRunError::Assembly(message)
                if message.contains("thermal solid") && message.contains("Thermal")
        ));
    }

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
        let facts = crate::planning::resource::PreparedInputResourceFacts {
            encoded_payload_bytes: crate::planning::resource::ByteCount(1024),
            transport_closure_bytes: crate::planning::resource::ByteCount(2048),
            decoded_resident_bytes_admission_budget: crate::planning::resource::ByteCount(8192),
        };
        let plan = build_static_resource_plan(StaticResourcePreflight {
            compilation: None,
            prepared_inputs: Some(&facts),
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
            prepared_inputs: Some(&facts),
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

        // Recomputed facts must match exactly, including larger valid transport envelopes.
        // Copying facts from the submitted plan would incorrectly accept these substitutions.
        let mut changed_facts = facts.clone();
        changed_facts.transport_closure_bytes.0 += 1;
        for prepared_inputs in [None, Some(&changed_facts)] {
            assert!(matches!(
                prepare_static_preassembly(
                    &model,
                    step,
                    ComputeBackend::Cpu,
                    &options,
                    Some(&constraints),
                    Some(StaticResourceAdmission {
                        prepared_inputs,
                        ..admitted
                    }),
                ),
                Err(FeaRunError::Resource(ResourcePlanningError::Invalid(_)))
            ));
        }

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
                prepared_inputs: Some(&facts),
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
    fn direct_physical_static_applies_authored_compilation_ceiling_before_lowering() {
        let mut model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
        model.constraints.push(model.constraints[0].clone());
        model.sequence.steps[0]
            .accuracy
            .resources
            .maximum_resident_bytes = Some(1);
        let error = prepare_static_preassembly(
            &model,
            &model.sequence.steps[0],
            ComputeBackend::Cpu,
            &LinearStaticSolveOptions::default(),
            None,
            None,
        )
        .err()
        .expect("direct preparation must reject before duplicate-source lowering");
        assert!(matches!(
            error,
            FeaRunError::Resource(ResourcePlanningError::AccuracyCeiling {
                resource: "resident memory bytes",
                required,
                maximum: 1,
            }) if required > 1
        ));
    }

    #[test]
    fn prepared_physical_static_rejects_memory_ceiling_before_assembly() {
        let mut model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
        model.sequence.steps[0]
            .accuracy
            .resources
            .maximum_resident_bytes = Some(1);
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
                    required,
                    maximum: 1,
                }
            ) if required > 1
        ));
    }
}
