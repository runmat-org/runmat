use super::*;
use crate::{execution::ThermalWorkload, planning::resource::ThermalHistoryResourceFacts};

fn workload() -> ThermalWorkload {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.thermal = Some(runmat_analysis_core::ThermalDomain {
        reference_temperature: runmat_analysis_core::Temperature::new(293.15),
    });
    let mut step = ProcedureStep::thermal_transient("thermal_schedule");
    for id in ["one", "two"] {
        step.observations.push(nodal_history_observation(
            id,
            runmat_analysis_core::ObservationQuantity::Temperature,
            runmat_analysis_core::SamplingPolicy::TimeSeconds(vec![0.0, 0.2, 0.65, 1.0]),
        ));
    }
    model.sequence = runmat_analysis_core::AnalysisSequence::from_steps("heat", vec![step]);
    ThermalWorkload {
        step_id: model.sequence.steps[0].step_id.clone(),
        model,
        backend: ComputeBackend::Cpu,
        formulation_requirement: runmat_analysis_core::FormulationRequirement::Experimental,
        procedure: ThermalProcedureControl::BackwardEuler { snapshot_count: 3 },
        residual_target: 1e-4,
    }
}

#[test]
fn preflight_inventory_matches_actual_retained_fields_and_expanded_solve() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mut workload = workload();
        workload.model.sequence.steps[0]
            .accuracy
            .resources
            .maximum_samples = 5;
        let mesh = runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(order);
        let schedule = workload.solve_schedule().unwrap();
        assert_eq!(schedule.points_s(), &[0.0, 0.2, 0.5, 0.65, 1.0]);
        let facts = ThermalHistoryResourceFacts::derive(&mesh, &schedule).unwrap();
        let result = workload.execute(&mesh).unwrap();
        assert_eq!(result.time_points_s, schedule.points_s());
        assert_eq!(result.residual_norms.len(), schedule.solve_count());
        let field_values: usize = result
            .temperature_snapshots
            .iter()
            .chain(&result.temperature_gradient_snapshots)
            .chain(&result.heat_flux_snapshots)
            .chain(&result.energy_storage_rate_snapshots)
            .map(|field| field.as_host_f64().unwrap().len())
            .sum();
        assert_eq!(facts.retained_field_bytes.0, field_values as u64 * 8);
        // A nominal three-snapshot estimate would miss two mandatory recovered snapshots.
        assert_eq!(facts.samples.0, 5);
    }
}

#[test]
fn impossible_sample_or_retained_memory_ceilings_fail_before_thermal_assembly() {
    let mesh = runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let mut workload = workload();
    workload.model.sequence.steps[0]
        .accuracy
        .resources
        .maximum_samples = 4;
    assert!(matches!(
        workload.execute(&mesh),
        Err(crate::FeaRunError::Resource(
            crate::planning::resource::ResourcePlanningError::AccuracyCeiling {
                resource: "thermal solve samples",
                required: 5,
                maximum: 4
            }
        ))
    ));
    workload.model.sequence.steps[0]
        .accuracy
        .resources
        .maximum_samples = 5;
    workload.model.sequence.steps[0]
        .accuracy
        .resources
        .maximum_resident_bytes = 439;
    // Isolate mandatory field storage from the independently validated observation budget.
    workload.model.sequence.steps[0].observations.clear();
    workload.procedure = ThermalProcedureControl::BackwardEuler { snapshot_count: 5 };
    let error = workload.execute(&mesh).unwrap_err();
    assert!(
        matches!(
            error,
            crate::FeaRunError::Resource(
                crate::planning::resource::ResourcePlanningError::AccuracyCeiling {
                    required: 440,
                    maximum: 439,
                    ..
                }
            )
        ),
        "{error}"
    );
    workload.model.sequence.steps[0]
        .accuracy
        .resources
        .maximum_resident_bytes = 1024 * 1024;
    workload.procedure = ThermalProcedureControl::BackwardEuler {
        snapshot_count: usize::MAX,
    };
    assert!(matches!(
        workload.execute(&mesh),
        Err(crate::FeaRunError::Resource(_))
    ));
}

#[test]
fn thermal_observation_preflight_accounts_catalog_and_combined_ceiling() {
    use crate::planning::resource::{
        ThermalObservationResourceFacts, ThermalOperatorResourceFacts,
    };
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mut workload = workload();
        for request in &mut workload.model.sequence.steps[0].observations {
            if let runmat_analysis_core::ObservationRequest::History(spec) = request {
                spec.cost = runmat_analysis_core::ObservationCostPolicy::Automatic;
            }
        }
        let mesh = runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(order);
        let step = &workload.model.sequence.steps[0];
        let observations = ThermalObservationResourceFacts::derive(&mesh, step).unwrap();
        let shared = observations.shared.as_ref().unwrap();
        assert_eq!(shared.retained_sample_count.0, 8);
        assert_eq!(
            observations.catalog_copy_bytes.0,
            (mesh.topology.nodes.len() + 6 * mesh.topology.volume_elements.len()) as u64 * 8
        );
        let result = workload.execute(&mesh).unwrap();
        let retained_samples: usize = result
            .run
            .observation_histories
            .iter()
            .map(|history| history.samples().len())
            .sum();
        assert_eq!(retained_samples as u64, shared.retained_sample_count.0);
        let history =
            ThermalHistoryResourceFacts::derive(&mesh, &workload.solve_schedule().unwrap())
                .unwrap();
        let operators = ThermalOperatorResourceFacts::derive(&mesh).unwrap();
        let numerical = operators.numerical_reservation_bytes(&history).unwrap();
        let total = numerical.0 + observations.reservation_bytes().unwrap().0;
        workload.model.sequence.steps[0]
            .accuracy
            .resources
            .maximum_resident_bytes = total - 1;
        let error = workload.execute(&mesh).unwrap_err();
        assert!(
            matches!(error, crate::FeaRunError::Resource(
            crate::planning::resource::ResourcePlanningError::AccuracyCeiling { required, maximum, .. }
        ) if required == total && maximum == total - 1),
            "{error}"
        );
        workload.model.sequence.steps[0]
            .accuracy
            .resources
            .maximum_resident_bytes = total;
        workload.execute(&mesh).unwrap();
        let mut overflowing = observations;
        overflowing.catalog_copy_bytes = crate::planning::resource::ByteCount(u64::MAX);
        assert!(overflowing.reservation_bytes().is_err());
    }
}

#[test]
fn thermal_observation_preflight_rejects_unavailable_sources_and_events() {
    use crate::planning::resource::ThermalObservationResourceFacts;
    let mesh = runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let mut workload = workload();
    let step = &mut workload.model.sequence.steps[0];
    if let runmat_analysis_core::ObservationRequest::History(spec) = &mut step.observations[0] {
        spec.quantity = runmat_analysis_core::ObservationQuantity::Displacement;
        spec.value_shape = spec.quantity.canonical_value_shape();
    }
    assert!(ThermalObservationResourceFacts::derive(&mesh, step).is_err());
    step.observations.clear();
    assert_eq!(
        ThermalObservationResourceFacts::derive(&mesh, step)
            .unwrap()
            .reservation_bytes()
            .unwrap()
            .0,
        0
    );
    step.observations.push(nodal_history_observation(
        "event",
        runmat_analysis_core::ObservationQuantity::Temperature,
        runmat_analysis_core::SamplingPolicy::Event(
            runmat_analysis_core::ObservationEvent::Solver(
                runmat_analysis_core::SolverObservationEvent::ConvergenceFailure,
            ),
        ),
    ));
    let error = ThermalObservationResourceFacts::derive(&mesh, step).unwrap_err();
    assert!(error.to_string().contains("no event detector"));
}
