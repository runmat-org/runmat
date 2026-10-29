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
    // The authored ceiling is part of the encoded model, so resident requirements are
    // measured under a non-binding authored ceiling (present, like the ceilings tests author).
    model.sequence.steps[0]
        .accuracy
        .resources
        .maximum_resident_bytes = Some(runmat_types::wire_integer::MAX_WIRE_INTEGER);
    ThermalWorkload {
        initial_fields: Vec::new(),
        step_id: model.sequence.steps[0].step_id.clone(),
        model,
        backend: ComputeBackend::Cpu,
        procedure: ThermalProcedureControl::BackwardEuler { snapshot_count: 3 },
        residual_target: 1e-4,
    }
}

#[test]
fn final_thermal_field_selection_is_validated_before_assembly() {
    use crate::planning::resource::ThermalObservationResourceFacts;
    use runmat_analysis_core::{
        ExcitationTarget, ObservationRequest, ObservationTarget, RegionTarget, SamplingPolicy,
    };
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(order);
        let mut workload = workload();
        let ObservationRequest::History(mut spec) =
            workload.model.sequence.steps[0].observations[0].clone()
        else {
            unreachable!()
        };
        spec.sampling = SamplingPolicy::Final;
        workload.model.sequence.steps[0].observations =
            vec![ObservationRequest::Field(spec.clone())];
        let facts =
            ThermalObservationResourceFacts::derive(&mesh, &workload.model.sequence.steps[0])
                .unwrap();
        assert!(
            facts.shared.is_none(),
            "final fields are not inline histories"
        );
        workload.prepare(&mesh).unwrap();
        spec.target = ObservationTarget::Spatial(ExcitationTarget::Region(RegionTarget::volume(
            "absent-volume",
        )));
        workload.model.sequence.steps[0].observations = vec![ObservationRequest::Field(spec)];
        assert!(
            ThermalObservationResourceFacts::derive(&mesh, &workload.model.sequence.steps[0])
                .is_err()
        );
        assert!(workload.prepare(&mesh).is_err());
    }
}

#[test]
fn deferred_thermal_observations_preserve_primary_fields_and_authored_identity() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let workload = workload();
        let mesh = runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(order);
        let primary = workload
            .prepare(&mesh)
            .unwrap()
            .execute_primary_fields()
            .unwrap();
        assert!(primary.run.observation_histories.is_empty());
        assert!(primary.run.series_reductions.is_empty());
        assert_eq!(primary.source_workload, workload.identity().unwrap());
        assert_eq!(
            primary.time_points_s,
            workload.solve_schedule().unwrap().points_s()
        );
        let inline = workload.execute(&mesh).unwrap();
        assert_eq!(inline.run.observation_histories.len(), 2);
        assert_eq!(primary.producer, inline.producer);
        // This is execution-product parity, not an independent numerical reference.
        assert_eq!(
            crate::execution::ThermalSolverResultManifest::from_result(&primary).unwrap(),
            crate::execution::ThermalSolverResultManifest::from_result(&inline).unwrap(),
        );
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
        let facts = ThermalHistoryResourceFacts::derive(&mesh, &schedule, &[]).unwrap();
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
        .maximum_resident_bytes = Some(439);
    // Isolate mandatory field storage from the independently validated observation budget.
    workload.model.sequence.steps[0].observations.clear();
    workload.procedure = ThermalProcedureControl::BackwardEuler { snapshot_count: 5 };
    let history =
        ThermalHistoryResourceFacts::derive(&mesh, &workload.solve_schedule().unwrap(), &[])
            .unwrap();
    let error = history.require_fits(439).unwrap_err();
    assert!(
        matches!(
            error,
            crate::planning::resource::ResourcePlanningError::AccuracyCeiling {
                required: 440,
                maximum: 439,
                ..
            }
        ),
        "{error}"
    );
    // The whole workload now rejects its larger input closure before history allocation.
    assert!(matches!(
        workload.execute(&mesh),
        Err(crate::FeaRunError::Resource(
            crate::planning::resource::ResourcePlanningError::AccuracyCeiling {
                resource: "thermal preassembly components",
                required,
                maximum: 439,
            }
        )) if required > 440
    ));
    workload.model.sequence.steps[0]
        .accuracy
        .resources
        .maximum_resident_bytes = Some(1024 * 1024);
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
        ThermalObservationResourceFacts, ThermalPreassemblyResourceFacts,
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
        let result = workload.execute(&mesh).unwrap();
        let source_sample_bytes: usize = [
            &result.temperature_snapshots[0],
            &result.temperature_gradient_snapshots[0],
            &result.heat_flux_snapshots[0],
            &result.energy_storage_rate_snapshots[0],
        ]
        .iter()
        .map(|field| std::mem::size_of_val(field.as_host_f64().unwrap()))
        .sum();
        assert_eq!(
            observations.catalog_copy_bytes.0,
            source_sample_bytes as u64
        );
        let retained_samples: usize = result
            .run
            .observation_histories
            .iter()
            .map(|history| history.samples().len())
            .sum();
        assert_eq!(retained_samples as u64, shared.retained_sample_count.0);
        let total = ThermalPreassemblyResourceFacts::derive(
            &[],
            &workload.model,
            step,
            &mesh,
            &workload.solve_schedule().unwrap(),
            &crate::admission::admit_nonstructural(
                runmat_analysis_core::PhysicsFamily::Thermal,
                workload.procedure.procedure(),
                Some(&mesh),
                runmat_analysis_core::SourceImplementation::CanonicalVolumeProjection,
                runmat_analysis_core::BoundaryTransferImplementation::CanonicalBoundaryProjection,
            )
            .unwrap(),
        )
        .unwrap()
        .reservation_bytes()
        .unwrap()
        .0;
        workload.model.sequence.steps[0]
            .accuracy
            .resources
            .maximum_resident_bytes = Some(total - 1);
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
            .maximum_resident_bytes = Some(total);
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

#[test]
fn deferred_primary_fields_do_not_reserve_unexecuted_dynamic_histories() {
    let mesh = runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let mut input = workload();
    // Preserve the authored dynamic cap. The native numerical task defers observation
    // execution to separately admitted tasks; it cannot invoke an inline history.
    input.model.sequence.steps[0]
        .accuracy
        .resources
        .maximum_samples = 1_000_000;
    // An authored memory ceiling: the default is unbounded (machine capacity applies at placement).
    input.model.sequence.steps[0]
        .accuracy
        .resources
        .maximum_resident_bytes = Some(8 << 30);
    for request in &mut input.model.sequence.steps[0].observations {
        if let runmat_analysis_core::ObservationRequest::History(spec) = request {
            spec.sampling = runmat_analysis_core::SamplingPolicy::EveryAcceptedIncrement;
            spec.cost = runmat_analysis_core::ObservationCostPolicy::Automatic;
        }
    }
    assert!(input.prepare(&mesh).is_err());
    let prepared = input.prepare_primary_fields(&mesh).unwrap();
    assert!(
        prepared.numerical_reservation_bytes().unwrap().0
            < input.model.sequence.steps[0]
                .accuracy
                .resources
                .resident_byte_budget()
    );
    let original = input.identity().unwrap();
    let result = prepared.execute_primary_fields().unwrap();
    assert_eq!(result.source_workload, original);
    assert!(result.run.observation_histories.is_empty());
}
