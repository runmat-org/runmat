use super::*;
use runmat_analysis_core::{FormulationRequirement, Temperature, ThermalDomain};
use runmat_analysis_fea::execution::{ThermalSolverResultManifest, ThermalWorkload};
use runmat_analysis_fea::{ComputeBackend, ThermalProcedureControl};
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

fn workload(procedure: ThermalProcedureControl) -> ThermalWorkload {
    let mut model = runmat_analysis_fea::fixtures::fixture_model(
        runmat_analysis_fea::fixtures::FixtureId::CantileverLinearStatic,
    );
    model.thermal = Some(ThermalDomain {
        reference_temperature: Temperature::new(293.15),
    });
    let step = match procedure {
        ThermalProcedureControl::Steady => ProcedureStep::thermal_steady("heat"),
        ThermalProcedureControl::BackwardEuler { .. } => ProcedureStep::thermal_transient("heat"),
    };
    model.sequence = AnalysisSequence::from_steps("thermal", vec![step]);
    ThermalWorkload {
        step_id: model.sequence.steps[0].step_id.clone(),
        model,
        backend: ComputeBackend::Cpu,
        formulation_requirement: FormulationRequirement::Experimental,
        procedure,
        residual_target: 1e-4,
    }
}

#[test]
fn presolve_transport_bounds_postsolve_closure_and_preserves_numerical_result() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        for procedure in [
            ThermalProcedureControl::Steady,
            ThermalProcedureControl::BackwardEuler { snapshot_count: 12 },
        ] {
            let mut workload = workload(procedure);
            let mesh = canonical_tetrahedron_solver_mesh(order);
            if procedure == ThermalProcedureControl::Steady {
                use runmat_analysis_core::{
                    AnalysisConstraint, PrescribedValue, RegionTarget, ThermalConstraint,
                    ThermalConstraintKind,
                };
                let projection =
                    runmat_analysis_fea::topology::AnalysisProjection::derive(&mesh).unwrap();
                let face = projection.boundary_faces[0].provenance[0]
                    .analysis_selection_id()
                    .unwrap();
                workload
                    .model
                    .constraints
                    .push(AnalysisConstraint::Thermal(ThermalConstraint {
                        constraint_id: "temperature".into(),
                        kind: ThermalConstraintKind::Prescribed {
                            target: RegionTarget::boundary(face),
                            value: PrescribedValue::Constant(Temperature::new(315.0)),
                        },
                    }));
            }
            let direct = workload.execute(&mesh).unwrap();
            let expected_temperature = if procedure == ThermalProcedureControl::Steady {
                315.0
            } else {
                293.15
            };
            for field in &direct.temperature_snapshots {
                assert!(field
                    .as_host_f64()
                    .unwrap()
                    .iter()
                    .all(|value| (value - expected_temperature).abs() < 1e-8));
            }
            let manifest = ThermalSolverResultManifest::from_result(&direct).unwrap();
            for chunk_bytes in [1024, 8192] {
                let preparation = workload.prepare(&mesh).unwrap();
                let projected = preparation.transport_profile(chunk_bytes).unwrap();
                let actual = manifest.transport_profile(chunk_bytes).unwrap();
                assert_eq!(projected.partition_count, actual.partition_count);
                assert!(projected.chunk_count_upper_bound >= actual.chunk_count_upper_bound);
                assert!(projected.payload_bytes_upper_bound >= actual.payload_bytes_upper_bound);
                let facts = projected_transport_output_resource_facts(&projected).unwrap();
                let post = projected_chunked_output_resource_facts(
                    &manifest.transport_plan(chunk_bytes).unwrap(),
                )
                .unwrap();
                assert!(
                    facts.construction_resident_bytes_admission_budget.0
                        >= post.construction_resident_bytes_admission_budget.0
                );
                let prepared = execute_thermal_solver_result_objects(
                    &workload,
                    &mesh,
                    chunk_bytes,
                    ObjectNamespace::ResultValue,
                    ObjectInventoryLimits::default(),
                )
                .unwrap();
                let expected = prepare_thermal_solver_result_objects(
                    &direct,
                    chunk_bytes,
                    ObjectNamespace::ResultValue,
                    ObjectInventoryLimits::default(),
                )
                .unwrap();
                assert_eq!(prepared, expected);
            }
        }
    }
}

#[test]
fn output_inventory_and_combined_residency_are_rejected_before_execution() {
    let mut workload = workload(ThermalProcedureControl::BackwardEuler { snapshot_count: 3 });
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let prepared = workload.prepare(&mesh).unwrap();
    let profile = prepared.transport_profile(1024).unwrap();
    let output = projected_transport_output_resource_facts(&profile).unwrap();
    let required = prepared.numerical_reservation_bytes().unwrap().0
        + output.construction_resident_bytes_admission_budget.0;
    drop(prepared);
    for limits in [
        ObjectInventoryLimits {
            max_total_bytes: profile.total_bytes_upper_bound - 1,
            ..ObjectInventoryLimits::default()
        },
        ObjectInventoryLimits {
            max_objects: profile.chunk_count_upper_bound as usize,
            ..ObjectInventoryLimits::default()
        },
        ObjectInventoryLimits {
            max_object_bytes: profile.root.encoded_bytes_upper_bound - 1,
            ..ObjectInventoryLimits::default()
        },
    ] {
        let error = execute_thermal_solver_result_objects(
            &workload,
            &mesh,
            1024,
            ObjectNamespace::ResultValue,
            limits,
        )
        .unwrap_err();
        assert!(
            error.to_string().contains("canonical payload projection"),
            "{error}"
        );
    }
    workload.model.sequence.steps[0]
        .accuracy
        .resources
        .maximum_resident_bytes = required - 1;
    let error = execute_thermal_solver_result_objects(
        &workload,
        &mesh,
        1024,
        ObjectNamespace::ResultValue,
        ObjectInventoryLimits::default(),
    )
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("thermal numerical and output residency"),
        "{error}"
    );
    workload.model.sequence.steps[0]
        .accuracy
        .resources
        .maximum_resident_bytes = required;
    execute_thermal_solver_result_objects(
        &workload,
        &mesh,
        1024,
        ObjectNamespace::ResultValue,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
}
