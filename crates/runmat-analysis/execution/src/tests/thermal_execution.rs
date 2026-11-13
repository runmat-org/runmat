use super::*;
use runmat_analysis_core::{Temperature, ThermalDomain};
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
        initial_fields: Vec::new(),
        step_id: model.sequence.steps[0].step_id.clone(),
        model,
        backend: ComputeBackend::Cpu,
        procedure,
        residual_target: 1e-4,
    }
}

#[test]
fn thermal_accuracy_binds_numerical_workload_and_topology_not_host_identity() {
    use runmat_analysis_fea::execution::{AccuracySourceKind, SequenceIdentity};
    let workload = workload(ThermalProcedureControl::BackwardEuler { snapshot_count: 3 });
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let limits = ObjectInventoryLimits::default();
    let task = |workload: ThermalWorkload, mesh| {
        prepare_thermal_task(ThermalTaskPreparation {
            maximum_resident_bytes: None,
            local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
            field_source: None,
            authored_study_digest: Digest::sha256("thermal-accuracy"),
            frozen_project_digest: None,
            program_revision: revision(),
            artifact_access: ArtifactAccess {
                authorization_scope: "thermal-accuracy".into(),
                encryption_context: Digest::sha256("thermal-accuracy"),
            },
            workload,
            mesh,
            output_policy: AnalysisOutputPolicy::new(1024, limits).unwrap(),
            required_capabilities: Default::default(),
        })
        .unwrap()
    };
    let prepared = task(workload.clone(), mesh.clone());
    let result = execute_thermal_solver_result_objects(
        &workload,
        &mesh,
        1024,
        ObjectNamespace::ResultValue,
        limits,
    )
    .unwrap();
    let receipt = thermal_receipt_from_prepared_result(&workload, &result).unwrap();
    assert_ne!(receipt.source_workload, prepared.host.workload.identity);
    let receipt_objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::NumericalReceipt,
        receipt.identity().digest(),
        &receipt.canonical_bytes().unwrap(),
        1024,
        ObjectNamespace::ResultValue,
        limits,
    )
    .unwrap();
    let request = |host| AccuracyPreparation {
        solve_host: host,
        source_kind: AccuracySourceKind::Thermal,
        source_numerical_receipt: receipt_objects.clone(),
        source_observations: None,
        source_context_inputs: prepared
            .inputs
            .iter()
            .filter(|input| {
                matches!(
                    input.manifest.kind,
                    AnalysisNumericalObjectKind::ThermalWorkload
                        | AnalysisNumericalObjectKind::Mesh
                )
            })
            .cloned()
            .collect(),
        sequence: SequenceIdentity::from_solver_contract(&workload.model.sequence).unwrap(),
        step: workload.model.sequence.steps[0].clone(),
        program_revision: revision(),
        chunk_bytes: 1024,
        inventory_limits: limits,
    };
    let accuracy = prepare_accuracy(request(prepared.host.clone())).unwrap();
    let source_receipt = runmat_analysis_fea::execution::AccuracySourceReceipt::Thermal(receipt);
    let source_inputs = request(prepared.host.clone()).source_context_inputs;
    let check = |inputs: &[PreparedAnalysisObjects]| {
        validate_accuracy_source_context(
            &accuracy.request,
            &source_receipt,
            None,
            &inputs.iter().collect::<Vec<_>>(),
        )
    };
    assert!(check(&source_inputs).is_ok());
    super::accuracy_source_context_tests::assert_accuracy_plan_identity_bound(
        &accuracy.request,
        &source_receipt,
        None,
        &source_inputs.iter().collect::<Vec<_>>(),
    );
    assert!(check(&[]).is_err());
    assert!(check(&source_inputs[..1]).is_err());
    assert!(check(&[source_inputs[0].clone(), source_inputs[0].clone()]).is_err());
    let other_mesh = task(
        workload.clone(),
        canonical_tetrahedron_solver_mesh(ElementOrder::Tet10),
    );
    let substituted_mesh: Vec<_> = other_mesh
        .inputs
        .iter()
        .filter(|input| input.manifest.kind != AnalysisNumericalObjectKind::ResourcePlan)
        .cloned()
        .collect();
    assert!(check(&substituted_mesh).is_err());
    assert!(prepare_accuracy(request(other_mesh.host)).is_err());
    let mut other_workload = workload.clone();
    other_workload
        .model
        .thermal
        .as_mut()
        .unwrap()
        .reference_temperature = Temperature::new(305.0);
    let substituted = task(other_workload, mesh);
    let substituted_inputs: Vec<_> = substituted
        .inputs
        .iter()
        .filter(|input| input.manifest.kind != AnalysisNumericalObjectKind::ResourcePlan)
        .cloned()
        .collect();
    // Both substituted objects are canonical and freshly hashed, but did not produce this receipt.
    assert!(check(&substituted_inputs).is_err());
    assert!(prepare_accuracy(request(substituted.host)).is_err());
}

#[test]
fn thermal_input_preparation_derives_real_transport_facts() {
    use runmat_analysis_fea::planning::resource::{FeaResourceDemand, FeaResourcePlan};
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let workload = workload(ThermalProcedureControl::BackwardEuler { snapshot_count: 3 });
        let mesh = canonical_tetrahedron_solver_mesh(order);
        for chunk_bytes in [1024, 8192] {
            let policy =
                AnalysisOutputPolicy::new(chunk_bytes, ObjectInventoryLimits::default()).unwrap();
            let prepared = prepare_thermal_inputs(&workload, &mesh, policy).unwrap();
            assert_eq!(prepared.inputs.len(), 2);
            assert_eq!(
                prepared.inputs[0].manifest.kind,
                AnalysisNumericalObjectKind::ThermalWorkload
            );
            assert_eq!(
                ThermalWorkload::from_canonical_bytes(&prepared.inputs[0].payload()).unwrap(),
                workload
            );
            let FeaResourceDemand::ThermalSolve(demand) = &prepared.resource_plan.input.demand
            else {
                panic!("wrong demand")
            };
            assert_eq!(
                demand.inputs,
                prepared_input_resource_facts(&prepared.inputs).unwrap()
            );
            let profile = workload
                .prepare(&mesh)
                .unwrap()
                .transport_profile(chunk_bytes)
                .unwrap();
            assert_eq!(
                demand.result_output,
                projected_transport_output_resource_facts(&profile).unwrap()
            );
            assert!(
                prepared.resource_plan.breakdown.phase_retained_bytes.0
                    >= demand.inputs.transport_closure_bytes.0
            );
            assert_eq!(
                FeaResourcePlan::from_canonical_bytes(
                    &prepared.resource_plan.canonical_bytes().unwrap()
                )
                .unwrap(),
                prepared.resource_plan
            );
            let tiny = AnalysisOutputPolicy::new(
                chunk_bytes,
                ObjectInventoryLimits {
                    max_objects: 1,
                    ..ObjectInventoryLimits::default()
                },
            )
            .unwrap();
            assert!(prepare_thermal_inputs(&workload, &mesh, tiny).is_err());
        }
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
                            target: RegionTarget::boundary(
                                runmat_meshing_core::fixtures::fixture_boundary_selection(&(face)),
                            ),
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
    // The authored ceiling is part of the encoded workload, so the requirement is measured
    // with a non-binding ceiling of the same canonical integer width as the ceilings below.
    workload.model.sequence.steps[0]
        .accuracy
        .resources
        .maximum_resident_bytes = Some(u64::from(u32::MAX));
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let prepared = workload.prepare(&mesh).unwrap();
    let profile = prepared.transport_profile(1024).unwrap();
    let output = projected_transport_output_resource_facts(&profile).unwrap();
    let required = prepared.numerical_reservation_bytes().unwrap().0
        + output.construction_resident_bytes_admission_budget.0;
    assert!(required > u64::from(u16::MAX) && required < u64::from(u32::MAX));
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
        .maximum_resident_bytes = Some(required - 1);
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
        .maximum_resident_bytes = Some(required);
    execute_thermal_solver_result_objects(
        &workload,
        &mesh,
        1024,
        ObjectNamespace::ResultValue,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
}
