use super::*;
use crate::physics::electrical::tests::{mesh, model, prescribe};
use runmat_analysis_core::{ProcedureStepId, RegionTarget, StaticElectricalEquation};
use runmat_meshing_core::ElementOrder;

fn workload(equation: StaticElectricalEquation) -> ElectricalStaticWorkload {
    let mut model = model(equation);
    prescribe(
        &mut model,
        "left".repeat(128),
        RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("slab_left")),
        0.0,
    );
    prescribe(
        &mut model,
        "right".repeat(128),
        RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("slab_right")),
        2.0,
    );
    ElectricalStaticWorkload {
        model,
        step_id: ProcedureStepId("potential".into()),
        relative_residual_target: 1e-11,
        element_temperature_k: None,
        operating_time_s: None,
    }
}

fn facts(
    workload: &ElectricalStaticWorkload,
    mesh: &SolverMeshArtifact,
) -> PreparedElectricalStaticResourceFacts {
    let input =
        (workload.canonical_bytes().unwrap().len() + mesh.canonical_encode().unwrap().len()) as u64;
    let output = electrical_static_result_payload_bytes_upper_bound(workload, mesh).unwrap();
    PreparedElectricalStaticResourceFacts {
        local_execution: LocalExecutionPlan::serial(),
        inputs: PreparedInputResourceFacts {
            encoded_payload_bytes: ByteCount(input),
            transport_closure_bytes: ByteCount(input + 2048),
            decoded_resident_bytes_admission_budget: ByteCount(input * 16),
        },
        result_output: PreparedOutputResourceFacts {
            encoded_payload_bytes_upper_bound: ByteCount(output),
            transport_closure_bytes_upper_bound: ByteCount(output + 2048),
            construction_resident_bytes_admission_budget: ByteCount(output * 16),
        },
        receipt_output: output_facts(crate::execution::ELECTRICAL_RECEIPT_MAX_ENCODED_BYTES),
        state_profiles_output: output_facts(
            workload
                .state_publication_profiles_encoding_profile(Digest::from_bytes(
                    *mesh.canonical_digest.bytes(),
                ))
                .unwrap()
                .encoded_bytes_upper_bound,
        ),
        output_set: PreparedOutputSetResourceFacts {
            encoded_bytes_upper_bound: ByteCount(2048),
        },
        machine_memory_bytes: 16 << 30,
    }
}

fn output_facts(bytes: u64) -> PreparedOutputResourceFacts {
    PreparedOutputResourceFacts {
        encoded_payload_bytes_upper_bound: ByteCount(bytes),
        transport_closure_bytes_upper_bound: ByteCount(bytes + 2048),
        construction_resident_bytes_admission_budget: ByteCount((bytes + 2048) * 16),
    }
}

#[test]
fn charged_results_fit_admission_and_cannot_use_source_free_plans() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        for charge in [-2e-11, 0.0, 2e-11] {
            let mut charged = workload(StaticElectricalEquation::Electrostatics);
            charged.model =
                crate::physics::electrical::tests::charge_density::charged_model(charge, 1000.0);
            let mesh = mesh(order, 2);
            let prepared = facts(&charged, &mesh);
            let plan =
                build_electrical_static_resource_plan(&charged, &mesh, prepared.clone()).unwrap();
            let result = charged.execute_with_resource_plan(&mesh, &plan).unwrap();
            assert!(
                result.canonical_bytes().unwrap().len() as u64
                    <= prepared.result_output.encoded_payload_bytes_upper_bound.0
            );
            assert!(result.response().charge_sources.is_some());
            let free = workload(StaticElectricalEquation::Electrostatics);
            let free_plan =
                build_electrical_static_resource_plan(&free, &mesh, facts(&free, &mesh)).unwrap();
            assert_ne!(plan, free_plan);
            assert!(charged
                .execute_with_resource_plan(&mesh, &free_plan)
                .is_err());
            let mut insufficient = plan.input.clone();
            insufficient.accuracy_ceiling.maximum_resident_bytes =
                Some(plan.request.memory_bytes - 1);
            assert!(matches!(
                build_resource_plan(insufficient),
                Err(ResourcePlanningError::AccuracyCeiling { .. })
            ));
        }
    }
}

#[test]
fn electrical_solve_declares_exact_three_distinct_output_envelopes() {
    use crate::execution::AnalysisNumericalObjectKind;
    for equation in [
        StaticElectricalEquation::Electrostatics,
        StaticElectricalEquation::SteadyConduction,
    ] {
        for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
            let workload = workload(equation);
            let mesh = mesh(order, 1);
            let prepared = facts(&workload, &mesh);
            let plan =
                build_electrical_static_resource_plan(&workload, &mesh, prepared.clone()).unwrap();
            let demand = &plan.input.demand;
            assert!(demand.governs_prepared_outputs());
            for (kind, expected) in [
                (
                    AnalysisNumericalObjectKind::SolverResult,
                    &prepared.result_output,
                ),
                (
                    AnalysisNumericalObjectKind::NumericalReceipt,
                    &prepared.receipt_output,
                ),
                (
                    AnalysisNumericalObjectKind::StatePublicationProfiles,
                    &prepared.state_profiles_output,
                ),
            ] {
                assert_eq!(demand.output_facts(kind), Some(expected));
            }
            for kind in [
                AnalysisNumericalObjectKind::SemanticState,
                AnalysisNumericalObjectKind::Field,
                AnalysisNumericalObjectKind::AccuracyReport,
                AnalysisNumericalObjectKind::PublishedResult,
            ] {
                assert_eq!(demand.output_facts(kind), None);
            }
            let cold =
                FeaResourcePlan::from_canonical_bytes(&plan.canonical_bytes().unwrap()).unwrap();
            assert_eq!(cold, plan);
        }
    }
}

#[test]
fn both_equations_orders_results_fit_source_bound_admission() {
    for equation in [
        StaticElectricalEquation::Electrostatics,
        StaticElectricalEquation::SteadyConduction,
    ] {
        for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
            let workload = workload(equation);
            let mesh = mesh(order, 2);
            let prepared = facts(&workload, &mesh);
            let result = workload.execute(&mesh).unwrap();
            assert!(
                result.canonical_bytes().unwrap().len() as u64
                    <= prepared.result_output.encoded_payload_bytes_upper_bound.0
            );
            let plan =
                build_electrical_static_resource_plan(&workload, &mesh, prepared.clone()).unwrap();
            assert_eq!(
                plan,
                build_electrical_static_resource_plan(&workload, &mesh, prepared).unwrap()
            );
            // This slab has unconstrained interior nodes: exercise the actual admitted
            // matrix/certificate path, not only an all-prescribed transport fixture.
            assert_eq!(
                workload.execute_with_resource_plan(&mesh, &plan).unwrap(),
                result
            );
            let mut altered = plan.input.clone();
            let FeaResourceDemand::ElectricalStatic(demand) = &mut altered.demand else {
                panic!("wrong physics demand")
            };
            demand.solver_admission.factor_nonzero_upper_bound.0 += 1;
            let altered = build_resource_plan(altered).unwrap();
            altered.validate_identity().unwrap();
            assert!(workload
                .execute_with_resource_plan(&mesh, &altered)
                .is_err());
            assert_eq!(
                plan.input.phase,
                FeaResourcePhase::ElectricalStatic(workload.identity().unwrap())
            );
            assert_eq!(plan.input.partitioning(), &PartitioningPolicy::Monolithic);
            let FeaResourceDemand::ElectricalStatic(demand) = &plan.input.demand else {
                panic!("wrong physics demand")
            };
            assert_eq!(
                demand.solver_admission,
                electrical_static_solver_admission(&workload, &mesh).unwrap()
            );
            let (_, envelope) = electrical_static_solver_envelope(&workload, &mesh).unwrap();
            let proof = local_energy_proof_demand(&mesh, envelope).unwrap();
            assert!(
                demand.solver_admission.update_count_upper_bound.0 >= proof.scalar_operations()
            );
            assert!(demand.solver_admission.resident_bytes_upper_bound.0 >= proof.resident_bytes());
            assert!(
                plan.breakdown.workspace_bytes
                    >= demand.solver_admission.resident_bytes_upper_bound
            );
            let mut insufficient = plan.input.clone();
            insufficient.accuracy_ceiling.maximum_resident_bytes =
                Some(plan.request.memory_bytes - 1);
            assert!(matches!(
                build_resource_plan(insufficient),
                Err(ResourcePlanningError::AccuracyCeiling { .. })
            ));
            let mut exact = plan.input.clone();
            exact.accuracy_ceiling.maximum_resident_bytes = Some(plan.request.memory_bytes);
            assert!(build_resource_plan(exact).is_ok());
            let receipt = crate::execution::ElectricalNumericalReceiptArtifact::new(
                &workload, &result, &mesh,
            )
            .unwrap();
            let receipt_output = plan
                .input
                .demand
                .output_facts(crate::execution::AnalysisNumericalObjectKind::NumericalReceipt)
                .unwrap();
            assert!(
                receipt.canonical_bytes().unwrap().len() as u64
                    <= receipt_output.encoded_payload_bytes_upper_bound.0
            );
        }
    }
}

#[test]
fn planning_support_budget_is_checked_before_geometry_or_graph_materialization() {
    let mut workload = workload(StaticElectricalEquation::Electrostatics);
    let mut mesh = mesh(ElementOrder::Tet10, 2);
    let required = support_preparation_bytes(&mesh).unwrap();
    workload.model.sequence.steps[0]
        .accuracy
        .resources
        .maximum_resident_bytes = Some(required - 1);
    // Invalid geometry would fail canonical admission if it were reached first.
    mesh.topology.nodes[0].coordinates_m =
        runmat_types::physical::Quantity::<runmat_types::physical::kind::Position, _>::new(
            mesh.topology.nodes[1].coordinates_m.canonical_value(),
        );
    let rejection = electrical_static_solver_admission(&workload, &mesh);
    assert!(
        matches!(
            rejection,
            Err(ResourcePlanningError::AccuracyCeiling {
                resource: "electrical support preparation bytes",
                ..
            })
        ),
        "unexpected rejection: {rejection:?}"
    );
}

#[test]
fn inconsistent_inputs_outputs_and_overflow_fail_closed() {
    let workload = workload(StaticElectricalEquation::SteadyConduction);
    let mesh = mesh(ElementOrder::Tet4, 1);
    for field in 0..7 {
        let mut prepared = facts(&workload, &mesh);
        match field {
            0 => prepared.inputs.encoded_payload_bytes.0 -= 1,
            1 => prepared.inputs.transport_closure_bytes = ByteCount(0),
            2 => prepared.inputs.decoded_resident_bytes_admission_budget = ByteCount(0),
            3 => prepared.result_output.encoded_payload_bytes_upper_bound.0 -= 1,
            4 => {
                prepared
                    .result_output
                    .construction_resident_bytes_admission_budget = ByteCount(0)
            }
            5 => prepared.result_output.transport_closure_bytes_upper_bound = ByteCount(u64::MAX),
            _ => prepared.output_set.encoded_bytes_upper_bound = ByteCount(u64::MAX),
        }
        assert!(build_electrical_static_resource_plan(&workload, &mesh, prepared).is_err());
    }
    assert!(add(u64::MAX, 1).is_err());
    assert!(mul(u64::MAX, 2).is_err());
}
