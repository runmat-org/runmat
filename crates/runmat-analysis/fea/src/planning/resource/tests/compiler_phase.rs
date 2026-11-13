//! Original construction authority and sequential compiler memory admission.
use super::*;

pub(super) fn assert_carried_request_identity(
    workload: &crate::execution::StructuralLinearWorkload,
    mesh: &SolverMeshArtifact,
    observations: &crate::results::observation::ResolvedObservationPlan,
    plan: &FeaResourcePlan,
) {
    let compiled = CanonicalStructuralConstraintPlan::compile_for_model_with_workspace_limit(
        &workload.model,
        Some(mesh),
        workload
            .procedure_step()
            .unwrap()
            .accuracy
            .resources
            .resident_byte_budget(),
    )
    .unwrap();
    let carried = build_structural_resource_plan_from_compilation(
        Some(mesh),
        &compiled,
        observations,
        workload,
        &[],
        LocalExecutionPlan::serial(),
        None,
        crate::planning::resource::StaticSolveSelection::WithinCeiling {
            machine_memory_bytes: None,
        },
    )
    .unwrap();
    assert_eq!(
        &carried, plan,
        "fresh and carried construction produce identical requests"
    );
    let FeaResourceDemand::StructuralLinearPipeline(demand) = &carried.input.demand else {
        unreachable!()
    };
    assert_eq!(
        demand.constraint_compilation_peak_bytes.0,
        compiled.workspace().peak_requested_bytes()
    );
    let equivalent = workload.clone();
    assert!(
        build_structural_resource_plan_from_compilation(
            Some(mesh),
            &compiled,
            observations,
            &equivalent,
            &[],
            LocalExecutionPlan::serial(),
            None,
            crate::planning::resource::StaticSolveSelection::WithinCeiling {
                machine_memory_bytes: None
            },
        )
        .is_err(),
        "foreign borrowed source cannot reuse construction evidence"
    );
    let equivalent_mesh = mesh.clone();
    assert!(build_structural_resource_plan_from_compilation(
        Some(&equivalent_mesh),
        &compiled,
        observations,
        workload,
        &[],
        LocalExecutionPlan::serial(),
        None,
        crate::planning::resource::StaticSolveSelection::WithinCeiling {
            machine_memory_bytes: None
        },
    )
    .is_err());
    let fresh = CanonicalStructuralConstraintPlan::compile_for_model_with_workspace_limit(
        &equivalent.model,
        Some(&equivalent_mesh),
        equivalent
            .procedure_step()
            .unwrap()
            .accuracy
            .resources
            .resident_byte_budget(),
    )
    .unwrap();
    assert_eq!(
        build_structural_resource_plan_from_compilation(
            Some(&equivalent_mesh),
            &fresh,
            observations,
            &equivalent,
            &[],
            LocalExecutionPlan::serial(),
            None,
            crate::planning::resource::StaticSolveSelection::WithinCeiling {
                machine_memory_bytes: None
            },
        )
        .unwrap(),
        *plan,
        "construction addresses never enter canonical identities"
    );
}

#[test]
fn original_compiler_phase_is_mandatory_and_governs_sequential_peak() {
    let mut planned = input(PartitioningPolicy::Monolithic);
    let baseline = build_resource_plan(planned.clone()).unwrap();
    let compiler = baseline.request.memory_bytes * 2;
    let FeaResourceDemand::StructuralLinearPipeline(demand) = &mut planned.demand else {
        unreachable!()
    };
    demand.constraint_compilation_peak_bytes = ByteCount(compiler);
    let plan = build_resource_plan(planned.clone()).unwrap();
    assert_eq!(
        plan.request.memory_bytes,
        planned.worker.fixed_memory_bytes.0
            + compiler
            + plan.breakdown.resource_plan_decoded_resident_bytes.0
            + plan.breakdown.resource_plan_transport_closure_bytes.0
    );
    planned.accuracy_ceiling.maximum_resident_bytes = Some(plan.request.memory_bytes);
    assert_eq!(
        build_resource_plan(planned.clone()).unwrap().request,
        plan.request
    );
    planned.accuracy_ceiling.maximum_resident_bytes =
        Some(planned.accuracy_ceiling.resident_byte_budget() - 1);
    assert!(matches!(
        build_resource_plan(planned),
        Err(ResourcePlanningError::AccuracyCeiling { .. })
    ));
    let mut invalid = plan.input.clone();
    let FeaResourceDemand::StructuralLinearPipeline(demand) = &mut invalid.demand else {
        unreachable!()
    };
    demand.constraint_compilation_peak_bytes = ByteCount(0);
    assert!(build_resource_plan(invalid).is_err());
    let mut cold = serde_json::to_value(&plan).unwrap();
    // Locate the required field independently of serde's enum representation.
    fn remove(value: &mut serde_json::Value) -> bool {
        match value {
            serde_json::Value::Object(fields) => {
                fields.remove("constraint_compilation_peak_bytes").is_some()
                    || fields.values_mut().any(remove)
            }
            serde_json::Value::Array(items) => items.iter_mut().any(remove),
            _ => false,
        }
    }
    assert!(remove(&mut cold));
    assert!(serde_json::from_value::<FeaResourcePlan>(cold).is_err());
}

#[test]
fn original_compiler_reservation_is_not_divided_by_operator_partitions() {
    let policy = PartitioningPolicy::OperatorDomain {
        maximum_partitions: 4,
        replicated_memory_bytes_per_partition: ByteCount(20),
        halo_memory_bytes_per_partition: ByteCount(10),
        replicated_scratch_bytes_per_partition: ByteCount(0),
        halo_scratch_bytes_per_partition: ByteCount(0),
    };
    let mut planned = input(policy);
    let baseline = build_resource_plan(planned.clone()).unwrap();
    let FeaResourceDemand::StructuralLinearPipeline(demand) = &mut planned.demand else {
        unreachable!()
    };
    demand.constraint_compilation_peak_bytes.0 += 1000;
    let changed = build_resource_plan(planned).unwrap();
    assert_eq!(
        changed.breakdown.partitionable_memory_bytes,
        baseline.breakdown.partitionable_memory_bytes
    );
    assert_eq!(
        changed.breakdown.fixed_memory_bytes.0,
        baseline.breakdown.fixed_memory_bytes.0 + 1000
    );
}
