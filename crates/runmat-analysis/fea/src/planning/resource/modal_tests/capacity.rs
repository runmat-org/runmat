use super::*;

#[test]
fn modal_capacity_binds_workload_workspace_payload_and_original_ceiling() {
    let (base, mesh, constraints, observations) = modal_case(1, ComputeBackend::Cpu);
    let bounded = StructuralModalWorkload::new(StructuralModalWorkloadInput {
        model: base.model().clone(),
        step_id: base.step_id().clone(),
        backend: base.backend(),
        observation_execution: base.observation_execution(),
        truncation: base.truncation(),
        convergence: base.convergence().with_maximum_subspace_dimension(2),
        residual_vector_directions: Vec::new(),
        prestress_transition: None,
    })
    .unwrap();
    assert_ne!(bounded.identity(), base.identity());
    let plan = build_structural_modal_resource_plan(
        Some(&mesh),
        &constraints,
        &observations,
        &bounded,
        None,
        &[],
        prepared(&mesh, &constraints, &bounded),
    )
    .unwrap();
    let FeaResourceDemand::StructuralModalExtraction(demand) = &plan.input.demand else {
        panic!("modal demand");
    };
    let SolveResourcePolicy::SymmetricSubspaceIteration {
        maximum_subspace_dimension,
        reduced_dimension,
        ..
    } = demand.operator.solve
    else {
        panic!("subspace policy");
    };
    assert_eq!(maximum_subspace_dimension.0, 2);
    assert!(reduced_dimension.0 > 2);
    assert_eq!(demand.retained_mode_count_upper_bound.0, 2);
    assert_eq!(
        demand.result_output.encoded_payload_bytes_upper_bound.0,
        structural_modal_result_payload_bytes_upper_bound(Some(&mesh), &constraints, &bounded)
            .unwrap()
    );
    assert_eq!(
        plan.input.accuracy_ceiling,
        bounded.procedure_step().unwrap().accuracy.resources
    );
    let mut understated = plan.input.clone();
    if let FeaResourceDemand::StructuralModalExtraction(demand) = &mut understated.demand {
        demand.retained_mode_count_upper_bound = super::super::Count(1);
    }
    assert!(super::super::build_resource_plan(understated).is_err());
    let mut ceiling = plan.input.clone();
    ceiling.accuracy_ceiling.maximum_resident_bytes = Some(plan.request.memory_bytes - 1);
    assert!(super::super::build_resource_plan(ceiling).is_err());
}
