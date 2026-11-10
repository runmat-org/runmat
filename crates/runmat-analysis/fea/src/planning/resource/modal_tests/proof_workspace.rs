use super::*;

#[test]
fn modal_preassembly_reserves_subgraph_proof_envelope_and_enforces_original_ceiling() {
    let (workload, mesh, constraints, observations) = modal_case(1, ComputeBackend::Cpu);
    let facts = super::super::structural_facts::derive_structural_resource_facts(
        workload.model(),
        Some(&mesh),
        &constraints,
    )
    .unwrap();
    let reduction = facts.constraint_matrix.affine_reduction().unwrap();
    let support = reduction
        .reduce_sparse_pattern(facts.symbolic.stiffness_upper_bound())
        .unwrap();
    let envelope = crate::algebra::CoercivityWorkEnvelope::derive(&support).unwrap();
    let plan = build_structural_modal_resource_plan(
        Some(&mesh),
        &constraints,
        &observations,
        &workload,
        None,
        &[],
        prepared(&mesh, &constraints, &workload),
    )
    .unwrap();
    let SolveResourcePolicy::SymmetricSubspaceIteration {
        workspace_bytes_upper_bound,
        ..
    } = plan.input.structural_operator().unwrap().solve
    else {
        panic!("modal workspace must retain typed subspace demand");
    };
    // The complete numerical workspace must contain the peak of every proof that can run on
    // the reduced support: the proving factorization always, and the anchored local-energy
    // proof only when the original energy source is captured (it reads that source).
    // An upper graph's own successful factorization cannot reserve this subgraph inventory.
    let FeaResourceDemand::StructuralModalExtraction(demand) = &plan.input.demand else {
        panic!("modal demand");
    };
    assert!(workspace_bytes_upper_bound.0 >= envelope.factor_resident_limit());
    if demand.original_energy_source_bytes.0 > 0 {
        assert!(workspace_bytes_upper_bound.0 >= envelope.resident_bytes_limit());
    }
    let mut too_small = plan.input.clone();
    too_small.accuracy_ceiling.maximum_resident_bytes = Some(plan.request.memory_bytes - 1);
    assert!(super::super::build_resource_plan(too_small).is_err());
    assert_eq!(
        plan.input.accuracy_ceiling.resident_byte_budget(),
        workload
            .procedure_step()
            .unwrap()
            .accuracy
            .resources
            .resident_byte_budget()
    );
}
