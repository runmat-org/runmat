use super::*;
use crate::planning::resource::{
    build_structural_modal_resource_plan_from_compilation, compiler_test_support,
};
#[test]
fn original_modal_compiler_inventory_and_source_scope_govern_canonical_request() {
    let (workload, mesh, constraints, observations) = modal_case(1, ComputeBackend::Cpu);
    let compiled = CanonicalStructuralConstraintPlan::compile_for_model_with_workspace_limit(
        workload.model(),
        Some(&mesh),
        workload
            .procedure_step()
            .unwrap()
            .accuracy
            .resources
            .resident_byte_budget(),
    )
    .unwrap();
    let fresh = build_structural_modal_resource_plan(
        Some(&mesh),
        &constraints,
        &observations,
        &workload,
        None,
        &[],
        prepared(&mesh, &constraints, &workload),
    )
    .unwrap();
    let carried = build_structural_modal_resource_plan_from_compilation(
        Some(&mesh),
        &compiled,
        &observations,
        &workload,
        None,
        &[],
        prepared(&mesh, &constraints, &workload),
    )
    .unwrap();
    assert_eq!(fresh, carried);
    compiler_test_support::assert_compiler_contract(
        &carried,
        compiled.workspace().peak_requested_bytes(),
    );
    let equivalent = workload.clone();
    assert!(build_structural_modal_resource_plan_from_compilation(
        Some(&mesh),
        &compiled,
        &observations,
        &equivalent,
        None,
        &[],
        prepared(&mesh, &constraints, &equivalent)
    )
    .is_err());
    let equivalent_mesh = mesh.clone();
    let independent = CanonicalStructuralConstraintPlan::compile_for_model_with_workspace_limit(
        equivalent.model(),
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
        build_structural_modal_resource_plan_from_compilation(
            Some(&equivalent_mesh),
            &independent,
            &observations,
            &equivalent,
            None,
            &[],
            prepared(&equivalent_mesh, &constraints, &equivalent)
        )
        .unwrap(),
        fresh
    );
}

/// A larger later numerical phase need not increase a compiler-dominated task peak.
pub(super) fn assert_residual_work_storage(
    base: &super::super::FeaResourcePlan,
    enriched: &super::super::FeaResourcePlan,
) {
    assert!(enriched.breakdown.workspace_bytes > base.breakdown.workspace_bytes);
    assert!(enriched.breakdown.phase_resident_bytes > base.breakdown.phase_resident_bytes);
    assert!(enriched.request.memory_bytes >= base.request.memory_bytes);
    if enriched.request.memory_bytes == base.request.memory_bytes {
        for plan in [base, enriched] {
            let super::super::FeaResourceDemand::StructuralModalExtraction(demand) =
                &plan.input.demand
            else {
                panic!("original modal demand required")
            };
            let original_peak = demand.constraint_compilation_peak_bytes.0
                + plan.input.worker.fixed_memory_bytes.0
                + plan.breakdown.input_transport_closure_bytes.0
                + plan.breakdown.input_decoded_resident_bytes.0
                + plan.breakdown.resource_plan_transport_closure_bytes.0
                + plan.breakdown.resource_plan_decoded_resident_bytes.0;
            assert_eq!(plan.request.memory_bytes, original_peak);
        }
    }
}
