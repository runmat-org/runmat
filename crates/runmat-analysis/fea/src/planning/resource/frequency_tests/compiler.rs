use super::*;
use crate::planning::resource::{
    compiler_test_support, FeaResourcePlan, StructuralFrequencyEntryResourceProjection,
};
#[test]
fn original_frequency_projection_reuses_compilation_without_source_scope_substitution() {
    let (workload, mesh, constraints, request) = direct_case(8);
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
    let states = StructuralFrequencyStateInputs {
        prestress: None,
        modal_basis: None,
    };
    let fresh = StructuralFrequencyEntryResourceProjection::new(
        &mesh,
        &constraints,
        &workload,
        states,
        &[],
    )
    .unwrap();
    let carried = StructuralFrequencyEntryResourceProjection::new_from_compilation(
        &mesh,
        &compiled,
        &workload,
        states,
        &[],
    )
    .unwrap();
    let plan = |projection: &StructuralFrequencyEntryResourceProjection| {
        projection
            .plan_for(
                &request,
                input_facts(1024),
                entry_output(&request),
                PreparedOutputSetResourceFacts {
                    encoded_bytes_upper_bound: ByteCount(64 * 1024),
                },
                crate::planning::resource::LocalExecutionPlan::serial(),
            )
            .unwrap()
    };
    assert_eq!(plan(&fresh), plan(&carried));
    compiler_test_support::assert_compiler_contract(
        &plan(&carried),
        compiled.workspace().peak_requested_bytes(),
    );
    let equivalent = workload.clone();
    assert!(
        StructuralFrequencyEntryResourceProjection::new_from_compilation(
            &mesh,
            &compiled,
            &equivalent,
            states,
            &[]
        )
        .is_err()
    );
    let independent = CanonicalStructuralConstraintPlan::compile_for_model_with_workspace_limit(
        equivalent.model(),
        Some(&mesh),
        equivalent
            .procedure_step()
            .unwrap()
            .accuracy
            .resources
            .resident_byte_budget(),
    )
    .unwrap();
    let repeated = StructuralFrequencyEntryResourceProjection::new_from_compilation(
        &mesh,
        &independent,
        &equivalent,
        states,
        &[],
    )
    .unwrap();
    assert_eq!(plan(&repeated), plan(&fresh));
}

pub(super) fn assert_recovery_compilation(
    mesh: &runmat_meshing_core::SolverMeshArtifact,
    workload: &StructuralFrequencyWorkload,
    request: &FieldRecoveryRequest,
    entry: &crate::execution::StructuralFrequencyEntryResult,
    original: &FeaResourcePlan,
) {
    let compiled = CanonicalStructuralConstraintPlan::compile_for_model_with_workspace_limit(
        workload.model(),
        Some(mesh),
        workload
            .procedure_step()
            .unwrap()
            .accuracy
            .resources
            .resident_byte_budget(),
    )
    .unwrap();
    let carried = super::super::build_structural_frequency_recovery_resource_plan_from_compilation(
        mesh,
        &compiled,
        workload,
        StructuralFrequencyRecoveryResourceInput {
            request,
            result: entry,
        },
        StructuralFrequencyStateInputs::default(),
        &[],
        prepared(
            input_facts(1024),
            recovery_output(mesh, compiled.plan(), workload),
        ),
        crate::planning::resource::LocalExecutionPlan::serial(),
    )
    .unwrap();
    assert_eq!(&carried, original);
    compiler_test_support::assert_compiler_contract(
        &carried,
        compiled.workspace().peak_requested_bytes(),
    );
    let foreign = workload.clone();
    assert!(
        super::super::StructuralFrequencyEntryResourceProjection::for_recovery_from_compilation(
            mesh,
            &compiled,
            &foreign,
            StructuralFrequencyStateInputs::default(),
            &[],
        )
        .is_err()
    );
}
