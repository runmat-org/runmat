use super::*;
use crate::constraints::CanonicalStructuralConstraintPlan;

// witness: FEA-SHARED-CONSTRAINTS-001/canonical_plan_and_affine_reduction @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Prepared structural constraint fixtures compare original canonical plans and bytes at
//   the exact requested workspace boundary and reject one byte below; no cross-family enforcement
//   or whole-task memory proof.
#[test]
fn exact_caller_boundary_preserves_original_plan_across_all_compiler_routes() {
    for (name, model, mesh) in crate::constraints::compilation_tests::cases() {
        let original =
            CanonicalStructuralConstraintPlan::compile_for_model(&model, mesh.as_ref()).unwrap();
        let first = CanonicalStructuralConstraintPlan::compile_for_model_with_workspace_limit(
            &model,
            mesh.as_ref(),
            u64::MAX,
        )
        .unwrap();
        let required = first.workspace().peak_requested_bytes();
        assert!(required > 0);
        assert_eq!(*first.plan(), original);
        let exact = CanonicalStructuralConstraintPlan::compile_for_model_with_workspace_limit(
            &model,
            mesh.as_ref(),
            required,
        )
        .unwrap();
        assert_eq!(exact.workspace(), first.workspace(), "{name}");
        assert_eq!(
            exact.plan().canonical_bytes().unwrap(),
            original.canonical_bytes().unwrap()
        );
        let error = CanonicalStructuralConstraintPlan::compile_for_model_with_workspace_limit(
            &model,
            mesh.as_ref(),
            required - 1,
        )
        .unwrap_err();
        assert!(
            matches!(error, ConstraintPlanError::WorkspaceLimit { required: actual, maximum, .. } if actual == required && maximum == required - 1),
            "{name}: {error}"
        );
    }
}

#[test]
fn compiler_requested_peak_fits_independent_execution_allocations() {
    for (name, model, mesh) in crate::constraints::compilation_tests::cases() {
        let (compiled, live, peak) = crate::topology::measure_test_allocations(|| {
            CanonicalStructuralConstraintPlan::compile_for_model_with_workspace_limit(
                &model,
                mesh.as_ref(),
                u64::MAX,
            )
        });
        let compiled = compiled.unwrap();
        let inventory = compiled.workspace().peak_requested_bytes();
        assert!(
            peak as u64 <= inventory,
            "{name}: actual {peak}, inventory {inventory}"
        );
        assert!(live <= peak);
        println!("compiler {name}: requested_peak={peak} retained={live} inventory={inventory}");
    }
}

#[test]
fn zero_limit_precedes_duplicate_source_validation_and_cancellation_stays_typed() {
    let (_, mut model, mesh) = crate::constraints::compilation_tests::cases().remove(0);
    model.constraints.push(model.constraints[0].clone());
    assert!(matches!(
        CanonicalStructuralConstraintPlan::compile_for_model_with_workspace_limit(
            &model,
            mesh.as_ref(),
            0
        ),
        Err(ConstraintPlanError::WorkspaceLimit {
            phase: ConstraintCompilationPhase::Preparation,
            maximum: 0,
            ..
        })
    ));
    let token = crate::progress::FeaCancellationToken::new();
    token.cancel();
    let _guard =
        crate::progress::replace_fea_progress_context(crate::progress::FeaProgressContext {
            cancellation: token,
            ..Default::default()
        });
    assert!(matches!(
        CanonicalStructuralConstraintPlan::compile_for_model_with_workspace_limit(
            &model,
            mesh.as_ref(),
            0
        ),
        Err(ConstraintPlanError::Cancelled)
    ));
}
