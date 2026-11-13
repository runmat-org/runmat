//! Original completed source and governed accounting cannot be replaced in cold replay.
use runmat_analysis_execution::{import_fluid_study_completion, CompletedFluidStudy};
use runmat_execution_artifact::cache::CacheImport;
pub(super) fn check(completed: &CompletedFluidStudy, source: &impl CacheImport) {
    let closures = completed.result_closures();
    crate::phase_execution_support::assert_foreign_allocation_cannot_be_sealed(
        &closures[1].execution,
    );
    let mut changed = completed.preparation().unwrap();
    changed.policy.numerical.momentum_relaxation = 0.75;
    changed.product = runmat_analysis_fea::study::plan_fluid_study_product(
        changed.product.compiled().clone(),
        &changed.mesh,
        changed.policy,
    )
    .unwrap();
    assert!(
        import_fluid_study_completion(source, changed, completed.result_closures().to_vec())
            .is_err()
    );
}
