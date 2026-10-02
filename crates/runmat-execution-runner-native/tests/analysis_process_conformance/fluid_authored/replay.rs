//! Original completed source and governed accounting cannot be replaced in cold replay.
use runmat_analysis_execution::{import_fluid_study_completion, CompletedFluidStudy};
use runmat_execution::resource::ResourceAccounting;
use runmat_execution_artifact::cache::CacheImport;
pub(super) fn check(completed: &CompletedFluidStudy, source: &impl CacheImport) {
    let mut closures = completed.result_closures().to_vec();
    let accounting = &mut closures[1].accounting;
    let mut allocation = accounting.resources.allocation.clone();
    allocation.memory_bytes += 1;
    accounting.resources = ResourceAccounting::for_reserved_interval(
        allocation,
        accounting.resources.usage.wall_millis,
        accounting.resources.artifact_bytes,
        accounting.resources.usage.egress_bytes,
        accounting.resources.usage.relay_bytes,
    )
    .unwrap();
    accounting.validate().unwrap();
    assert!(
        import_fluid_study_completion(source, completed.preparation().unwrap(), closures).is_err()
    );
    let mut changed = completed.preparation().unwrap();
    changed.policy.numerical.momentum_relaxation = 0.75;
    changed.product = runmat_analysis_fea::study::plan_fluid_study_product(
        changed.product.compiled.clone(),
        &changed.mesh,
        changed.policy,
    )
    .unwrap();
    assert!(
        import_fluid_study_completion(source, changed, completed.result_closures().to_vec())
            .is_err()
    );
}
