//! Cold completed-chain substitution checks; all inputs are retained, never re-solved.
use runmat_analysis_execution::{
    import_maxwell_study_completion, CompletedMaxwellStudy, MaxwellStudyResultPhase,
};
use runmat_analysis_fea::{properties::compile_property_map, study::plan_maxwell_study};
use runmat_execution_artifact::cache::CacheImport;

pub(super) fn check(completed: &CompletedMaxwellStudy, source: &impl CacheImport) {
    let closures = completed.result_closures();
    let later = closures
        .iter()
        .position(|closure| closure.phase == MaxwellStudyResultPhase::Accuracy)
        .expect("completed authored witness includes accuracy after recovery");
    assert!(later > 1);

    let mut omitted = closures.to_vec();
    omitted.remove(later);
    assert!(
        import_maxwell_study_completion(source, completed.preparation().unwrap(), omitted).is_err()
    );

    let mut duplicated = closures.to_vec();
    duplicated.insert(later + 1, closures[later].clone());
    assert!(
        import_maxwell_study_completion(source, completed.preparation().unwrap(), duplicated)
            .is_err()
    );

    // Even internally valid scalar accounting cannot replace the original
    // immutable scheduler association through construction or cold decoding.
    crate::phase_execution_support::assert_foreign_allocation_cannot_be_sealed(
        &closures[later].execution,
    );

    // A coherent new compiled constitutive intent does not own the old task/result chain.
    let mut changed = completed.preparation().unwrap();
    changed.product.compiled.model.materials[0]
        .electrical
        .as_mut()
        .unwrap()
        .conductivity_s_per_m = runmat_types::physical::Quantity::new(
        changed.product.compiled.model.materials[0]
            .electrical
            .as_mut()
            .unwrap()
            .conductivity_s_per_m
            .canonical_value()
            * (1.125),
    );
    changed.product.compiled.properties =
        compile_property_map(&changed.product.compiled.model, Some(&changed.mesh)).unwrap();
    changed.product.workload =
        plan_maxwell_study(&changed.product.compiled, changed.policy).unwrap();
    changed
        .product
        .compiled
        .validate_against(&changed.mesh)
        .unwrap();
    assert!(import_maxwell_study_completion(source, changed, closures.to_vec()).is_err());
}
