use runmat_analysis_fea::fixtures::{fixture_model, FixtureId};
use runmat_analysis_fea::ComputeBackend;
use runmat_runtime::analysis::{
    analysis_run_nonlinear_with_options_op, AnalysisNonlinearRunOptions, AnalysisRunPrepContext,
};
use runmat_runtime::operations::OperationContext;

#[test]
fn prep_context_changes_nonlinear_solve_profile_with_bounded_quality() {
    let model = fixture_model(FixtureId::NonlinearAssemblyStress);

    let baseline = analysis_run_nonlinear_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisNonlinearRunOptions::production_recommended(),
        OperationContext::new(Some("trace-prep-solve-base".to_string()), None),
    )
    .expect("baseline nonlinear run should succeed");

    let prep_enhanced = analysis_run_nonlinear_with_options_op(
        &model,
        ComputeBackend::Cpu,
        AnalysisNonlinearRunOptions {
            prep_context: Some(AnalysisRunPrepContext {
                prepared_mesh_count: 48,
                prepared_node_count: 60_000,
                prepared_element_count: 120_000,
                mapped_region_count: 24,
                min_scaled_jacobian: 0.62,
                mean_aspect_ratio: 2.4,
                inverted_element_count: 0,
            }),
            ..AnalysisNonlinearRunOptions::production_recommended()
        },
        OperationContext::new(Some("trace-prep-solve-enhanced".to_string()), None),
    )
    .expect("prep-enhanced nonlinear run should succeed");

    assert!(prep_enhanced.data.publishable);
    assert!(prep_enhanced
        .data
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_PREP_CONTEXT"));

    let base_nonlinear = baseline
        .data
        .nonlinear_results
        .as_ref()
        .expect("baseline nonlinear payload present");
    let prep_nonlinear = prep_enhanced
        .data
        .nonlinear_results
        .as_ref()
        .expect("prep nonlinear payload present");

    let base_max_iters = base_nonlinear
        .iteration_counts
        .iter()
        .copied()
        .max()
        .unwrap_or(0);
    let prep_max_iters = prep_nonlinear
        .iteration_counts
        .iter()
        .copied()
        .max()
        .unwrap_or(0);
    assert!(base_max_iters.abs_diff(prep_max_iters) <= 16);
    assert!(
        base_max_iters != prep_max_iters
            || base_nonlinear.failed_increments != prep_nonlinear.failed_increments
    );
}
