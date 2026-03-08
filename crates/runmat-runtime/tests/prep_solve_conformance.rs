use runmat_analysis_core::AnalysisStepKind;
use runmat_analysis_fea::ComputeBackend;
use runmat_runtime::analysis::{
    analysis_create_model_op, analysis_run_nonlinear_with_options_op,
    AnalysisCreateModelIntentSpec, AnalysisCreateModelProfile, AnalysisNonlinearRunOptions,
};
use runmat_runtime::geometry::{
    geometry_load_op, geometry_prep_for_analysis_op, GeometryPrepForAnalysisSpec,
};
use runmat_runtime::operations::OperationContext;

const TRIANGLE_STL: &str = "solid tri\n  facet normal 0 0 1\n    outer loop\n      vertex 0 0 0\n      vertex 1 0 0\n      vertex 0 1 0\n    endloop\n  endfacet\nendsolid tri\n";

#[test]
fn prep_artifact_reference_changes_nonlinear_solve_profile_with_bounded_quality() {
    let geometry = geometry_load_op(
        "/fixtures/prep_solve_tri.stl",
        TRIANGLE_STL.as_bytes(),
        OperationContext::new(Some("trace-prep-solve-load".to_string()), None),
    )
    .expect("geometry load should succeed");
    let prep = geometry_prep_for_analysis_op(
        &geometry.data,
        GeometryPrepForAnalysisSpec::default(),
        OperationContext::new(Some("trace-prep-solve-prep".to_string()), None),
    )
    .expect("geometry prep should succeed");

    let created = analysis_create_model_op(
        &geometry.data,
        AnalysisCreateModelIntentSpec {
            model_id: "prep_solve_model".to_string(),
            profile: AnalysisCreateModelProfile::NonlinearStructural,
            prep_context: None,
        },
        OperationContext::new(Some("trace-prep-solve-create".to_string()), None),
    )
    .expect("create model should succeed");
    let mut model = created.data;
    model.steps[0].kind = AnalysisStepKind::Nonlinear;

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
            prep_artifact_id: Some(prep.data.prep_artifact_id.clone()),
            ..AnalysisNonlinearRunOptions::production_recommended()
        },
        OperationContext::new(Some("trace-prep-solve-enhanced".to_string()), None),
    )
    .expect("prep-enhanced nonlinear run should succeed");

    assert!(prep_enhanced
        .data
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_PREP_CONTEXT"));
    assert!(prep_enhanced
        .data
        .run
        .diagnostics
        .iter()
        .any(|diag| diag.code == "FEA_PREP_ASSEMBLY"));

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
