use crate::{
    build_cad_topology,
    eval::{
        build_cad_evaluation_model, build_cad_evaluation_model_with_provider,
        summarize_cad_evaluation, CadEvaluationSource, CadFaceEvaluationRequest,
        CadFaceEvaluatorProvider, NoopCadFaceEvaluatorProvider,
    },
};
use runmat_geometry_core::{CadFaceEvaluationSample, CadFaceEvaluationSampleSource};

use super::fixtures::*;

#[test]
fn uses_imported_evaluator_face_samples_when_available() {
    let topology = cube_topology();
    let geometry = geometry_with_face_evaluator();
    let cad_topology = build_cad_topology(&geometry, &topology).expect("cad topology");

    let model = build_cad_evaluation_model(&cad_topology, &topology).expect("evaluation model");

    assert_eq!(model.source, CadEvaluationSource::ImportedEvaluatorSamples);
    assert_eq!(model.report.evaluator_face_count, 1);
    assert_eq!(model.report.point_evaluation_supported_face_count, 2);
    assert_eq!(model.report.projection_supported_face_count, 2);
    assert_eq!(model.report.normal_supported_face_count, 2);
    assert_eq!(model.report.derivative_supported_face_count, 2);
    assert_eq!(model.report.curvature_supported_face_count, 2);
    assert_eq!(model.report.exact_query_face_count, 0);
    assert_eq!(model.report.missing_exact_query_face_count, 2);
    assert_eq!(model.report.missing_derivative_query_face_count, 2);
    assert_eq!(model.report.missing_curvature_query_face_count, 2);
    assert_eq!(model.report.evaluator_sample_count, 0);
    assert!(model.face_frames.iter().any(|frame| frame.evaluator_backed
        && frame.origin_m == [0.25, 0.25, 0.75]
        && frame.unit_normal == [0.0, 0.0, 1.0]));
}

#[test]
fn no_op_provider_keeps_imported_evaluator_metadata_sample_based() {
    let topology = cube_topology();
    let geometry = geometry_with_face_evaluator();
    let cad_topology = build_cad_topology(&geometry, &topology).expect("cad topology");

    let model = build_cad_evaluation_model_with_provider(
        &cad_topology,
        &topology,
        &NoopCadFaceEvaluatorProvider,
    )
    .expect("evaluation model");

    assert_eq!(model.source, CadEvaluationSource::ImportedEvaluatorSamples);
    assert_eq!(model.report.live_query_face_count, 0);
    assert_eq!(model.report.exact_query_face_count, 0);
    assert_eq!(model.report.missing_exact_query_face_count, 2);
    assert_eq!(model.report.missing_derivative_query_face_count, 2);
    assert_eq!(model.report.missing_curvature_query_face_count, 2);
    assert!(model
        .face_frames
        .iter()
        .all(|frame| !frame.live_query_backed));
}

#[test]
fn live_evaluator_provider_samples_drive_parametric_cad_frames() {
    #[derive(Debug)]
    struct LiveProvider;

    impl CadFaceEvaluatorProvider for LiveProvider {
        fn evaluate_face(
            &self,
            request: &CadFaceEvaluationRequest<'_>,
        ) -> Vec<CadFaceEvaluationSample> {
            assert_eq!(request.imported_face_id, Some(1));
            assert_eq!(request.evaluator_id, Some("cad_face_1"));
            assert!(request.supports_projection);
            assert!(request.supports_normal);
            assert_eq!(request.reference_point_m, [0.25, 0.25, 0.75]);
            assert_eq!(request.reference_unit_normal, [0.0, 0.0, 1.0]);
            vec![CadFaceEvaluationSample {
                source: CadFaceEvaluationSampleSource::BackendQuery,
                point_m: [0.5, 0.5, 1.01],
                uv: Some([0.5, 0.5]),
                projected_point_m: Some([0.5, 0.5, 1.0]),
                unit_normal: Some([0.0, 0.0, 1.0]),
                projection_error_m: Some(0.01),
            }]
        }
    }

    let topology = cube_topology();
    let geometry = geometry_with_face_evaluator();
    let cad_topology = build_cad_topology(&geometry, &topology).expect("cad topology");

    let model = build_cad_evaluation_model_with_provider(&cad_topology, &topology, &LiveProvider)
        .expect("evaluation model");
    let report = summarize_cad_evaluation(&model, &topology).expect("summary");
    let frame = model
        .face_frames
        .iter()
        .find(|frame| frame.live_query_backed)
        .expect("live-query frame");

    assert_eq!(model.source, CadEvaluationSource::ParametricCad);
    assert_eq!(model.report.live_query_face_count, 2);
    assert_eq!(model.report.exact_query_face_count, 2);
    assert_eq!(model.report.projection_supported_face_count, 2);
    assert_eq!(model.report.normal_supported_face_count, 2);
    assert_eq!(report.projection_supported_face_count, 2);
    assert_eq!(report.normal_supported_face_count, 2);
    assert_eq!(model.report.missing_exact_query_face_count, 0);
    assert_eq!(model.report.missing_derivative_query_face_count, 2);
    assert_eq!(model.report.missing_curvature_query_face_count, 2);
    assert_eq!(model.report.evaluator_sample_count, 2);
    assert_eq!(report.live_query_face_count, 2);
    assert_eq!(report.missing_exact_query_face_count, 0);
    assert_eq!(report.source, CadEvaluationSource::ParametricCad);
    assert_eq!(frame.origin_m, [0.5, 0.5, 1.0]);
    assert_eq!(frame.evaluator_samples.len(), 1);
    assert_eq!(frame.evaluator_max_projection_error_m, 0.01);
}
