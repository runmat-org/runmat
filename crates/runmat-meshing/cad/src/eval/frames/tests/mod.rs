use super::*;
use crate::build_cad_topology;
use crate::eval::{project_to_face, summarize_cad_evaluation, CadEvaluationSource};
use runmat_geometry_core::{CadFaceEvaluationSample, CadFaceEvaluationSampleSource};

mod exact_samples;
mod fixtures;
mod providers;

use fixtures::*;

#[test]
fn builds_planar_face_evaluation_frames() {
    let topology = cube_topology();
    let geometry = geometry_for_topology();
    let cad_topology = build_cad_topology(&geometry, &topology).expect("cad topology");

    let model = build_cad_evaluation_model(&cad_topology, &topology).expect("evaluation model");
    let report = summarize_cad_evaluation(&model, &topology).expect("summary");

    assert_eq!(model.face_frames.len(), topology.faces.len());
    assert_eq!(report.face_frame_count, topology.faces.len());
    assert_eq!(report.source, CadEvaluationSource::PlanarFacetApproximation);
    assert_eq!(report.evaluator_face_count, 0);
    assert_eq!(report.projection_query_count, topology.faces.len() * 3);
    assert_eq!(report.max_projection_error_m, 0.0);
    assert_eq!(report.max_normal_deviation, 0.0);
}

#[test]
fn projects_points_to_face_frame() {
    let topology = cube_topology();
    let geometry = geometry_for_topology();
    let cad_topology = build_cad_topology(&geometry, &topology).expect("cad topology");
    let model = build_cad_evaluation_model(&cad_topology, &topology).expect("evaluation model");

    let projection = project_to_face(&model.face_frames[0], [0.25, 0.25, 0.5]);
    let outside_projection = project_to_face(&model.face_frames[0], [10.0, 10.0, 0.5]);

    assert!(projection.distance_m > 0.0);
    assert!(projection.uv_in_bounds);
    assert!(!outside_projection.uv_in_bounds);
    assert!(dot(projection.unit_normal, model.face_frames[0].unit_normal) > 0.999);
    assert_eq!(
        model.face_frames[0].uv_domain_source.as_deref(),
        Some("source_face_projection")
    );
}

#[test]
fn backend_samples_expose_derivative_and_curvature_estimates() {
    let topology = cube_topology();
    let mut geometry = geometry_with_face_evaluator();
    geometry.source_geometry.cad_evaluators[0].faces[0].evaluation_samples = vec![
        CadFaceEvaluationSample {
            source: CadFaceEvaluationSampleSource::BackendQuery,
            point_m: [0.0, 0.0, 1.0],
            uv: Some([0.0, 0.0]),
            projected_point_m: Some([0.0, 0.0, 1.0]),
            unit_normal: Some([0.0, 0.0, 1.0]),
            projection_error_m: Some(0.0),
        },
        CadFaceEvaluationSample {
            source: CadFaceEvaluationSampleSource::BackendQuery,
            point_m: [1.0, 0.0, 1.0],
            uv: Some([1.0, 0.0]),
            projected_point_m: Some([1.0, 0.0, 1.0]),
            unit_normal: Some([0.0, 0.05, 0.998749217771909]),
            projection_error_m: Some(0.0),
        },
        CadFaceEvaluationSample {
            source: CadFaceEvaluationSampleSource::BackendQuery,
            point_m: [1.0, 1.0, 1.0],
            uv: Some([1.0, 1.0]),
            projected_point_m: Some([1.0, 1.0, 1.0]),
            unit_normal: Some([0.04, 0.0, 0.9991996797437437]),
            projection_error_m: Some(0.0),
        },
        CadFaceEvaluationSample {
            source: CadFaceEvaluationSampleSource::BackendQuery,
            point_m: [0.0, 1.0, 1.0],
            uv: Some([0.0, 1.0]),
            projected_point_m: Some([0.0, 1.0, 1.0]),
            unit_normal: Some([0.0, 0.04, 0.9991996797437437]),
            projection_error_m: Some(0.0),
        },
    ];
    let cad_topology = build_cad_topology(&geometry, &topology).expect("cad topology");

    let model = build_cad_evaluation_model(&cad_topology, &topology).expect("evaluation model");
    let report = summarize_cad_evaluation(&model, &topology).expect("summary");
    let frame = model
        .face_frames
        .iter()
        .find(|frame| frame.evaluator_samples.len() == 3)
        .expect("sample-backed frame");

    assert_eq!(model.report.derivative_query_count, 2);
    assert_eq!(model.report.curvature_query_count, 2);
    assert_eq!(report.derivative_query_count, 2);
    assert_eq!(report.curvature_query_count, 2);
    assert_eq!(report.missing_derivative_query_face_count, 0);
    assert_eq!(report.missing_curvature_query_face_count, 0);
    assert_eq!(frame.u_derivative_m_per_uv, Some([1.0, 0.0, 0.0]));
    assert_eq!(frame.v_derivative_m_per_uv, Some([0.0, 1.0, 0.0]));
    assert!(frame.max_curvature_estimate_1_per_m.unwrap_or(0.0) > 0.0);
    assert_eq!(
        report.max_curvature_estimate_1_per_m,
        model.report.max_curvature_estimate_1_per_m
    );
}

#[test]
fn cad_derivative_estimates_use_projected_backend_points() {
    let topology = cube_topology();
    let mut geometry = geometry_with_face_evaluator();
    geometry.source_geometry.cad_evaluators[0].faces[0].evaluation_samples = vec![
        CadFaceEvaluationSample {
            source: CadFaceEvaluationSampleSource::BackendQuery,
            point_m: [0.0, 0.0, 1.0],
            uv: Some([0.0, 0.0]),
            projected_point_m: Some([0.0, 0.0, 1.0]),
            unit_normal: Some([0.0, 0.0, 1.0]),
            projection_error_m: Some(0.0),
        },
        CadFaceEvaluationSample {
            source: CadFaceEvaluationSampleSource::BackendQuery,
            point_m: [1.0, 0.0, 1.4],
            uv: Some([1.0, 0.0]),
            projected_point_m: Some([1.0, 0.0, 1.0]),
            unit_normal: Some([0.0, 0.0, 1.0]),
            projection_error_m: Some(0.4),
        },
        CadFaceEvaluationSample {
            source: CadFaceEvaluationSampleSource::BackendQuery,
            point_m: [1.0, 1.0, 0.6],
            uv: Some([1.0, 1.0]),
            projected_point_m: Some([1.0, 1.0, 1.0]),
            unit_normal: Some([0.0, 0.0, 1.0]),
            projection_error_m: Some(0.4),
        },
    ];
    let cad_topology = build_cad_topology(&geometry, &topology).expect("cad topology");

    let model = build_cad_evaluation_model(&cad_topology, &topology).expect("evaluation model");
    let frame = model
        .face_frames
        .iter()
        .find(|frame| frame.evaluator_samples.len() == 3)
        .expect("sample-backed frame");

    assert_eq!(frame.u_derivative_m_per_uv, Some([1.0, 0.0, 0.0]));
    assert_eq!(frame.v_derivative_m_per_uv, Some([0.0, 1.0, 0.0]));
}

#[test]
fn cad_curvature_estimates_orient_backend_normals_to_face_frame() {
    let topology = cube_topology();
    let mut geometry = geometry_with_face_evaluator();
    geometry.source_geometry.cad_evaluators[0].faces[0].evaluation_samples = vec![
        CadFaceEvaluationSample {
            source: CadFaceEvaluationSampleSource::BackendQuery,
            point_m: [0.0, 0.0, 1.0],
            uv: Some([0.0, 0.0]),
            projected_point_m: Some([0.0, 0.0, 1.0]),
            unit_normal: Some([0.0, 0.0, 1.0]),
            projection_error_m: Some(0.0),
        },
        CadFaceEvaluationSample {
            source: CadFaceEvaluationSampleSource::BackendQuery,
            point_m: [1.0, 0.0, 1.0],
            uv: Some([1.0, 0.0]),
            projected_point_m: Some([1.0, 0.0, 1.0]),
            unit_normal: Some([0.0, 0.0, -1.0]),
            projection_error_m: Some(0.0),
        },
    ];
    let cad_topology = build_cad_topology(&geometry, &topology).expect("cad topology");

    let model = build_cad_evaluation_model(&cad_topology, &topology).expect("evaluation model");
    let frame = model
        .face_frames
        .iter()
        .find(|frame| frame.evaluator_samples.len() == 2)
        .expect("sample-backed frame");

    assert_eq!(frame.unit_normal, [0.0, 0.0, 1.0]);
    assert_eq!(frame.max_curvature_estimate_1_per_m, Some(0.0));
}
