use super::*;
use crate::build_cad_topology;
use crate::eval::{
    face_uv_contains, project_to_face, summarize_cad_evaluation, CadEvaluationSource,
};
use runmat_geometry_core::{CadFaceEvaluationSample, CadFaceEvaluationSampleSource};

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
fn exact_backend_query_samples_drive_parametric_cad_frames() {
    let topology = cube_topology();
    let mut geometry = geometry_with_face_evaluator();
    geometry.source_geometry.cad_evaluators[0].faces[0].evaluation_samples =
        vec![CadFaceEvaluationSample {
            source: CadFaceEvaluationSampleSource::BackendQuery,
            point_m: [0.5, 0.5, 1.0],
            uv: Some([0.5, 0.5]),
            projected_point_m: Some([0.5, 0.5, 1.0]),
            unit_normal: Some([0.0, 0.0, 1.0]),
            projection_error_m: Some(2.0e-6),
        }];
    let cad_topology = build_cad_topology(&geometry, &topology).expect("cad topology");

    let model = build_cad_evaluation_model(&cad_topology, &topology).expect("evaluation model");
    let report = summarize_cad_evaluation(&model, &topology).expect("summary");

    assert_eq!(model.source, CadEvaluationSource::ImportedEvaluatorSamples);
    assert_eq!(model.report.evaluator_sample_count, 2);
    assert_eq!(model.report.exact_query_face_count, 2);
    assert_eq!(model.report.max_projection_error_m, 2.0e-6);
    assert_eq!(report.max_projection_error_m, 2.0e-6);
    let frame = model
        .face_frames
        .iter()
        .find(|frame| frame.exact_query_backed)
        .expect("one frame should be exact-query backed");
    assert_eq!(frame.origin_m, [0.5, 0.5, 1.0]);
    assert_eq!(frame.unit_normal, [0.0, 0.0, 1.0]);
    assert_eq!(frame.evaluator_max_projection_error_m, 2.0e-6);
    assert_eq!(frame.evaluator_samples.len(), 1);
    assert_eq!(frame.evaluator_samples[0].uv, Some([0.5, 0.5]));
    assert!(frame.uv_bounds.is_some());
    assert!(face_uv_contains(frame, [0.5, 0.5]));
}

#[test]
fn exact_backend_samples_define_uv_domain_when_sufficient() {
    let topology = cube_topology();
    let mut geometry = geometry_with_face_evaluator();
    geometry.source_geometry.cad_evaluators[0].faces[0].evaluation_samples = vec![
        CadFaceEvaluationSample {
            source: CadFaceEvaluationSampleSource::BackendQuery,
            point_m: [0.0, 0.0, 1.0],
            uv: Some([2.0, 4.0]),
            projected_point_m: Some([0.0, 0.0, 1.0]),
            unit_normal: Some([0.0, 0.0, 1.0]),
            projection_error_m: Some(0.0),
        },
        CadFaceEvaluationSample {
            source: CadFaceEvaluationSampleSource::BackendQuery,
            point_m: [1.0, 0.0, 1.0],
            uv: Some([5.0, 4.0]),
            projected_point_m: Some([1.0, 0.0, 1.0]),
            unit_normal: Some([0.0, 0.0, 1.0]),
            projection_error_m: Some(0.0),
        },
        CadFaceEvaluationSample {
            source: CadFaceEvaluationSampleSource::BackendQuery,
            point_m: [1.0, 1.0, 1.0],
            uv: Some([5.0, 7.0]),
            projected_point_m: Some([1.0, 1.0, 1.0]),
            unit_normal: Some([0.0, 0.0, 1.0]),
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

    assert_eq!(frame.uv_bounds, Some([[2.0, 4.0], [5.0, 7.0]]));
    assert_eq!(frame.uv_bounds_sample_count, 3);
    assert_eq!(frame.uv_domain_source.as_deref(), Some("exact_samples"));
    assert!(face_uv_contains(frame, [3.0, 6.0]));
    assert!(!face_uv_contains(frame, [6.0, 6.0]));
    assert!(model.report.uv_domain_face_count > 0);
    assert!(report.uv_domain_face_count > 0);
    assert!(report.uv_projection_out_of_bounds_count > 0);
}

#[test]
fn cad_face_frames_normalize_backend_normals() {
    let topology = cube_topology();
    let mut geometry = geometry_with_face_evaluator();
    geometry.source_geometry.cad_evaluators[0].faces[0].evaluation_samples =
        vec![CadFaceEvaluationSample {
            source: CadFaceEvaluationSampleSource::BackendQuery,
            point_m: [0.5, 0.5, 1.0],
            uv: Some([0.5, 0.5]),
            projected_point_m: Some([0.5, 0.5, 1.0]),
            unit_normal: Some([0.0, 0.0, 2.0]),
            projection_error_m: Some(0.0),
        }];
    let cad_topology = build_cad_topology(&geometry, &topology).expect("cad topology");

    let model = build_cad_evaluation_model(&cad_topology, &topology).expect("evaluation model");
    let frame = model
        .face_frames
        .iter()
        .find(|frame| frame.exact_query_backed)
        .expect("one frame should be exact-query backed");
    let projection = project_to_face(frame, [0.5, 0.5, 1.25]);

    assert_eq!(frame.unit_normal, [0.0, 0.0, 1.0]);
    assert!((norm(frame.v_axis) - 1.0).abs() <= 1.0e-12);
    assert!((projection.distance_m - 0.25).abs() <= 1.0e-12);
}

#[test]
fn cad_face_frames_orient_backend_normals_to_source_face() {
    let topology = cube_topology();
    let mut geometry = geometry_with_face_evaluator();
    geometry.source_geometry.cad_evaluators[0].faces[0].evaluation_samples =
        vec![CadFaceEvaluationSample {
            source: CadFaceEvaluationSampleSource::BackendQuery,
            point_m: [0.5, 0.5, 1.0],
            uv: Some([0.5, 0.5]),
            projected_point_m: Some([0.5, 0.5, 1.0]),
            unit_normal: Some([0.0, 0.0, -1.0]),
            projection_error_m: Some(0.0),
        }];
    let cad_topology = build_cad_topology(&geometry, &topology).expect("cad topology");

    let model = build_cad_evaluation_model(&cad_topology, &topology).expect("evaluation model");
    let frame = model
        .face_frames
        .iter()
        .find(|frame| frame.exact_query_backed)
        .expect("one frame should be exact-query backed");

    assert_eq!(frame.unit_normal, [0.0, 0.0, 1.0]);
    assert_eq!(frame.v_axis, [0.0, 1.0, 0.0]);
}

#[test]
fn exact_backend_query_uses_projected_point_for_frame_origin() {
    let topology = cube_topology();
    let mut geometry = geometry_with_face_evaluator();
    geometry.source_geometry.cad_evaluators[0].faces[0].evaluation_samples =
        vec![CadFaceEvaluationSample {
            source: CadFaceEvaluationSampleSource::BackendQuery,
            point_m: [0.5, 0.5, 1.02],
            uv: Some([0.5, 0.5]),
            projected_point_m: Some([0.5, 0.5, 1.0]),
            unit_normal: Some([0.0, 0.0, 1.0]),
            projection_error_m: Some(0.02),
        }];
    let cad_topology = build_cad_topology(&geometry, &topology).expect("cad topology");

    let model = build_cad_evaluation_model(&cad_topology, &topology).expect("evaluation model");
    let frame = model
        .face_frames
        .iter()
        .find(|frame| frame.exact_query_backed)
        .expect("one frame should be exact-query backed");

    assert_eq!(frame.origin_m, [0.5, 0.5, 1.0]);
    assert_eq!(frame.evaluator_samples[0].point_m, [0.5, 0.5, 1.02]);
    assert_eq!(
        frame.evaluator_samples[0].projected_point_m,
        Some([0.5, 0.5, 1.0])
    );
    assert_eq!(frame.evaluator_max_projection_error_m, 0.02);
}

#[test]
fn exact_backend_query_samples_drive_matching_projection() {
    let topology = cube_topology();
    let mut geometry = geometry_with_face_evaluator();
    geometry.source_geometry.cad_evaluators[0].faces[0].evaluation_samples =
        vec![CadFaceEvaluationSample {
            source: CadFaceEvaluationSampleSource::BackendQuery,
            point_m: [0.5, 0.5, 1.02],
            uv: Some([0.5, 0.5]),
            projected_point_m: Some([0.5, 0.5, 1.0]),
            unit_normal: Some([0.0, 0.0, 2.0]),
            projection_error_m: Some(0.02),
        }];
    let cad_topology = build_cad_topology(&geometry, &topology).expect("cad topology");

    let model = build_cad_evaluation_model(&cad_topology, &topology).expect("evaluation model");
    let frame = model
        .face_frames
        .iter()
        .find(|frame| frame.exact_query_backed)
        .expect("one frame should be exact-query backed");
    let query_projection = project_to_face(frame, [0.5, 0.5, 1.02]);
    let projected_point_projection = project_to_face(frame, [0.5, 0.5, 1.0]);
    let fallback_projection = project_to_face(frame, [0.25, 0.25, 1.02]);

    assert_eq!(query_projection.point_m, [0.5, 0.5, 1.0]);
    assert_eq!(query_projection.uv, [0.5, 0.5]);
    assert!((query_projection.distance_m - 0.02).abs() <= 1.0e-12);
    assert_eq!(query_projection.unit_normal, [0.0, 0.0, 1.0]);
    assert_eq!(projected_point_projection.point_m, [0.5, 0.5, 1.0]);
    assert_eq!(projected_point_projection.uv, [0.5, 0.5]);
    assert_ne!(fallback_projection.uv, [0.5, 0.5]);
}

#[test]
fn merged_cad_face_samples_are_filtered_to_source_triangle_frames() {
    let topology = cube_topology();
    let mut geometry = geometry_with_face_evaluator();
    geometry.source_geometry.cad_evaluators[0].faces[0].evaluation_samples = vec![
        CadFaceEvaluationSample {
            source: CadFaceEvaluationSampleSource::BackendQuery,
            point_m: [0.75, 0.25, 1.0],
            uv: Some([0.75, 0.25]),
            projected_point_m: Some([0.75, 0.25, 1.0]),
            unit_normal: Some([0.0, 0.0, 1.0]),
            projection_error_m: Some(0.0),
        },
        CadFaceEvaluationSample {
            source: CadFaceEvaluationSampleSource::BackendQuery,
            point_m: [0.25, 0.75, 1.0],
            uv: Some([0.25, 0.75]),
            projected_point_m: Some([0.25, 0.75, 1.0]),
            unit_normal: Some([0.0, 0.0, 1.0]),
            projection_error_m: Some(0.0),
        },
    ];
    let cad_topology = build_cad_topology(&geometry, &topology).expect("cad topology");

    let model = build_cad_evaluation_model(&cad_topology, &topology).expect("evaluation model");
    let lower_triangle_frame = model
        .face_frames
        .iter()
        .find(|frame| frame.source_face_id == 2)
        .expect("lower source triangle frame");
    let upper_triangle_frame = model
        .face_frames
        .iter()
        .find(|frame| frame.source_face_id == 3)
        .expect("upper source triangle frame");

    assert_eq!(model.report.exact_query_face_count, 2);
    assert_eq!(model.report.evaluator_sample_count, 2);
    assert_eq!(model.report.evaluator_rejected_sample_count, 2);
    assert_eq!(lower_triangle_frame.evaluator_samples.len(), 1);
    assert_eq!(upper_triangle_frame.evaluator_samples.len(), 1);
    assert_eq!(lower_triangle_frame.origin_m, [0.75, 0.25, 1.0]);
    assert_eq!(upper_triangle_frame.origin_m, [0.25, 0.75, 1.0]);
}

#[test]
fn exact_backend_query_prefers_lowest_projection_error_sample() {
    let topology = cube_topology();
    let mut geometry = geometry_with_face_evaluator();
    geometry.source_geometry.cad_evaluators[0].faces[0].evaluation_samples = vec![
        CadFaceEvaluationSample {
            source: CadFaceEvaluationSampleSource::BackendQuery,
            point_m: [0.7, 0.5, 1.01],
            uv: Some([0.7, 0.5]),
            projected_point_m: Some([0.7, 0.5, 1.0]),
            unit_normal: Some([0.0, 0.0, 1.0]),
            projection_error_m: Some(0.01),
        },
        CadFaceEvaluationSample {
            source: CadFaceEvaluationSampleSource::BackendQuery,
            point_m: [0.8, 0.5, 1.001],
            uv: Some([0.8, 0.5]),
            projected_point_m: Some([0.8, 0.5, 1.0]),
            unit_normal: Some([0.0, 0.0, 1.0]),
            projection_error_m: Some(0.001),
        },
    ];
    let cad_topology = build_cad_topology(&geometry, &topology).expect("cad topology");

    let model = build_cad_evaluation_model(&cad_topology, &topology).expect("evaluation model");
    let frame = model
        .face_frames
        .iter()
        .find(|frame| frame.exact_query_backed)
        .expect("one frame should be exact-query backed");

    assert_eq!(frame.origin_m, [0.8, 0.5, 1.0]);
    assert_eq!(frame.unit_normal, [0.0, 0.0, 1.0]);
    assert_eq!(frame.evaluator_max_projection_error_m, 0.01);
    assert_eq!(frame.evaluator_samples.len(), 2);
}

#[test]
fn exact_backend_query_prefers_measured_projection_error_over_unknown() {
    let topology = cube_topology();
    let mut geometry = geometry_with_face_evaluator();
    geometry.source_geometry.cad_evaluators[0].faces[0].evaluation_samples = vec![
        CadFaceEvaluationSample {
            source: CadFaceEvaluationSampleSource::BackendQuery,
            point_m: [0.6, 0.5, 1.0],
            uv: Some([0.6, 0.5]),
            projected_point_m: Some([0.6, 0.5, 1.0]),
            unit_normal: Some([0.0, 0.0, 1.0]),
            projection_error_m: None,
        },
        CadFaceEvaluationSample {
            source: CadFaceEvaluationSampleSource::BackendQuery,
            point_m: [0.8, 0.5, 1.002],
            uv: Some([0.8, 0.5]),
            projected_point_m: Some([0.8, 0.5, 1.0]),
            unit_normal: Some([0.0, 0.0, 1.0]),
            projection_error_m: Some(0.002),
        },
    ];
    let cad_topology = build_cad_topology(&geometry, &topology).expect("cad topology");

    let model = build_cad_evaluation_model(&cad_topology, &topology).expect("evaluation model");
    let frame = model
        .face_frames
        .iter()
        .find(|frame| frame.exact_query_backed)
        .expect("one frame should be exact-query backed");

    assert_eq!(frame.origin_m, [0.8, 0.5, 1.0]);
    assert_eq!(frame.evaluator_max_projection_error_m, 0.002);
    assert_eq!(frame.evaluator_samples.len(), 2);
}

#[test]
fn evaluator_sample_report_counts_invalid_and_over_budget_samples() {
    let topology = cube_topology();
    let mut geometry = geometry_with_face_evaluator();
    geometry.source_geometry.cad_evaluators[0].faces[0].evaluation_samples = (0..10)
        .map(|index| CadFaceEvaluationSample {
            source: CadFaceEvaluationSampleSource::BackendQuery,
            point_m: [index as f64 * 0.01, index as f64 * 0.01, 1.0],
            uv: Some([index as f64 * 0.01, index as f64 * 0.01]),
            projected_point_m: Some([index as f64 * 0.01, index as f64 * 0.01, 1.0]),
            unit_normal: Some([0.0, 0.0, 1.0]),
            projection_error_m: Some(0.0),
        })
        .chain(std::iter::once(CadFaceEvaluationSample {
            source: CadFaceEvaluationSampleSource::BackendQuery,
            point_m: [f64::NAN, 0.5, 1.0],
            uv: Some([0.5, 0.5]),
            projected_point_m: Some([0.5, 0.5, 1.0]),
            unit_normal: Some([0.0, 0.0, 1.0]),
            projection_error_m: Some(0.0),
        }))
        .collect();
    let cad_topology = build_cad_topology(&geometry, &topology).expect("cad topology");

    let model = build_cad_evaluation_model(&cad_topology, &topology).expect("evaluation model");
    let frame = model
        .face_frames
        .iter()
        .find(|frame| frame.evaluator_samples.len() == 8)
        .expect("sample-backed frame should retain bounded valid samples");

    assert_eq!(frame.evaluator_sample_count, 8);
    assert_eq!(frame.evaluator_rejected_sample_count, 3);
    assert_eq!(model.report.evaluator_sample_count, 16);
    assert_eq!(model.report.evaluator_rejected_sample_count, 6);
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
