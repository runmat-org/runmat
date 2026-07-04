use super::*;
use super::{
    boundary::{single_face_curve_segment_loop, FaceCurveSegment},
    elements::append_curve_fan_face_elements,
    geometry::{point_in_polygon_2d, triangle_centroid_2d},
    sampling::ExactCadSampleSurfaceReport,
};
use crate::math::{cross, dot, sub};
use runmat_geometry_core::{CadFaceEvaluationSample, CadFaceEvaluationSampleSource};
use runmat_meshing_cad::{build_cad_evaluation_model, build_cad_topology, SourceTopologyFace};
use runmat_meshing_curve::{
    discretize_topology_curves, CurveDiscretizationOptions, CurveValidationError,
};

mod cad_surfaces;
mod exact_samples;
mod fixtures;
mod loop_topology;

use fixtures::*;

#[test]
fn discretizes_source_faces_as_surface_elements() {
    let surface = discretize_topology_surfaces(
        &single_triangle_topology(),
        SurfaceDiscretizationOptions::default(),
    )
    .expect("surface should discretize");

    assert_eq!(surface.nodes.len(), 3);
    assert_eq!(surface.elements.len(), 1);
    assert_eq!(surface.elements[0].source_face_id, 7);
    assert_eq!(surface.elements[0].cad_face_id, None);
    assert_eq!(surface.elements[0].source_edge_ids, [0, 1, 2]);
    assert_eq!(surface.elements[0].parametric_node_uv, [[0.0, 0.0]; 3]);
    assert_eq!(surface.elements[0].max_projection_error_m, 0.0);
    assert!(surface.curve_boundary_validation.is_none());
    assert!(surface.loop_coverage.is_none());
    assert_eq!(surface.elements[0].region_ids, vec!["face_a".to_string()]);
    assert!((surface.elements[0].area_m2 - 0.5).abs() < 1.0e-12);
}

#[test]
fn rejects_cad_surface_vertex_outside_uv_domain() {
    let topology = single_triangle_topology();
    let mut geometry = geometry_with_face_domain_sample();
    geometry.source_geometry.cad_evaluators[0].faces[0].evaluation_samples = vec![
        CadFaceEvaluationSample {
            source: CadFaceEvaluationSampleSource::BackendQuery,
            point_m: [0.0, 0.0, 0.0],
            uv: Some([0.0, 0.0]),
            projected_point_m: Some([0.0, 0.0, 0.0]),
            unit_normal: Some([0.0, 0.0, 1.0]),
            projection_error_m: Some(0.0),
        },
        CadFaceEvaluationSample {
            source: CadFaceEvaluationSampleSource::BackendQuery,
            point_m: [0.5, 0.0, 0.0],
            uv: Some([0.5, 0.0]),
            projected_point_m: Some([0.5, 0.0, 0.0]),
            unit_normal: Some([0.0, 0.0, 1.0]),
            projection_error_m: Some(0.0),
        },
        CadFaceEvaluationSample {
            source: CadFaceEvaluationSampleSource::BackendQuery,
            point_m: [0.0, 0.5, 0.0],
            uv: Some([0.0, 0.5]),
            projected_point_m: Some([0.0, 0.5, 0.0]),
            unit_normal: Some([0.0, 0.0, 1.0]),
            projection_error_m: Some(0.0),
        },
    ];
    let cad_topology = build_cad_topology(&geometry, &topology).expect("cad topology");
    let cad_evaluation =
        build_cad_evaluation_model(&cad_topology, &topology).expect("cad evaluation");

    let err = discretize_cad_surfaces(
        &topology,
        &cad_evaluation,
        SurfaceDiscretizationOptions::default(),
    )
    .expect_err("out-of-domain source vertex should fail");

    assert_eq!(
        err,
        SurfaceDiscretizationError::CadProjectionOutsideFaceDomain {
            face_id: 7,
            node_id: 1,
        }
    );
}

#[test]
fn rejects_missing_face_vertices() {
    let mut topology = single_triangle_topology();
    topology.vertices.pop();

    let err = discretize_topology_surfaces(&topology, SurfaceDiscretizationOptions::default())
        .expect_err("missing face vertex should fail");

    assert_eq!(
        err,
        SurfaceDiscretizationError::MissingFaceVertex {
            face_id: 7,
            node_id: 2,
        }
    );
}

#[test]
fn rejects_invalid_curve_boundary_before_surface_triangulation() {
    let topology = single_triangle_topology();
    let geometry = geometry_with_face_domain_sample();
    let cad_topology = build_cad_topology(&geometry, &topology).expect("cad topology");
    let cad_evaluation =
        build_cad_evaluation_model(&cad_topology, &topology).expect("cad evaluation");
    let mut curves = discretize_topology_curves(
        &topology,
        CurveDiscretizationOptions {
            target_size_m: 0.5,
            min_segments_per_edge: 1,
            max_segments_per_edge: 8,
        },
    )
    .expect("curves should discretize");
    curves
        .nodes
        .iter_mut()
        .find(|node| node.source_edge_id == 0 && node.parameter == 0.0)
        .expect("source edge endpoint")
        .coordinates_m = [0.01, 0.0, 0.0];

    let err = discretize_cad_surfaces_with_curves(
        &topology,
        &cad_evaluation,
        &curves,
        SurfaceDiscretizationOptions::default(),
    )
    .expect_err("invalid curve boundary should fail before surface triangulation");

    assert!(matches!(
        err,
        SurfaceDiscretizationError::InvalidCurveBoundary(CurveValidationError::EndpointDrift {
            source_edge_id: 0,
            parameter: 0.0,
            ..
        })
    ));
}
