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
use runmat_meshing_curve::{discretize_topology_curves, CurveDiscretizationOptions};

mod fixtures;

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
    assert_eq!(surface.elements[0].region_ids, vec!["face_a".to_string()]);
    assert!((surface.elements[0].area_m2 - 0.5).abs() < 1.0e-12);
}

#[test]
fn discretizes_surfaces_with_cad_face_ownership() {
    let topology = single_triangle_topology();
    let cad_topology =
        build_cad_topology(&geometry_for_topology(), &topology).expect("cad topology");
    let cad_evaluation =
        build_cad_evaluation_model(&cad_topology, &topology).expect("cad evaluation");

    let surface = discretize_cad_surfaces(
        &topology,
        &cad_evaluation,
        SurfaceDiscretizationOptions::default(),
    )
    .expect("cad-owned surface should discretize");

    assert_eq!(surface.elements.len(), 1);
    assert_eq!(
        surface.elements[0].cad_face_id,
        Some("cad_face_7".to_string())
    );
    assert_eq!(surface.elements[0].parametric_node_uv.len(), 3);
    assert_eq!(surface.elements[0].max_projection_error_m, 0.0);
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
fn centroid_subdivision_preserves_cad_face_ownership_and_boundary_edges() {
    let topology = single_triangle_topology();
    let cad_topology =
        build_cad_topology(&geometry_for_topology(), &topology).expect("cad topology");
    let cad_evaluation =
        build_cad_evaluation_model(&cad_topology, &topology).expect("cad evaluation");

    let surface = discretize_cad_surfaces(
        &topology,
        &cad_evaluation,
        SurfaceDiscretizationOptions {
            centroid_subdivision: true,
            ..SurfaceDiscretizationOptions::default()
        },
    )
    .expect("cad-owned surface should subdivide");

    assert_eq!(surface.nodes.len(), 4);
    assert_eq!(surface.elements.len(), 3);
    assert!(surface
        .elements
        .iter()
        .all(|element| element.cad_face_id == Some("cad_face_7".to_string())));
    assert!(surface
        .elements
        .iter()
        .any(|element| { element.node_ids[0..2] == [0, 1] && element.source_edge_ids[0] == 0 }));
    assert_eq!(surface.nodes[3].coordinates_m, [1.0 / 3.0, 1.0 / 3.0, 0.0]);
}

#[test]
fn curve_driven_cad_surface_uses_curve_boundary_nodes() {
    let topology = single_triangle_topology();
    let cad_topology =
        build_cad_topology(&geometry_for_topology(), &topology).expect("cad topology");
    let cad_evaluation =
        build_cad_evaluation_model(&cad_topology, &topology).expect("cad evaluation");
    let curves = discretize_topology_curves(
        &topology,
        CurveDiscretizationOptions {
            target_size_m: 0.25,
            min_segments_per_edge: 2,
            max_segments_per_edge: 2,
        },
    )
    .expect("curves should discretize");

    let surface = discretize_cad_surfaces_with_curves(
        &topology,
        &cad_evaluation,
        &curves,
        SurfaceDiscretizationOptions {
            max_curve_segments_per_edge: 2,
            ..SurfaceDiscretizationOptions::default()
        },
    )
    .expect("cad-owned curve surface should discretize");

    assert_eq!(surface.elements.len(), 4);
    assert!(surface.nodes.len() > topology.vertices.len());
    assert!(surface
        .elements
        .iter()
        .all(|element| element.cad_face_id == Some("cad_face_7".to_string())));
    assert!(surface.elements.iter().any(|element| {
        element
            .source_edge_ids
            .iter()
            .any(|edge_id| *edge_id != INTERNAL_SOURCE_EDGE_ID)
    }));
    assert!(surface.elements.iter().any(|element| {
        element
            .source_edge_ids
            .iter()
            .any(|edge_id| *edge_id == INTERNAL_SOURCE_EDGE_ID)
    }));
}

#[test]
fn curve_driven_cad_surface_preserves_single_triangle_loop_without_extra_fan_node() {
    let topology = single_triangle_topology();
    let cad_topology =
        build_cad_topology(&geometry_for_topology(), &topology).expect("cad topology");
    let cad_evaluation =
        build_cad_evaluation_model(&cad_topology, &topology).expect("cad evaluation");
    let curves = discretize_topology_curves(
        &topology,
        CurveDiscretizationOptions {
            target_size_m: 10.0,
            min_segments_per_edge: 1,
            max_segments_per_edge: 1,
        },
    )
    .expect("curves should discretize");

    let surface = discretize_cad_surfaces_with_curves(
        &topology,
        &cad_evaluation,
        &curves,
        SurfaceDiscretizationOptions::default(),
    )
    .expect("cad-owned curve surface should discretize");

    assert_eq!(surface.nodes.len(), topology.vertices.len());
    assert_eq!(surface.elements.len(), 1);
    assert_eq!(surface.elements[0].node_ids, [0, 1, 2]);
    assert!(surface.elements[0]
        .source_edge_ids
        .iter()
        .all(|edge_id| *edge_id != INTERNAL_SOURCE_EDGE_ID));
}

#[test]
fn curve_boundary_fan_orients_elements_to_cad_frame() {
    let face = single_triangle_topology().faces[0].clone();
    let frame = planar_test_frame(face.face_id);
    let mut nodes = vec![
        SurfaceNode {
            node_id: 0,
            source_vertex_id: 0,
            coordinates_m: [0.0, 0.0, 0.0],
        },
        SurfaceNode {
            node_id: 1,
            source_vertex_id: 1,
            coordinates_m: [1.0, 0.0, 0.0],
        },
        SurfaceNode {
            node_id: 2,
            source_vertex_id: 2,
            coordinates_m: [0.0, 1.0, 0.0],
        },
    ];
    let segments = [
        FaceCurveSegment {
            node_ids: [1, 0],
            source_edge_id: 0,
        },
        FaceCurveSegment {
            node_ids: [2, 1],
            source_edge_id: 1,
        },
        FaceCurveSegment {
            node_ids: [0, 2],
            source_edge_id: 2,
        },
    ];
    let mut elements = Vec::<SurfaceElement>::new();

    append_curve_fan_face_elements(&face, &frame, &segments, &mut nodes, &mut elements);

    assert_eq!(elements.len(), 3);
    assert!(elements.iter().all(|element| {
        let points = element
            .node_ids
            .map(|node_id| nodes[node_id as usize].coordinates_m);
        dot(
            cross(sub(points[1], points[0]), sub(points[2], points[0])),
            frame.unit_normal,
        ) > 0.0
    }));
    for source_edge_id in [0, 1, 2] {
        assert!(elements
            .iter()
            .any(|element| element.source_edge_ids.contains(&source_edge_id)));
    }
}

#[test]
fn curve_driven_cad_surface_uses_exact_face_domain_samples() {
    let topology = single_triangle_topology();
    let cad_topology =
        build_cad_topology(&geometry_with_face_domain_sample(), &topology).expect("cad topology");
    let cad_evaluation =
        build_cad_evaluation_model(&cad_topology, &topology).expect("cad evaluation");
    assert_eq!(cad_evaluation.report.evaluator_rejected_sample_count, 1);
    let curves = discretize_topology_curves(
        &topology,
        CurveDiscretizationOptions {
            target_size_m: 1.0,
            min_segments_per_edge: 1,
            max_segments_per_edge: 1,
        },
    )
    .expect("curves should discretize");

    let surface = discretize_cad_surfaces_with_curves(
        &topology,
        &cad_evaluation,
        &curves,
        SurfaceDiscretizationOptions {
            max_curve_segments_per_edge: 1,
            ..SurfaceDiscretizationOptions::default()
        },
    )
    .expect("cad-owned curve surface should discretize");

    assert_eq!(surface.nodes.len(), topology.vertices.len() + 1);
    assert!(surface.elements.len() >= 2);
    assert_eq!(surface.exact_cad_sample_node_count, 1);
    assert_eq!(surface.rejected_exact_cad_sample_count, 0);
    assert!(surface
        .nodes
        .iter()
        .any(|node| node.coordinates_m == [0.25, 0.25, 0.0]));
    assert!(
        (surface
            .elements
            .iter()
            .map(|element| element.area_m2)
            .sum::<f64>()
            - topology.faces[0].area_m2)
            .abs()
            <= 1.0e-12
    );
    assert!(surface
        .elements
        .iter()
        .all(|element| element.cad_face_id == Some("face_a".to_string())));
}

#[test]
fn curve_driven_cad_surface_preserves_area_with_multiple_exact_samples() {
    let topology = single_triangle_topology();
    let cad_topology = build_cad_topology(&geometry_with_area_regressing_face_samples(), &topology)
        .expect("cad topology");
    let cad_evaluation =
        build_cad_evaluation_model(&cad_topology, &topology).expect("cad evaluation");
    let curves = discretize_topology_curves(
        &topology,
        CurveDiscretizationOptions {
            target_size_m: 1.0,
            min_segments_per_edge: 1,
            max_segments_per_edge: 1,
        },
    )
    .expect("curves should discretize");

    let surface = discretize_cad_surfaces_with_curves(
        &topology,
        &cad_evaluation,
        &curves,
        SurfaceDiscretizationOptions {
            max_curve_segments_per_edge: 1,
            ..SurfaceDiscretizationOptions::default()
        },
    )
    .expect("cad-owned curve surface should discretize");
    let recovered_area = surface
        .elements
        .iter()
        .filter(|element| element.source_face_id == 7)
        .map(|element| element.area_m2)
        .sum::<f64>();

    assert_eq!(surface.nodes.len(), topology.vertices.len() + 3);
    assert_eq!(surface.elements.len(), 7);
    assert_eq!(surface.exact_cad_sample_node_count, 3);
    assert_eq!(surface.rejected_exact_cad_sample_count, 0);
    assert!((recovered_area - topology.faces[0].area_m2).abs() <= 1.0e-12);
    assert!(surface
        .elements
        .iter()
        .all(|element| element.cad_face_id == Some("face_a".to_string())));
}

#[test]
fn curve_driven_cad_surface_splits_edge_hit_exact_samples_without_cracks() {
    let topology = single_triangle_topology();
    let cad_topology = build_cad_topology(&geometry_with_edge_hit_face_samples(), &topology)
        .expect("cad topology");
    let cad_evaluation =
        build_cad_evaluation_model(&cad_topology, &topology).expect("cad evaluation");
    let curves = discretize_topology_curves(
        &topology,
        CurveDiscretizationOptions {
            target_size_m: 1.0,
            min_segments_per_edge: 1,
            max_segments_per_edge: 1,
        },
    )
    .expect("curves should discretize");

    let surface = discretize_cad_surfaces_with_curves(
        &topology,
        &cad_evaluation,
        &curves,
        SurfaceDiscretizationOptions {
            max_curve_segments_per_edge: 1,
            ..SurfaceDiscretizationOptions::default()
        },
    )
    .expect("cad-owned curve surface should discretize");
    let recovered_area = surface
        .elements
        .iter()
        .filter(|element| element.source_face_id == 7)
        .map(|element| element.area_m2)
        .sum::<f64>();

    assert_eq!(surface.exact_cad_sample_node_count, 2);
    assert_eq!(surface.rejected_exact_cad_sample_count, 0);
    assert!((recovered_area - topology.faces[0].area_m2).abs() <= 1.0e-12);
    assert_local_surface_edges_are_recovered(&surface.elements);
}

#[test]
fn curve_driven_cad_surface_rejects_samples_outside_concave_trim_loop() {
    let mut topology = single_triangle_topology();
    topology.faces[0].area_m2 = 0.275;
    let cad_topology = build_cad_topology(&geometry_with_concave_trim_rejected_sample(), &topology)
        .expect("cad topology");
    let cad_evaluation =
        build_cad_evaluation_model(&cad_topology, &topology).expect("cad evaluation");
    let curves = concave_trim_curve_discretization();

    let surface = discretize_cad_surfaces_with_curves(
        &topology,
        &cad_evaluation,
        &curves,
        SurfaceDiscretizationOptions {
            max_curve_segments_per_edge: 2,
            ..SurfaceDiscretizationOptions::default()
        },
    )
    .expect("concave trimmed surface should discretize");
    let recovered_area = surface
        .elements
        .iter()
        .filter(|element| element.source_face_id == 7)
        .map(|element| element.area_m2)
        .sum::<f64>();
    let trim_loop = [[0.0, 0.0], [0.5, 0.45], [1.0, 0.0], [0.0, 1.0]];

    assert_eq!(surface.exact_cad_sample_node_count, 0);
    assert_eq!(surface.rejected_exact_cad_sample_count, 1);
    assert!((recovered_area - topology.faces[0].area_m2).abs() <= 1.0e-12);
    assert!(!surface
        .nodes
        .iter()
        .any(|node| node.coordinates_m == [0.5, 0.2, 0.0]));
    assert!(surface.elements.iter().all(|element| {
        let centroid = triangle_centroid_2d(element.node_ids.map(|node_id| {
            let point = surface.nodes[node_id as usize].coordinates_m;
            [point[0], point[1]]
        }));
        point_in_polygon_2d(centroid, &trim_loop)
    }));
    assert_surface_edges_are_recovered(&surface.elements, &[[0, 3], [1, 3], [1, 2], [0, 2]]);
}

#[test]
fn curve_driven_face_elements_triangulate_holed_loop_domain() {
    let face = SourceTopologyFace {
        face_id: 7,
        source_triangle_id: 11,
        node_ids: [0, 1, 2],
        edge_ids: [0, 1, 2],
        region_ids: vec!["face_a".to_string()],
        area_m2: 0.96,
        unit_normal: [0.0, 0.0, 1.0],
    };
    let frame = planar_test_frame(7);
    let mut nodes = square_with_square_hole_surface_nodes();
    let segment_loops = vec![
        vec![
            FaceCurveSegment {
                node_ids: [0, 1],
                source_edge_id: 0,
            },
            FaceCurveSegment {
                node_ids: [1, 2],
                source_edge_id: 1,
            },
            FaceCurveSegment {
                node_ids: [2, 3],
                source_edge_id: 2,
            },
            FaceCurveSegment {
                node_ids: [3, 0],
                source_edge_id: 3,
            },
        ],
        vec![
            FaceCurveSegment {
                node_ids: [4, 5],
                source_edge_id: 4,
            },
            FaceCurveSegment {
                node_ids: [5, 6],
                source_edge_id: 5,
            },
            FaceCurveSegment {
                node_ids: [6, 7],
                source_edge_id: 6,
            },
            FaceCurveSegment {
                node_ids: [7, 4],
                source_edge_id: 7,
            },
        ],
    ];
    let mut elements = Vec::<SurfaceElement>::new();

    let report =
        append_curve_driven_face_elements(&face, &frame, &segment_loops, &mut nodes, &mut elements);
    let recovered_area = elements.iter().map(|element| element.area_m2).sum::<f64>();
    let hole = [[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]];

    assert_eq!(report, ExactCadSampleSurfaceReport::default());
    assert!(!elements.is_empty());
    assert!(
        (recovered_area - face.area_m2).abs() <= 1.0e-12,
        "recovered_area={recovered_area} expected_area={} element_count={}",
        face.area_m2,
        elements.len()
    );
    assert!(elements.iter().all(|element| {
        let centroid = triangle_centroid_2d(element.node_ids.map(|node_id| {
            let point = nodes[node_id as usize].coordinates_m;
            [point[0], point[1]]
        }));
        !point_in_polygon_2d(centroid, &hole)
    }));
    assert_surface_edges_are_recovered(
        &elements,
        &[
            [0, 1],
            [1, 2],
            [2, 3],
            [0, 3],
            [4, 5],
            [5, 6],
            [6, 7],
            [4, 7],
        ],
    );
}

#[test]
fn single_loop_extractor_reports_multiple_face_curve_loops() {
    let segments = vec![
        FaceCurveSegment {
            node_ids: [0, 1],
            source_edge_id: 0,
        },
        FaceCurveSegment {
            node_ids: [1, 2],
            source_edge_id: 1,
        },
        FaceCurveSegment {
            node_ids: [2, 0],
            source_edge_id: 2,
        },
        FaceCurveSegment {
            node_ids: [3, 4],
            source_edge_id: 3,
        },
        FaceCurveSegment {
            node_ids: [4, 5],
            source_edge_id: 4,
        },
        FaceCurveSegment {
            node_ids: [5, 3],
            source_edge_id: 5,
        },
    ];

    let err = single_face_curve_segment_loop(7, &segments)
        .expect_err("multi-loop face topology should fail closed");

    assert_eq!(
        err,
        SurfaceDiscretizationError::MultipleFaceLoopsUnsupported {
            face_id: 7,
            loop_count: 2,
            loop_node_counts: vec![3, 3],
            loop_source_edge_ids: vec![vec![0, 1, 2], vec![3, 4, 5]],
        }
    );
    assert!(err
        .to_string()
        .contains("boundary loops with node counts [3, 3]"));
    assert!(err
        .to_string()
        .contains("source edge loops [[0, 1, 2], [3, 4, 5]]"));
}

#[test]
fn face_curve_segment_loops_order_shuffled_single_loop_deterministically() {
    let segments = vec![
        FaceCurveSegment {
            node_ids: [3, 0],
            source_edge_id: 13,
        },
        FaceCurveSegment {
            node_ids: [1, 2],
            source_edge_id: 11,
        },
        FaceCurveSegment {
            node_ids: [2, 3],
            source_edge_id: 12,
        },
        FaceCurveSegment {
            node_ids: [0, 1],
            source_edge_id: 10,
        },
    ];

    let loops = face_curve_segment_loops(7, &segments).expect("loop should be valid");

    assert_eq!(loops.len(), 1);
    assert_eq!(
        loops[0],
        vec![
            FaceCurveSegment {
                node_ids: [0, 1],
                source_edge_id: 10,
            },
            FaceCurveSegment {
                node_ids: [1, 2],
                source_edge_id: 11,
            },
            FaceCurveSegment {
                node_ids: [2, 3],
                source_edge_id: 12,
            },
            FaceCurveSegment {
                node_ids: [3, 0],
                source_edge_id: 13,
            },
        ]
    );
}

#[test]
fn rejects_open_face_curve_loop_before_triangulation() {
    let segments = vec![
        FaceCurveSegment {
            node_ids: [0, 1],
            source_edge_id: 0,
        },
        FaceCurveSegment {
            node_ids: [1, 2],
            source_edge_id: 1,
        },
    ];

    let err = single_face_curve_segment_loop(7, &segments)
        .expect_err("open face loop should fail closed");

    assert_eq!(
        err,
        SurfaceDiscretizationError::InvalidFaceLoopTopology {
            face_id: 7,
            node_id: 0,
            incident_segment_count: 1,
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
