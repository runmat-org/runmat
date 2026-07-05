use runmat_meshing_cad::{
    CadFaceEvaluationFrame, SourceTopologyEdge, SourceTopologyFace, SourceTopologyModel,
    SourceTopologyVertex,
};

use super::super::SurfaceNode;

mod assertions;
mod curves;
mod geometry;

pub(super) use assertions::{
    assert_local_surface_edges_are_recovered, assert_surface_edges_are_recovered,
};
pub(super) use curves::concave_trim_curve_discretization;
pub(super) use geometry::{
    geometry_for_topology, geometry_with_area_regressing_face_samples,
    geometry_with_concave_trim_rejected_sample, geometry_with_edge_hit_face_samples,
    geometry_with_face_domain_sample,
};

pub(super) fn single_triangle_topology() -> SourceTopologyModel {
    SourceTopologyModel {
        mesh_id: "surface".to_string(),
        source_geometry_id: "geo".to_string(),
        source_geometry_revision: 1,
        source_geometry_sha256: None,
        vertices: vec![
            SourceTopologyVertex {
                vertex_id: 0,
                coordinates_m: [0.0, 0.0, 0.0],
            },
            SourceTopologyVertex {
                vertex_id: 1,
                coordinates_m: [1.0, 0.0, 0.0],
            },
            SourceTopologyVertex {
                vertex_id: 2,
                coordinates_m: [0.0, 1.0, 0.0],
            },
        ],
        edges: vec![
            SourceTopologyEdge {
                edge_id: 0,
                node_ids: [0, 1],
                adjacent_face_ids: vec![7],
                region_ids: vec!["face_a".to_string()],
                length_m: 1.0,
            },
            SourceTopologyEdge {
                edge_id: 1,
                node_ids: [1, 2],
                adjacent_face_ids: vec![7],
                region_ids: vec!["face_a".to_string()],
                length_m: 2.0_f64.sqrt(),
            },
            SourceTopologyEdge {
                edge_id: 2,
                node_ids: [0, 2],
                adjacent_face_ids: vec![7],
                region_ids: vec!["face_a".to_string()],
                length_m: 1.0,
            },
        ],
        faces: vec![SourceTopologyFace {
            face_id: 7,
            source_triangle_id: 11,
            node_ids: [0, 1, 2],
            edge_ids: [0, 1, 2],
            region_ids: vec!["face_a".to_string()],
            material_region_ids: Vec::new(),
            area_m2: 0.5,
            unit_normal: [0.0, 0.0, 1.0],
        }],
        bounds_min_m: [0.0, 0.0, 0.0],
        bounds_max_m: [1.0, 1.0, 0.0],
        region_ids: vec!["face_a".to_string()],
        material_region_ids: Vec::new(),
    }
}

pub(super) fn planar_test_frame(source_face_id: u32) -> CadFaceEvaluationFrame {
    CadFaceEvaluationFrame {
        face_id: "face_a".to_string(),
        source_face_id,
        origin_m: [0.0, 0.0, 0.0],
        u_axis: [1.0, 0.0, 0.0],
        v_axis: [0.0, 1.0, 0.0],
        unit_normal: [0.0, 0.0, 1.0],
        area_m2: 1.0,
        evaluator_backed: false,
        exact_query_backed: false,
        live_query_backed: false,
        evaluator_sample_count: 0,
        evaluator_rejected_sample_count: 0,
        evaluator_max_projection_error_m: 0.0,
        evaluator_samples: Vec::new(),
        u_derivative_m_per_uv: None,
        v_derivative_m_per_uv: None,
        max_curvature_estimate_1_per_m: None,
        uv_bounds: Some([[0.0, 0.0], [1.0, 1.0]]),
        uv_bounds_sample_count: 4,
        uv_domain_source: Some("test_domain".to_string()),
    }
}

pub(super) fn square_with_square_hole_surface_nodes() -> Vec<SurfaceNode> {
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.4, 0.4, 0.0],
        [0.6, 0.4, 0.0],
        [0.6, 0.6, 0.0],
        [0.4, 0.6, 0.0],
    ]
    .into_iter()
    .enumerate()
    .map(|(node_id, coordinates_m)| SurfaceNode {
        node_id: node_id as u32,
        source_vertex_id: node_id as u32,
        coordinates_m,
    })
    .collect()
}
