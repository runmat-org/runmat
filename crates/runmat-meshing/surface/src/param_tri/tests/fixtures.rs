use std::collections::BTreeMap;

use runmat_geometry_core::{
    CadEvaluatorSet, CadFaceEvaluationSample, CadFaceEvaluationSampleSource, CadFaceEvaluator,
    CadLabelRef, CadRegionOwnership, CadSemanticKind, EntityIdRange, EntityKind, Region,
    RegionEntityMapping,
};
use runmat_meshing_cad::{
    CadFaceEvaluationFrame, SourceTopologyEdge, SourceTopologyFace, SourceTopologyModel,
    SourceTopologyVertex,
};
use runmat_meshing_curve::{CurveDiscretization, CurveElement, CurveNode};

use super::super::{geometry::sorted_node_pair, SurfaceElement, SurfaceNode};

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
            area_m2: 0.5,
            unit_normal: [0.0, 0.0, 1.0],
        }],
        bounds_min_m: [0.0, 0.0, 0.0],
        bounds_max_m: [1.0, 1.0, 0.0],
        region_ids: vec!["face_a".to_string()],
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

pub(super) fn geometry_for_topology() -> runmat_geometry_core::GeometryAsset {
    runmat_geometry_core::GeometryAsset {
        geometry_id: "geo".to_string(),
        source: runmat_geometry_core::GeometrySource {
            path: "/fixtures/surface.step".to_string(),
            sha256: "surface".to_string(),
            importer_version: "test".to_string(),
        },
        source_geometry: runmat_geometry_core::SourceGeometry {
            kind: runmat_geometry_core::SourceGeometryKind::Cad,
            assembly: None,
            material_evidence: Vec::new(),
            cad_evaluators: Vec::new(),
        },
        tessellation_profile: runmat_geometry_core::TessellationProfile::default(),
        units: runmat_geometry_core::UnitSystem::Meter,
        revision: 1,
        meshes: Vec::new(),
        surface_meshes: Vec::new(),
        regions: Vec::new(),
        region_entity_mappings: Vec::new(),
        diagnostics: Vec::new(),
    }
}

pub(super) fn geometry_with_face_domain_sample() -> runmat_geometry_core::GeometryAsset {
    let mut geometry = geometry_for_topology();
    geometry.regions = vec![Region {
        region_id: "face_a".to_string(),
        name: "face".to_string(),
        tag: Some("cad_face".to_string()),
        cad_ownership: Some(CadRegionOwnership {
            face_id: Some(7),
            label: Some(CadLabelRef {
                label_entry: "0:1:7".to_string(),
                name: "face".to_string(),
                kind: CadSemanticKind::Face,
            }),
            owner_path: Vec::new(),
            layers: Vec::new(),
            color: None,
            material: None,
        }),
    }];
    geometry.region_entity_mappings = vec![RegionEntityMapping {
        region_id: "face_a".to_string(),
        mesh_id: "surface".to_string(),
        entity_kind: EntityKind::Face,
        ranges: vec![EntityIdRange {
            start: 11,
            count: 1,
        }],
    }];
    geometry.source_geometry.cad_evaluators = vec![CadEvaluatorSet {
        evaluator_id: "cad_evaluator_test".to_string(),
        backend: "test".to_string(),
        format_name: "step".to_string(),
        requires_source_geometry: true,
        faces: vec![CadFaceEvaluator {
            evaluator_id: "cad_face_7".to_string(),
            imported_face_id: 7,
            name: "face".to_string(),
            supports_point_evaluation: true,
            supports_projection: true,
            supports_normal: true,
            supports_derivatives: true,
            supports_curvature: true,
            reference_point_m: Some([0.25, 0.25, 0.0]),
            reference_unit_normal: Some([0.0, 0.0, 1.0]),
            evaluation_samples: vec![
                CadFaceEvaluationSample {
                    source: CadFaceEvaluationSampleSource::BackendQuery,
                    point_m: [0.25, 0.25, 0.03],
                    uv: Some([0.25, 0.25]),
                    projected_point_m: Some([0.25, 0.25, 0.0]),
                    unit_normal: Some([0.0, 0.0, 1.0]),
                    projection_error_m: Some(0.03),
                },
                CadFaceEvaluationSample {
                    source: CadFaceEvaluationSampleSource::BackendQuery,
                    point_m: [1.25, 0.25, 0.0],
                    uv: Some([1.25, 0.25]),
                    projected_point_m: Some([1.25, 0.25, 0.0]),
                    unit_normal: Some([0.0, 0.0, 1.0]),
                    projection_error_m: Some(0.0),
                },
            ],
        }],
        curves: Vec::new(),
    }];
    geometry
}

pub(super) fn geometry_with_area_regressing_face_samples() -> runmat_geometry_core::GeometryAsset {
    let mut geometry = geometry_with_face_domain_sample();
    geometry.source_geometry.cad_evaluators[0].faces[0].evaluation_samples = vec![
        CadFaceEvaluationSample {
            source: CadFaceEvaluationSampleSource::BackendQuery,
            point_m: [0.30, 0.10, 0.0],
            uv: Some([0.30, 0.10]),
            projected_point_m: Some([0.30, 0.10, 0.0]),
            unit_normal: Some([0.0, 0.0, 1.0]),
            projection_error_m: Some(0.0),
        },
        CadFaceEvaluationSample {
            source: CadFaceEvaluationSampleSource::BackendQuery,
            point_m: [0.75, 0.10, 0.0],
            uv: Some([0.75, 0.10]),
            projected_point_m: Some([0.75, 0.10, 0.0]),
            unit_normal: Some([0.0, 0.0, 1.0]),
            projection_error_m: Some(0.0),
        },
        CadFaceEvaluationSample {
            source: CadFaceEvaluationSampleSource::BackendQuery,
            point_m: [0.70, 0.25, 0.0],
            uv: Some([0.70, 0.25]),
            projected_point_m: Some([0.70, 0.25, 0.0]),
            unit_normal: Some([0.0, 0.0, 1.0]),
            projection_error_m: Some(0.0),
        },
    ];
    geometry
}

pub(super) fn geometry_with_edge_hit_face_samples() -> runmat_geometry_core::GeometryAsset {
    let mut geometry = geometry_with_face_domain_sample();
    geometry.source_geometry.cad_evaluators[0].faces[0].evaluation_samples = vec![
        CadFaceEvaluationSample {
            source: CadFaceEvaluationSampleSource::BackendQuery,
            point_m: [0.50, 0.25, 0.0],
            uv: Some([0.50, 0.25]),
            projected_point_m: Some([0.50, 0.25, 0.0]),
            unit_normal: Some([0.0, 0.0, 1.0]),
            projection_error_m: Some(0.0),
        },
        CadFaceEvaluationSample {
            source: CadFaceEvaluationSampleSource::BackendQuery,
            point_m: [0.25, 0.125, 0.0],
            uv: Some([0.25, 0.125]),
            projected_point_m: Some([0.25, 0.125, 0.0]),
            unit_normal: Some([0.0, 0.0, 1.0]),
            projection_error_m: Some(0.0),
        },
    ];
    geometry
}

pub(super) fn geometry_with_concave_trim_rejected_sample() -> runmat_geometry_core::GeometryAsset {
    let mut geometry = geometry_with_face_domain_sample();
    geometry.source_geometry.cad_evaluators[0].faces[0].evaluation_samples =
        vec![CadFaceEvaluationSample {
            source: CadFaceEvaluationSampleSource::BackendQuery,
            point_m: [0.5, 0.2, 0.0],
            uv: Some([0.5, 0.2]),
            projected_point_m: Some([0.5, 0.2, 0.0]),
            unit_normal: Some([0.0, 0.0, 1.0]),
            projection_error_m: Some(0.0),
        }];
    geometry
}

pub(super) fn concave_trim_curve_discretization() -> CurveDiscretization {
    CurveDiscretization {
        nodes: vec![
            CurveNode {
                node_id: 0,
                source_edge_id: 0,
                parameter: 0.0,
                coordinates_m: [0.0, 0.0, 0.0],
            },
            CurveNode {
                node_id: 1,
                source_edge_id: 0,
                parameter: 0.5,
                coordinates_m: [0.5, 0.45, 0.0],
            },
            CurveNode {
                node_id: 2,
                source_edge_id: 0,
                parameter: 1.0,
                coordinates_m: [1.0, 0.0, 0.0],
            },
            CurveNode {
                node_id: 3,
                source_edge_id: 1,
                parameter: 0.0,
                coordinates_m: [1.0, 0.0, 0.0],
            },
            CurveNode {
                node_id: 4,
                source_edge_id: 1,
                parameter: 1.0,
                coordinates_m: [0.0, 1.0, 0.0],
            },
            CurveNode {
                node_id: 5,
                source_edge_id: 2,
                parameter: 0.0,
                coordinates_m: [0.0, 0.0, 0.0],
            },
            CurveNode {
                node_id: 6,
                source_edge_id: 2,
                parameter: 1.0,
                coordinates_m: [0.0, 1.0, 0.0],
            },
        ],
        elements: vec![
            CurveElement {
                element_id: 0,
                source_edge_id: 0,
                node_ids: [0, 1],
                length_m: 0.6726812023536856,
            },
            CurveElement {
                element_id: 1,
                source_edge_id: 0,
                node_ids: [1, 2],
                length_m: 0.6726812023536856,
            },
            CurveElement {
                element_id: 2,
                source_edge_id: 1,
                node_ids: [3, 4],
                length_m: 2.0_f64.sqrt(),
            },
            CurveElement {
                element_id: 3,
                source_edge_id: 2,
                node_ids: [5, 6],
                length_m: 1.0,
            },
        ],
    }
}

pub(super) fn assert_local_surface_edges_are_recovered(elements: &[SurfaceElement]) {
    assert_surface_edges_are_recovered(elements, &[[0, 1], [0, 2], [1, 2]]);
}

pub(super) fn assert_surface_edges_are_recovered(
    elements: &[SurfaceElement],
    boundary_edges: &[[u32; 2]],
) {
    let mut counts = BTreeMap::<[u32; 2], usize>::new();
    for element in elements {
        for edge in [
            sorted_node_pair(element.node_ids[0], element.node_ids[1]),
            sorted_node_pair(element.node_ids[1], element.node_ids[2]),
            sorted_node_pair(element.node_ids[2], element.node_ids[0]),
        ] {
            *counts.entry(edge).or_default() += 1;
        }
    }
    for (edge, count) in counts {
        let is_boundary = boundary_edges.contains(&edge);
        assert_eq!(
            count,
            if is_boundary { 1 } else { 2 },
            "unexpected local surface edge count for {edge:?}"
        );
    }
}
