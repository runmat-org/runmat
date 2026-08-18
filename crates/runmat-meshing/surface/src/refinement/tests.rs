use runmat_geometry_core::{
    GeometryEvaluationControl, GeometryEvaluationError, GeometryModel, PersistentEntityId,
    PersistentEntityKind, PortableExactEvaluator,
};
use runmat_meshing_core::{
    MetricCombinationRule, MetricFieldRequest, MetricSourceKind, MetricTensor3, StableDigest,
    SurfaceQualityTargets,
};
use runmat_meshing_curve::{
    apply_shared_curve_splits, discretize_shared_curves, shared_curve_interior_node_id,
    CurveResolutionPolicy, SharedCurveDiscretizationOptions, SharedCurveEvaluationContext,
    UniformCurveMetric,
};

use crate::{
    ExactFaceDelaunayTriangle, ExactFaceGeometry, ExactFaceGeometryVertex,
    ExactFaceMetricEvaluation, ExactFacePslg, ExactFacePslgSegment, ExactFacePslgVertex,
    ExactFaceTriangleGeometry, ParametricMetricTensor,
};

use super::*;

#[test]
fn canonical_bad_triangle_selects_its_metric_circumcenter() {
    let geometry = geometry([2.0, 1.0, 1.0], 0.5, 1.0, 0.0, 0.0);
    let candidate = select_exact_face_refinement_candidate(&geometry, quality())
        .unwrap()
        .unwrap();

    assert_eq!(
        candidate.reason,
        ExactFaceRefinementReason::MetricEdgeLength
    );
    assert_eq!(candidate.triangle_index, 0);
    assert_eq!(candidate.uv, [1.0, 1.0]);

    let mut invalid = quality();
    invalid.minimum_metric_angle_degrees = 60.0;
    assert_eq!(
        select_exact_face_refinement_candidate(&geometry, invalid)
            .unwrap_err()
            .kind,
        ExactFaceRefinementErrorKind::InvalidQuality
    );
}

#[test]
fn geometric_deviation_uses_the_exact_triangle_centroid() {
    let geometry = geometry([0.5; 3], 0.5, 1.0, 0.25, 0.0);
    let candidate = select_exact_face_refinement_candidate(&geometry, quality())
        .unwrap()
        .unwrap();

    assert_eq!(
        candidate.reason,
        ExactFaceRefinementReason::ChordalDeviation
    );
    assert_eq!(candidate.uv, geometry.triangles[0].centroid.uv);
}

#[test]
fn encroaching_candidate_requests_one_exact_parameter_split_before_insertion() {
    let (document, topology, registry) = runmat_geometry_fixtures::exact_circle();
    let GeometryModel::ExactBRep { model } = &document.model else {
        panic!("fixture must be exact")
    };
    let evaluator = PortableExactEvaluator::new(&registry, &topology, model).unwrap();
    let face_id = topology.faces[0].id.clone();
    let pslg = one_segment_pslg(face_id.clone());
    let encroaching = candidate(face_id.clone(), [0.0, 0.0]);

    let disposition = classify_exact_face_refinement_candidate(
        &encroaching,
        &pslg,
        &topology,
        &metric_request(),
        &evaluator,
        &Control,
    )
    .unwrap();
    let ExactFaceCandidateDisposition::SplitProtectedSegment(split) = disposition else {
        panic!("diametral candidate must split the protected segment")
    };
    assert_eq!(split.source_face_id, face_id);
    assert_eq!(split.pslg_segment_index, 0);
    assert_eq!(split.curve_split.endpoint_node_ids, [node(2), node(1)]);
    assert_eq!(split.curve_split.edge_parameters, [2.0, 6.0]);
    assert_eq!(split.curve_split.split_parameter, 4.0);

    let mut forward = pslg.clone();
    forward.segments[0].vertex_indices = [1, 0];
    forward.segments[0].edge_parameters = [2.0, 6.0];
    let ExactFaceCandidateDisposition::SplitProtectedSegment(forward_split) =
        classify_exact_face_refinement_candidate(
            &encroaching,
            &forward,
            &topology,
            &metric_request(),
            &evaluator,
            &Control,
        )
        .unwrap()
    else {
        panic!("opposite coedge use must request the same protected split")
    };
    assert_eq!(forward_split.curve_split, split.curve_split);

    let outside = candidate(topology.faces[0].id.clone(), [0.0, 2.0]);
    assert_eq!(
        classify_exact_face_refinement_candidate(
            &outside,
            &pslg,
            &topology,
            &metric_request(),
            &evaluator,
            &Control,
        )
        .unwrap(),
        ExactFaceCandidateDisposition::Insert
    );
}

#[test]
fn protected_split_rebuilds_the_global_curve_and_face_pslg() {
    let (document, topology, registry) = runmat_geometry_fixtures::exact_circle();
    let GeometryModel::ExactBRep { model } = &document.model else {
        panic!("fixture must be exact")
    };
    let evaluator = PortableExactEvaluator::new(&registry, &topology, model).unwrap();
    let metric = UniformCurveMetric::from_target_size_m(1.0).unwrap();
    let options = curve_options();
    let curves = discretize_shared_curves(
        &topology, &evaluator, &evaluator, &metric, &Control, options,
    )
    .unwrap();
    let boundary = crate::build_exact_surface_boundary(&topology, &curves).unwrap();
    let pslg = crate::build_exact_face_pslg(&boundary.faces[0]).unwrap();
    let segment = &pslg.segments[0];
    let endpoints = segment
        .vertex_indices
        .map(|index| pslg.vertices[index as usize].uv);
    let encroaching = candidate(
        pslg.source_face_id.clone(),
        [
            (endpoints[0][0] + endpoints[1][0]) * 0.5,
            (endpoints[0][1] + endpoints[1][1]) * 0.5,
        ],
    );
    let disposition = classify_exact_face_refinement_candidate(
        &encroaching,
        &pslg,
        &topology,
        &metric_request(),
        &evaluator,
        &Control,
    )
    .unwrap();
    let ExactFaceCandidateDisposition::SplitProtectedSegment(split) = disposition else {
        panic!("segment midpoint must be classified as encroaching")
    };
    let inserted_id = shared_curve_interior_node_id(
        &split.curve_split.source_edge_id,
        split.curve_split.split_parameter,
    );
    let context =
        SharedCurveEvaluationContext::new(&topology, &evaluator, &evaluator, &metric, &Control);
    let refined = apply_shared_curve_splits(
        &curves,
        context,
        options,
        std::slice::from_ref(&split.curve_split),
    )
    .unwrap();
    let rebuilt_boundary = crate::build_exact_surface_boundary(&topology, &refined).unwrap();
    let rebuilt = crate::build_exact_face_pslg(&rebuilt_boundary.faces[0]).unwrap();

    assert_eq!(
        refined.edges[0].nodes.len(),
        curves.edges[0].nodes.len() + 1
    );
    assert_eq!(rebuilt.segments.len(), pslg.segments.len() + 1);
    assert!(refined.edges[0]
        .nodes
        .iter()
        .any(|node| node.node_id == inserted_id));
    assert_eq!(
        rebuilt
            .segments
            .iter()
            .filter(|segment| {
                segment
                    .vertex_indices
                    .into_iter()
                    .any(|index| rebuilt.vertices[index as usize].node_id == inserted_id)
            })
            .count(),
        2
    );
}

fn geometry(
    metric_edges: [f64; 3],
    minimum_angle: f64,
    aspect: f64,
    chordal: f64,
    normal: f64,
) -> ExactFaceGeometry {
    let face_id = id(PersistentEntityKind::Face, "face");
    let uvs = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]];
    let vertices = uvs
        .into_iter()
        .enumerate()
        .map(|(index, uv)| ExactFaceGeometryVertex {
            pslg_vertex_index: index as u32,
            evaluation: evaluation(&face_id, uv),
            unit_normal: [0.0, 0.0, 1.0],
        })
        .collect();
    let triangle = ExactFaceTriangleGeometry {
        triangle: ExactFaceDelaunayTriangle {
            vertex_indices: [0, 1, 2],
        },
        centroid: evaluation(&face_id, [2.0 / 3.0, 2.0 / 3.0]),
        unit_normal: [0.0, 0.0, 1.0],
        physical_area_m2: 2.0,
        metric_edge_lengths: metric_edges,
        minimum_metric_angle_rad: minimum_angle,
        physical_aspect_ratio: aspect,
        chordal_deviation_m: chordal,
        normal_deviation_rad: normal,
    };
    ExactFaceGeometry {
        source_face_id: face_id,
        vertices,
        triangles: vec![triangle],
        maximum_metric_edge_length: metric_edges.into_iter().fold(0.0, f64::max),
        minimum_metric_angle_rad: minimum_angle,
        maximum_physical_aspect_ratio: aspect,
        maximum_chordal_deviation_m: chordal,
        maximum_normal_deviation_rad: normal,
    }
}

fn evaluation(face_id: &PersistentEntityId, uv: [f64; 2]) -> ExactFaceMetricEvaluation {
    ExactFaceMetricEvaluation {
        source_face_id: face_id.clone(),
        uv,
        point_m: [uv[0], uv[1], 0.0],
        derivative_u_m: [1.0, 0.0, 0.0],
        derivative_v_m: [0.0, 1.0, 0.0],
        physical_metric: identity_metric(),
        sizing_metric: identity_metric(),
        active_sources: vec![MetricSourceKind::Global],
        applied_contribution_count: 0,
        clipped_contribution_count: 0,
        rejected_contribution_count: 0,
    }
}

fn one_segment_pslg(source_face_id: PersistentEntityId) -> ExactFacePslg {
    ExactFacePslg {
        source_face_id,
        vertices: vec![
            ExactFacePslgVertex {
                node_id: node(1),
                seam_image: None,
                uv: [-1.0, 0.0],
            },
            ExactFacePslgVertex {
                node_id: node(2),
                seam_image: None,
                uv: [1.0, 0.0],
            },
        ],
        segments: vec![ExactFacePslgSegment {
            source_coedge_id: id(PersistentEntityKind::Coedge, "coedge"),
            source_edge_id: id(PersistentEntityKind::Edge, "edge"),
            vertex_indices: [0, 1],
            edge_parameters: [6.0, 2.0],
        }],
        loops: Vec::new(),
    }
}

fn candidate(source_face_id: PersistentEntityId, uv: [f64; 2]) -> ExactFaceRefinementCandidate {
    ExactFaceRefinementCandidate {
        source_face_id,
        triangle_index: 0,
        triangle: ExactFaceDelaunayTriangle {
            vertex_indices: [0, 1, 2],
        },
        reason: ExactFaceRefinementReason::MetricEdgeLength,
        uv,
    }
}

fn quality() -> SurfaceQualityTargets {
    SurfaceQualityTargets {
        minimum_metric_angle_degrees: 10.0,
        maximum_physical_aspect_ratio: 10.0,
        maximum_chordal_deviation_m: 0.1,
        maximum_normal_deviation_degrees: 30.0,
    }
}

fn metric_request() -> MetricFieldRequest {
    MetricFieldRequest {
        combination: MetricCombinationRule::MostRestrictiveIntersection,
        global_metric: MetricTensor3::isotropic_length_m(1.0).unwrap(),
        maximum_grading_ratio: 2.0,
        contributions: Vec::new(),
    }
}

fn curve_options() -> SharedCurveDiscretizationOptions {
    SharedCurveDiscretizationOptions {
        resolution: CurveResolutionPolicy {
            maximum_chordal_deviation_m: 0.01,
            maximum_tangent_change_rad: 0.2,
            minimum_metric_edge_length: 0.01,
            maximum_metric_edge_length: 1.0,
        },
        maximum_nodes_per_edge: 1_024,
        maximum_subdivision_depth: 20,
        geometry_absolute_error_m: 1.0e-10,
        pcurve_absolute_error: 1.0e-10,
        arc_length_absolute_error_m: 1.0e-10,
    }
}

fn identity_metric() -> ParametricMetricTensor {
    ParametricMetricTensor {
        uu: 1.0,
        uv: 0.0,
        vv: 1.0,
    }
}

fn node(value: u8) -> StableDigest {
    StableDigest::from_bytes([value; 32])
}

fn id(kind: PersistentEntityKind, name: &str) -> PersistentEntityId {
    PersistentEntityId {
        kind,
        source_topology_id: name.into(),
        assembly_path: vec!["part".into()],
    }
}

struct Control;

impl GeometryEvaluationControl for Control {
    fn checkpoint(&self) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }

    fn consume_iterations(&self, _count: u64) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }

    fn consume_search_work(&self, _count: u64) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }

    fn consume_allocation_bytes(&self, _count: u64) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }
}
