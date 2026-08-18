use runmat_geometry_core::{
    GeometryEvaluationControl, GeometryEvaluationError, GeometryModel, PersistentEntityId,
    PersistentEntityKind, PortableExactEvaluator,
};
use runmat_meshing_core::{
    MetricCombinationRule, MetricFieldRequest, MetricSourceKind, MetricTensor3, StableDigest,
    SurfaceQualityTargets,
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
    assert_eq!(split.endpoint_node_ids, [node(2), node(1)]);
    assert_eq!(split.edge_parameters, [2.0, 6.0]);
    assert_eq!(split.split_parameter, 4.0);

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
