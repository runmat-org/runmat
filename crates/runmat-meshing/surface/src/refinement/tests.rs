mod cooperative_classification;
mod cooperative_insertion;
mod cooperative_selection;
mod local_queries;
mod protected_edge_inventory;
mod separated_size_candidate;
mod sized_curve_boundary;

use runmat_geometry_core::{
    GeometryEvaluationControl, GeometryEvaluationError, GeometryModel, PersistentEntityId,
    PersistentEntityKind, PortableExactEvaluator, TopologicalOrientation,
};
use runmat_meshing_core::{
    MeshingCancellationSignal, MetricCombinationRule, MetricFieldRequest, MetricSourceKind,
    MetricTensor3, NeverCancelled, StableDigest, SurfaceQualityTargets,
};
use runmat_meshing_curve::{
    apply_shared_curve_splits, discretize_shared_curves, shared_curve_interior_node_id,
    CurveResolutionPolicy, SharedCurveDiscretizationOptions, SharedCurveEvaluationContext,
    UniformCurveMetric,
};

use crate::{
    exact_face_chart_cut_node_id, ExactFaceChartParameterization, ExactFaceDelaunayTriangle,
    ExactFaceGeometry, ExactFaceGeometryVertex, ExactFaceMetricEvaluation, ExactFacePslg,
    ExactFacePslgLoop, ExactFacePslgLoopSource, ExactFacePslgSegment, ExactFacePslgSegmentSource,
    ExactFacePslgVertex, ExactFaceTriangleGeometry, ParametricMetricTensor,
};

use super::*;

#[test]
fn diametral_disk_uses_one_conservative_policy_at_its_floating_boundary() {
    let metric = ParametricMetricTensor {
        uu: 1.0,
        uv: 0.0,
        vv: 1.0,
    };
    let first = [-1.0, 0.0];
    let second = [1.0, 0.0];

    assert_eq!(
        super::encroachment::metric_diametral_disk_contains(
            [0.0, 1.0 + f64::EPSILON],
            first,
            second,
            metric,
        ),
        Some(true),
    );
    assert_eq!(
        super::encroachment::metric_diametral_disk_contains(
            [0.0, 1.0 + 1.0e-8],
            first,
            second,
            metric,
        ),
        Some(false),
    );
}

#[test]
fn canonical_bad_triangle_selects_a_reason_specific_construction() {
    let geometry = geometry([2.0, 1.0, 1.0], 0.5, 1.0, 0.0, 0.0);
    let pslg = triangle_pslg(&geometry);
    let collars = empty_collars(&geometry);
    let candidate = select_exact_face_refinement_candidate(&geometry, &pslg, &collars, quality())
        .unwrap()
        .unwrap();

    assert_eq!(
        candidate.reason,
        ExactFaceRefinementReason::MetricEdgeLength
    );
    assert_eq!(candidate.triangle_index, 0);
    assert_eq!(candidate.uv, [1.0, 0.0]);

    let mut duplicate_geometry = geometry.clone();
    let mut duplicate_pslg = pslg.clone();
    duplicate_pslg.vertices.push(ExactFacePslgVertex {
        node_id: node(9),
        seam_image: None,
        uv: candidate.uv,
    });
    duplicate_geometry.vertices.push(ExactFaceGeometryVertex {
        pslg_vertex_index: 3,
        point_m: evaluation(&geometry.source_face_id, candidate.uv).point_m,
        evaluation: evaluation(&geometry.source_face_id, candidate.uv),
        unit_normal: [0.0, 0.0, 1.0],
    });
    let duplicate_collars = empty_collars(&duplicate_geometry);
    let fallback = select_exact_face_refinement_candidate(
        &duplicate_geometry,
        &duplicate_pslg,
        &duplicate_collars,
        quality(),
    )
    .unwrap()
    .unwrap();
    assert_eq!(fallback.uv, geometry.triangles[0].deviation_samples[3].uv);

    let mut obtuse_geometry = self::geometry([2.0, 1.0, 1.0], 0.5, 1.0, 0.0, 0.0);
    obtuse_geometry.vertices[2].evaluation =
        evaluation(&obtuse_geometry.source_face_id, [0.2, 0.1]);
    obtuse_geometry.triangles[0].deviation_samples[3] = evaluation(
        &obtuse_geometry.source_face_id,
        [(0.0 + 2.0 + 0.2) / 3.0, 0.1 / 3.0],
    );
    let obtuse_pslg = triangle_pslg(&obtuse_geometry);
    let obtuse_collars = empty_collars(&obtuse_geometry);
    let obtuse_candidate = select_exact_face_refinement_candidate(
        &obtuse_geometry,
        &obtuse_pslg,
        &obtuse_collars,
        quality(),
    )
    .unwrap()
    .unwrap();
    assert_eq!(obtuse_candidate.uv, [1.0, 0.0]);

    let mut invalid = quality();
    invalid.minimum_metric_angle_degrees = 60.0;
    assert_eq!(
        select_exact_face_refinement_candidate(&geometry, &pslg, &collars, invalid)
            .unwrap_err()
            .kind,
        ExactFaceRefinementErrorKind::InvalidQuality
    );
}

#[test]
fn refinement_batch_never_inserts_adjacent_stale_cavities() {
    let face = id(PersistentEntityKind::Face, "batch-face");
    let mut adjacent = candidate(face.clone(), [20.0, 20.0]);
    adjacent.triangle_index = 1;
    adjacent.triangle.vertex_indices = [1, 2, 3];
    let mut independent = candidate(face.clone(), [30.0, 30.0]);
    independent.triangle_index = 2;
    independent.triangle.vertex_indices = [4, 5, 6];
    let mut geometry = geometry([2.0, 1.0, 1.0], 0.5, 1.0, 0.0, 0.0);
    let mut adjacent_geometry = geometry.triangles[0].clone();
    adjacent_geometry.triangle = adjacent.triangle;
    let mut independent_geometry = geometry.triangles[0].clone();
    independent_geometry.triangle = independent.triangle;
    geometry
        .triangles
        .extend([adjacent_geometry, independent_geometry]);
    let pslg = triangle_pslg(&geometry);

    let selected = super::select::independent_candidate_batch(
        vec![candidate(face, [10.0, 10.0]), adjacent, independent.clone()],
        &geometry,
        &pslg,
        0.01,
        None,
    )
    .unwrap();

    assert_eq!(selected.len(), 2);
    assert_eq!(selected[1], independent);
}

#[test]
fn shape_candidate_batch_respects_the_two_child_spacing_collar() {
    let face = id(PersistentEntityKind::Face, "batch-spacing-face");
    let mut geometry = geometry([0.5; 3], 0.5, 1.0, 0.0, 0.0);
    geometry.vertices[0].evaluation = evaluation(&geometry.source_face_id, [0.0, 0.0]);
    let pslg = triangle_pslg(&geometry);
    let mut near = candidate(face, [0.015, 0.0]);
    near.reason = ExactFaceRefinementReason::MetricAngle;

    let selected =
        super::select::independent_candidate_batch(vec![near], &geometry, &pslg, 0.02, None)
            .unwrap();

    assert!(selected.is_empty());
}

#[test]
fn near_node_sizing_candidates_are_relocated_or_dropped_not_fatal() {
    let face = id(PersistentEntityKind::Face, "near-node-face");
    let geometry = geometry([2.0, 2.0, 2.0], 0.5, 1.0, 0.0, 0.0);
    let pslg = triangle_pslg(&geometry);

    // A metric-edge-length candidate beside a corner moves to a separated point in its
    // triangle, measured in the sizing at the new point.
    let near = candidate(face.clone(), [0.01, 0.0]);
    let selected = super::select::independent_candidate_batch(
        vec![near.clone()],
        &geometry,
        &pslg,
        0.02,
        None,
    )
    .unwrap();
    assert_eq!(selected.len(), 1);
    assert_ne!(selected[0].uv, near.uv);
    for vertex in &pslg.vertices {
        let distance = (selected[0].uv[0] - vertex.uv[0]).hypot(selected[0].uv[1] - vertex.uv[1]);
        assert!(distance >= 0.02, "relocated candidate stays separated");
    }
    let repeated = super::select::independent_candidate_batch(
        vec![near.clone()],
        &geometry,
        &pslg,
        0.02,
        None,
    )
    .unwrap();
    assert_eq!(repeated, selected, "relocation is deterministic");

    // With no separated point in the triangle, a boundary-support candidate is dropped and the
    // batch is empty, so refinement terminates instead of retrying the same target.
    let mut support = candidate(face.clone(), [0.01, 0.0]);
    support.reason = ExactFaceRefinementReason::BoundaryMetricSupport;
    assert!(
        super::select::independent_candidate_batch(vec![support], &geometry, &pslg, 2.0, None)
            .unwrap()
            .is_empty()
    );

    // Geometric fidelity targets still have no substitute point.
    let mut chordal = candidate(face, [0.01, 0.0]);
    chordal.reason = ExactFaceRefinementReason::ChordalDeviation;
    assert!(super::select::independent_candidate_batch(
        vec![chordal],
        &geometry,
        &pslg,
        0.02,
        None
    )
    .is_err());
}

#[test]
fn shape_refinement_respects_the_surface_minimum_edge_collar() {
    let geometry = geometry([0.005, 0.5, 0.5], 0.001, 1_000.0, 0.0, 0.0);
    let mut pslg = triangle_pslg(&geometry);
    pslg.segments = [[0, 1], [1, 2], [2, 0]]
        .into_iter()
        .enumerate()
        .map(|(index, vertex_indices)| ExactFacePslgSegment {
            source: ExactFacePslgSegmentSource::ExactTrim {
                source_coedge_id: id(
                    PersistentEntityKind::Coedge,
                    &format!("collar-coedge-{index}"),
                ),
                source_edge_id: id(PersistentEntityKind::Edge, &format!("collar-edge-{index}")),
            },
            vertex_indices,
            edge_parameters: Some([0.0, 1.0]),
        })
        .collect();
    let collars = empty_collars(&geometry);

    assert!(
        select_exact_face_refinement_candidate(&geometry, &pslg, &collars, quality())
            .unwrap()
            .is_none()
    );

    let mut unprotected = pslg;
    unprotected.segments.clear();
    unprotected.loops.clear();
    assert_eq!(
        select_exact_face_refinement_candidate(
            &geometry,
            &unprotected,
            &empty_collars(&geometry),
            quality(),
        )
        .unwrap_err()
        .kind,
        ExactFaceRefinementErrorKind::UnsatisfiedQuality
    );
}

#[test]
fn shape_candidate_does_not_refine_a_sizing_compliant_protected_edge() {
    let face = id(PersistentEntityKind::Face, "split-floor-face");
    let mut pslg = ExactFacePslg {
        source_face_id: face,
        vertices: vec![
            ExactFacePslgVertex {
                node_id: node(1),
                seam_image: None,
                uv: [0.4925, -0.005],
            },
            ExactFacePslgVertex {
                node_id: node(2),
                seam_image: None,
                uv: [0.5075, -0.005],
            },
        ],
        segments: Vec::new(),
        loops: Vec::new(),
    };
    pslg.segments.push(ExactFacePslgSegment {
        source: ExactFacePslgSegmentSource::ExactTrim {
            source_coedge_id: id(PersistentEntityKind::Coedge, "split-floor-coedge"),
            source_edge_id: id(PersistentEntityKind::Edge, "split-floor-edge"),
        },
        vertex_indices: [0, 1],
        edge_parameters: Some([0.0, 1.0]),
    });

    assert!(
        super::select::shape_candidate_encroaches_sized_protected_segment(
            [0.5, 0.0],
            &pslg,
            identity_metric(),
        )
    );
    assert!(
        !super::select::shape_candidate_encroaches_sized_protected_segment(
            [0.5, 0.02],
            &pslg,
            identity_metric(),
        )
    );
}

#[test]
fn acute_material_corner_has_canonical_independently_validated_collar() {
    let (pslg, geometry) = acute_corner_geometry();
    let mut targets = quality();
    targets.minimum_metric_angle_degrees = 20.0;
    let collars = derive_exact_face_feature_collars(&pslg, &geometry, targets).unwrap();

    assert_eq!(collars.collars.len(), 1);
    assert_eq!(collars.collars[0].pslg_vertex_index, 0);
    assert_eq!(collars.collars[0].incident_segment_indices, [0, 2]);
    assert!((collars.collars[0].feature_angle_rad.to_degrees() - 10.0).abs() < 1.0e-12);
    validate_exact_face_feature_collars(&collars, &pslg, &geometry, targets).unwrap();
    assert!(
        select_exact_face_refinement_candidate(&geometry, &pslg, &collars, targets)
            .unwrap()
            .is_none()
    );
    let mut long_edge = geometry.clone();
    long_edge.triangles[0].metric_edge_lengths[0] = 2.0;
    assert_eq!(
        select_exact_face_refinement_candidate(&long_edge, &pslg, &collars, targets)
            .unwrap()
            .unwrap()
            .reason,
        ExactFaceRefinementReason::MetricEdgeLength
    );

    let mut tampered = collars;
    tampered.collars[0].feature_angle_rad += 0.01;
    assert_eq!(
        validate_exact_face_feature_collars(&tampered, &pslg, &geometry, targets)
            .unwrap_err()
            .kind,
        ExactFaceRefinementErrorKind::InvalidGeometry
    );

    targets.minimum_metric_angle_degrees = 5.0;
    assert!(derive_exact_face_feature_collars(&pslg, &geometry, targets)
        .unwrap()
        .collars
        .is_empty());
}

#[test]
fn geometric_deviation_uses_the_exact_sample_witness() {
    let geometry = geometry([0.5; 3], 0.5, 1.0, 0.25, 0.0);
    let pslg = triangle_pslg(&geometry);
    let collars = empty_collars(&geometry);
    let candidate = select_exact_face_refinement_candidate(&geometry, &pslg, &collars, quality())
        .unwrap()
        .unwrap();

    assert_eq!(
        candidate.reason,
        ExactFaceRefinementReason::ChordalDeviation
    );
    assert_eq!(candidate.uv, geometry.triangles[0].chordal_deviation_uv);
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
    forward.segments[0].edge_parameters = Some([2.0, 6.0]);
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
fn shape_candidate_never_rewrites_a_sizing_validated_shared_curve() {
    let (document, topology, registry) = runmat_geometry_fixtures::exact_circle();
    let GeometryModel::ExactBRep { model } = &document.model else {
        panic!("fixture must be exact")
    };
    let evaluator = PortableExactEvaluator::new(&registry, &topology, model).unwrap();
    let face_id = topology.faces[0].id.clone();
    let pslg = one_segment_pslg(face_id.clone());
    let classify = |uv: [f64; 2]| {
        let mut shape = candidate(face_id.clone(), uv);
        shape.reason = ExactFaceRefinementReason::PhysicalAspectRatio;
        classify_exact_face_refinement_candidate(
            &shape,
            &pslg,
            &topology,
            &metric_request(),
            &evaluator,
            &Control,
        )
        .unwrap()
    };
    // A shape point never splits the curve: inside its diametral disk it is inserted. Exactly on
    // the curve it can be neither inserted nor used to split it, so it is withdrawn.
    assert_eq!(classify([0.0, 1.5]), ExactFaceCandidateDisposition::Insert);
    assert_eq!(classify([0.0, 0.1]), ExactFaceCandidateDisposition::Insert);
    assert_eq!(
        classify([0.0, 0.0]),
        ExactFaceCandidateDisposition::Withdraw
    );
}

#[test]
fn candidate_batch_requests_every_independently_encroached_segment_once() {
    let (document, topology, registry) = runmat_geometry_fixtures::exact_circle();
    let GeometryModel::ExactBRep { model } = &document.model else {
        panic!("fixture must be exact")
    };
    let evaluator = PortableExactEvaluator::new(&registry, &topology, model).unwrap();
    let face_id = topology.faces[0].id.clone();
    let mut pslg = one_segment_pslg(face_id.clone());
    pslg.vertices.extend([
        ExactFacePslgVertex {
            node_id: node(3),
            seam_image: None,
            uv: [-1.0, 2.0],
        },
        ExactFacePslgVertex {
            node_id: node(4),
            seam_image: None,
            uv: [1.0, 2.0],
        },
    ]);
    pslg.segments.push(ExactFacePslgSegment {
        source: ExactFacePslgSegmentSource::ExactTrim {
            source_coedge_id: id(PersistentEntityKind::Coedge, "second-coedge"),
            source_edge_id: id(PersistentEntityKind::Edge, "second-edge"),
        },
        vertex_indices: [2, 3],
        edge_parameters: Some([0.0, 1.0]),
    });
    let candidates = [
        candidate(face_id.clone(), [0.0, 0.0]),
        candidate(face_id, [0.0, 2.0]),
    ];

    let classifications = classify_exact_face_refinement_candidates_in_parameterization(
        &candidates,
        &pslg,
        &topology,
        &metric_request(),
        &ExactFaceChartParameterization::EvaluatorParameters,
        &evaluator,
        &Control,
    )
    .unwrap();

    assert_eq!(classifications.protected_splits.len(), 2);
    assert!(classifications.chart_cut_splits.is_empty());
    assert!(classifications.insertions.is_empty());
    assert_eq!(
        classifications
            .protected_splits
            .iter()
            .map(|split| split.pslg_segment_index)
            .collect::<Vec<_>>(),
        vec![0, 1]
    );
}

#[test]
fn encroaching_chart_cut_requests_both_periodic_images() {
    let (document, topology, registry) = runmat_geometry_fixtures::exact_circle();
    let GeometryModel::ExactBRep { model } = &document.model else {
        panic!("fixture must be exact")
    };
    let evaluator = PortableExactEvaluator::new(&registry, &topology, model).unwrap();
    let face_id = topology.faces[0].id.clone();
    let pslg = paired_cut_pslg(face_id.clone());
    let disposition = classify_exact_face_refinement_candidate(
        &candidate(face_id.clone(), [0.0, 0.0]),
        &pslg,
        &topology,
        &metric_request(),
        &evaluator,
        &Control,
    )
    .unwrap();
    let ExactFaceCandidateDisposition::SplitChartCut(split) = disposition else {
        panic!("encroaching chart cut must split both periodic images")
    };
    assert_eq!(split.source_face_id, face_id);
    assert_eq!(split.cut_id, node(9));
    assert_eq!(split.images.map(|image| image.pslg_segment_index), [0, 1]);
    assert_eq!(split.images[0].midpoint_uv, [0.0, 0.0]);
    assert_eq!(split.images[1].midpoint_uv, [0.0, 1.0]);
    assert_eq!(
        split.node_id,
        exact_face_chart_cut_node_id(node(9), [node(1), node(2)])
    );

    let mut unpaired = pslg;
    unpaired.segments.pop();
    assert_eq!(
        classify_exact_face_refinement_candidate(
            &candidate(topology.faces[0].id.clone(), [0.0, 0.0]),
            &unpaired,
            &topology,
            &metric_request(),
            &evaluator,
            &Control,
        )
        .unwrap_err()
        .kind,
        ExactFaceRefinementErrorKind::InvalidGeometry
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

#[test]
fn nonencroaching_candidate_is_inserted_into_validated_trimmed_topology() {
    let (document, topology, registry) = runmat_geometry_fixtures::exact_circle();
    let GeometryModel::ExactBRep { model } = &document.model else {
        panic!("fixture must be exact")
    };
    let evaluator = PortableExactEvaluator::new(&registry, &topology, model).unwrap();
    let metric = UniformCurveMetric::from_target_size_m(1.0).unwrap();
    let curves = discretize_shared_curves(
        &topology,
        &evaluator,
        &evaluator,
        &metric,
        &Control,
        curve_options(),
    )
    .unwrap();
    let boundary = crate::build_exact_surface_boundary(&topology, &curves).unwrap();
    let face_boundary = &boundary.faces[0];
    let pslg = crate::build_exact_face_pslg(face_boundary).unwrap();
    let options = crate::ExactFaceDelaunayOptions::default();
    let delaunay =
        crate::triangulate_exact_face_pslg(&pslg, face_boundary, &NeverCancelled, options).unwrap();
    let constrained = crate::recover_exact_face_segments(
        &delaunay,
        &pslg,
        face_boundary,
        &NeverCancelled,
        options,
    )
    .unwrap();
    let trimmed = crate::carve_exact_face_domain(
        &constrained,
        &pslg,
        face_boundary,
        &NeverCancelled,
        options,
    )
    .unwrap();
    let initial = ExactFaceRefinedTopology {
        pslg: pslg.clone(),
        constrained: constrained.clone(),
        trimmed: trimmed.clone(),
    };
    let triangle = trimmed.triangles[0];
    let corners = triangle
        .vertex_indices
        .map(|index| pslg.vertices[index as usize].uv);
    let uv = [
        corners.iter().map(|point| point[0]).sum::<f64>() / 3.0,
        corners.iter().map(|point| point[1]).sum::<f64>() / 3.0,
    ];
    let candidate = ExactFaceRefinementCandidate {
        source_face_id: pslg.source_face_id.clone(),
        triangle_index: 0,
        triangle,
        reason: ExactFaceRefinementReason::PhysicalAspectRatio,
        uv,
    };

    // Cancellation must reach each admission walk before triangulation starts.
    for cancel_at in [
        1,
        pslg.vertices.len() + 1,
        pslg.vertices.len() + pslg.segments.len() + 1,
    ] {
        let signal = cooperative_insertion::CancelDuringCandidateWalk::new(cancel_at);
        let failure = insert_exact_face_refinement_candidate(
            face_boundary,
            &initial,
            &candidate,
            &signal,
            options,
        )
        .unwrap_err();
        assert_eq!(
            failure.kind,
            ExactFaceRefinementErrorKind::Delaunay(crate::ExactFaceDelaunayErrorKind::Cancelled)
        );
        assert_eq!(signal.observed(), cancel_at);
    }

    let refined = insert_exact_face_refinement_candidate(
        face_boundary,
        &initial,
        &candidate,
        &NeverCancelled,
        options,
    )
    .unwrap();
    let repeated = insert_exact_face_refinement_candidate(
        face_boundary,
        &initial,
        &candidate,
        &NeverCancelled,
        options,
    )
    .unwrap();

    assert_eq!(refined, repeated);
    assert_eq!(refined.pslg.vertices.len(), pslg.vertices.len() + 1);
    assert_eq!(
        refined
            .pslg
            .vertices
            .iter()
            .filter(|vertex| {
                vertex.node_id == crate::exact_face_interior_node_id(&pslg.source_face_id, uv)
            })
            .count(),
        1
    );

    let second_triangle = trimmed.triangles[1];
    let second_corners = second_triangle
        .vertex_indices
        .map(|index| pslg.vertices[index as usize].uv);
    let second_candidate = ExactFaceRefinementCandidate {
        source_face_id: pslg.source_face_id.clone(),
        triangle_index: 1,
        triangle: second_triangle,
        reason: ExactFaceRefinementReason::PhysicalAspectRatio,
        uv: [
            second_corners.iter().map(|point| point[0]).sum::<f64>() / 3.0,
            second_corners.iter().map(|point| point[1]).sum::<f64>() / 3.0,
        ],
    };
    let (batched, inserted) = insert_exact_face_refinement_candidates(
        face_boundary,
        &initial,
        &[
            candidate.clone(),
            second_candidate.clone(),
            candidate.clone(),
        ],
        &NeverCancelled,
        options,
    )
    .unwrap();
    let (reordered, reordered_inserted) = insert_exact_face_refinement_candidates(
        face_boundary,
        &initial,
        &[second_candidate, candidate.clone()],
        &NeverCancelled,
        options,
    )
    .unwrap();
    assert_eq!(inserted, 2);
    assert_eq!(reordered_inserted, 2);
    assert_eq!(batched, reordered);
    assert_eq!(batched.pslg.vertices.len(), pslg.vertices.len() + 2);

    assert_eq!(
        refined.trimmed.boundary_segments.len(),
        trimmed.boundary_segments.len()
    );
    let mut tampered = refined.pslg.clone();
    let interior = tampered
        .vertices
        .iter_mut()
        .find(|vertex| vertex.uv == uv)
        .unwrap();
    interior.node_id = node(99);
    assert!(crate::validate_exact_face_pslg(&tampered, face_boundary).is_err());

    let mut duplicate = candidate.clone();
    duplicate.uv = pslg.vertices[triangle.vertex_indices[0] as usize].uv;
    assert_eq!(
        insert_exact_face_refinement_candidate(
            face_boundary,
            &initial,
            &duplicate,
            &NeverCancelled,
            options,
        )
        .unwrap_err()
        .kind,
        ExactFaceRefinementErrorKind::InvalidGeometry
    );

    let mut limited = options;
    limited.maximum_triangles = trimmed.triangles.len();
    assert_eq!(
        insert_exact_face_refinement_candidate(
            face_boundary,
            &initial,
            &candidate,
            &NeverCancelled,
            limited,
        )
        .unwrap_err()
        .kind,
        ExactFaceRefinementErrorKind::Delaunay(crate::ExactFaceDelaunayErrorKind::ElementLimit)
    );
    assert_eq!(
        insert_exact_face_refinement_candidate(
            face_boundary,
            &initial,
            &candidate,
            &CancelledSignal,
            options,
        )
        .unwrap_err()
        .kind,
        ExactFaceRefinementErrorKind::Delaunay(crate::ExactFaceDelaunayErrorKind::Cancelled)
    );

    let coarse_request = MetricFieldRequest {
        entity_supports: Vec::new(),
        located_sources: Vec::new(),
        combination: MetricCombinationRule::MostRestrictiveIntersection,
        global_metric: MetricTensor3::isotropic_length_m(100.0).unwrap(),
        maximum_grading_ratio: 2.0,
        contributions: Vec::new(),
        cells_across_gap: None,
    };
    let permissive_quality = SurfaceQualityTargets {
        minimum_metric_edge_length: 0.001,
        minimum_metric_angle_degrees: 0.01,
        maximum_physical_aspect_ratio: 1.0e9,
        maximum_chordal_deviation_m: runmat_types::physical::Length::new(1.0e9),
        maximum_normal_deviation_degrees: 180.0,
    };
    let converged = refine_exact_face_until_blocked(
        face_boundary,
        &initial,
        ExactFaceRefinementContext::new(
            &topology,
            &coarse_request,
            &evaluator,
            &Control,
            &NeverCancelled,
        ),
        ExactFaceRefinementPolicy {
            quality: permissive_quality,
            acceptance: crate::ExactFaceAcceptanceOptions::default(),
            delaunay: options,
            refinement: ExactFaceRefinementOptions {
                maximum_interior_insertions: 1,
            },
        },
    )
    .unwrap();
    let ExactFaceRefinementOutcome::Converged(converged) = converged else {
        panic!("permissive exact-circle refinement must converge")
    };
    assert_eq!(converged.interior_insertion_count, 0);
    let chart_id = node(77);
    let chart = crate::ExactFaceChart {
        chart_id,
        source_face_id: pslg.source_face_id.clone(),
        periodicity: [None, None],
        parameterization: crate::ExactFaceChartParameterization::EvaluatorParameters,
        boundary: face_boundary.clone(),
        pslg: pslg.clone(),
    };
    let chart_domain = crate::ExactFaceChartConstrainedDomain {
        chart_id,
        delaunay: delaunay.clone(),
        constrained: constrained.clone(),
        trimmed: trimmed.clone(),
    };
    let chart_converged = refine_exact_face_chart_until_blocked(
        &chart,
        &chart_domain,
        ExactFaceRefinementContext::new(
            &topology,
            &coarse_request,
            &evaluator,
            &Control,
            &NeverCancelled,
        ),
        ExactFaceRefinementPolicy {
            quality: permissive_quality,
            acceptance: crate::ExactFaceAcceptanceOptions::default(),
            delaunay: options,
            refinement: ExactFaceRefinementOptions {
                maximum_interior_insertions: 1,
            },
        },
        ExactFaceChartRefinementOptions {
            maximum_chart_cut_splits: 1,
        },
    )
    .unwrap();
    let ExactFaceChartRefinementOutcome::Converged(chart_converged) = chart_converged else {
        panic!("permissive admitted chart refinement must converge")
    };
    assert_eq!(chart_converged.chart_id, chart_id);
    assert_eq!(chart_converged.mesh.interior_insertion_count, 0);
    assert_eq!(chart_converged.chart_cut_split_count, 0);
    let acceptance_options = crate::ExactFaceAcceptanceOptions {
        minimum_subdivision_depth: 1,
        maximum_subdivision_depth: 2,
        refinement_margin_ratio: 0.5,
        maximum_samples: 100_000,
    };
    let acceptance = crate::accept_exact_face_mesh(
        &converged,
        ExactFaceRefinementContext::new(
            &topology,
            &coarse_request,
            &evaluator,
            &Control,
            &NeverCancelled,
        ),
        permissive_quality,
        acceptance_options,
    )
    .unwrap();
    assert_eq!(
        acceptance.triangles.len(),
        converged.geometry.triangles.len()
    );
    assert!(acceptance.sample_count > acceptance.triangles.len() as u64);
    assert!(acceptance.maximum_chordal_deviation_m.canonical_value() < 1.0e-12);
    assert!(acceptance.maximum_normal_deviation_rad < 1.0e-7);
    let chart_acceptance = crate::accept_exact_face_chart_mesh(
        &chart,
        &chart_converged,
        ExactFaceRefinementContext::new(
            &topology,
            &coarse_request,
            &evaluator,
            &Control,
            &NeverCancelled,
        ),
        permissive_quality,
        acceptance_options,
    )
    .unwrap();
    assert_eq!(chart_acceptance.chart_id, chart_id);
    assert_eq!(chart_acceptance.acceptance, acceptance);
    let mut mismatched_chart_acceptance = chart_acceptance;
    mismatched_chart_acceptance.chart_id = node(79);
    assert_eq!(
        crate::validate_exact_face_chart_acceptance(
            &mismatched_chart_acceptance,
            &chart,
            &chart_converged,
            ExactFaceRefinementContext::new(
                &topology,
                &coarse_request,
                &evaluator,
                &Control,
                &NeverCancelled,
            ),
            permissive_quality,
            acceptance_options,
        )
        .unwrap_err()
        .kind,
        crate::ExactFaceAcceptanceErrorKind::InvalidInput
    );
    let mut tampered_acceptance = acceptance.clone();
    tampered_acceptance.maximum_chordal_deviation_m = runmat_types::physical::Length::new(1.0);
    assert_eq!(
        crate::validate_exact_face_acceptance(
            &tampered_acceptance,
            &converged,
            ExactFaceRefinementContext::new(
                &topology,
                &coarse_request,
                &evaluator,
                &Control,
                &NeverCancelled,
            ),
            permissive_quality,
            acceptance_options,
        )
        .unwrap_err()
        .kind,
        crate::ExactFaceAcceptanceErrorKind::InvalidInput
    );
    assert_eq!(
        crate::accept_exact_face_mesh(
            &converged,
            ExactFaceRefinementContext::new(
                &topology,
                &coarse_request,
                &evaluator,
                &Control,
                &NeverCancelled,
            ),
            permissive_quality,
            crate::ExactFaceAcceptanceOptions {
                maximum_samples: 1,
                ..acceptance_options
            },
        )
        .unwrap_err()
        .kind,
        crate::ExactFaceAcceptanceErrorKind::ResourceLimit
    );

    let strict_aspect = SurfaceQualityTargets {
        maximum_physical_aspect_ratio: 1.0,
        ..permissive_quality
    };
    assert_eq!(
        refine_exact_face_until_blocked(
            face_boundary,
            &initial,
            ExactFaceRefinementContext::new(
                &topology,
                &coarse_request,
                &evaluator,
                &Control,
                &NeverCancelled,
            ),
            ExactFaceRefinementPolicy {
                quality: strict_aspect,
                acceptance: crate::ExactFaceAcceptanceOptions::default(),
                delaunay: options,
                refinement: ExactFaceRefinementOptions {
                    maximum_interior_insertions: 1,
                },
            },
        )
        .unwrap_err()
        .kind,
        ExactFaceRefinementErrorKind::ResourceLimit
    );
    assert_eq!(
        refine_exact_face_chart_until_blocked(
            &chart,
            &chart_domain,
            ExactFaceRefinementContext::new(
                &topology,
                &coarse_request,
                &evaluator,
                &Control,
                &NeverCancelled,
            ),
            ExactFaceRefinementPolicy {
                quality: strict_aspect,
                acceptance: crate::ExactFaceAcceptanceOptions::default(),
                delaunay: options,
                refinement: ExactFaceRefinementOptions {
                    maximum_interior_insertions: 1,
                },
            },
            ExactFaceChartRefinementOptions {
                maximum_chart_cut_splits: 1,
            },
        )
        .unwrap_err()
        .kind,
        ExactFaceRefinementErrorKind::ResourceLimit
    );

    assert_eq!(
        refine_exact_face_until_blocked(
            face_boundary,
            &initial,
            ExactFaceRefinementContext::new(
                &topology,
                &coarse_request,
                &evaluator,
                &Control,
                &NeverCancelled,
            ),
            ExactFaceRefinementPolicy {
                quality: permissive_quality,
                acceptance: crate::ExactFaceAcceptanceOptions::default(),
                delaunay: options,
                refinement: ExactFaceRefinementOptions {
                    maximum_interior_insertions: 0,
                },
            },
        )
        .unwrap_err()
        .kind,
        ExactFaceRefinementErrorKind::InvalidOptions
    );
    assert_eq!(
        refine_exact_face_chart_until_blocked(
            &chart,
            &chart_domain,
            ExactFaceRefinementContext::new(
                &topology,
                &coarse_request,
                &evaluator,
                &Control,
                &NeverCancelled,
            ),
            ExactFaceRefinementPolicy {
                quality: permissive_quality,
                acceptance: crate::ExactFaceAcceptanceOptions::default(),
                delaunay: options,
                refinement: ExactFaceRefinementOptions {
                    maximum_interior_insertions: 1,
                },
            },
            ExactFaceChartRefinementOptions {
                maximum_chart_cut_splits: 0,
            },
        )
        .unwrap_err()
        .kind,
        ExactFaceRefinementErrorKind::InvalidOptions
    );
    let mut mismatched_chart_domain = chart_domain;
    mismatched_chart_domain.chart_id = node(78);
    assert_eq!(
        refine_exact_face_chart_until_blocked(
            &chart,
            &mismatched_chart_domain,
            ExactFaceRefinementContext::new(
                &topology,
                &coarse_request,
                &evaluator,
                &Control,
                &NeverCancelled,
            ),
            ExactFaceRefinementPolicy {
                quality: permissive_quality,
                acceptance: crate::ExactFaceAcceptanceOptions::default(),
                delaunay: options,
                refinement: ExactFaceRefinementOptions {
                    maximum_interior_insertions: 1,
                },
            },
            ExactFaceChartRefinementOptions {
                maximum_chart_cut_splits: 1,
            },
        )
        .unwrap_err()
        .kind,
        ExactFaceRefinementErrorKind::InvalidGeometry
    );
}

pub(super) fn geometry(
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
            point_m: evaluation(&face_id, uv).point_m,
            evaluation: evaluation(&face_id, uv),
            unit_normal: [0.0, 0.0, 1.0],
        })
        .collect();
    let triangle = ExactFaceTriangleGeometry {
        triangle: ExactFaceDelaunayTriangle {
            vertex_indices: [0, 1, 2],
        },
        deviation_samples: deviation_samples(&face_id, uvs),
        unit_normal: [0.0, 0.0, 1.0],
        physical_area_m2: 2.0,
        metric_edge_lengths: metric_edges,
        minimum_metric_angle_rad: minimum_angle,
        physical_aspect_ratio: aspect,
        chordal_deviation_m: chordal,
        chordal_deviation_uv: [2.0 / 3.0, 2.0 / 3.0],
        normal_deviation_rad: normal,
        normal_deviation_uv: [2.0 / 3.0, 2.0 / 3.0],
    };
    ExactFaceGeometry {
        source_face_id: face_id,
        vertices,
        triangles: vec![triangle],
        maximum_metric_edge_length: metric_edges.into_iter().fold(0.0, f64::max),
        minimum_metric_angle_rad: minimum_angle,
        maximum_physical_aspect_ratio: aspect,
        maximum_chordal_deviation_m: runmat_types::physical::Length::new(chordal),
        maximum_normal_deviation_rad: normal,
    }
}

fn evaluation(face_id: &PersistentEntityId, uv: [f64; 2]) -> ExactFaceMetricEvaluation {
    ExactFaceMetricEvaluation {
        provenance: std::sync::Arc::new(crate::ExactFaceMetricProvenance {
            source_face_id: face_id.clone(),
            active_sources: vec![MetricSourceKind::Global],
            applied_contribution_count: 0,
            clipped_contribution_count: 0,
            rejected_contribution_count: 0,
        }),
        uv,
        evaluator_uv: uv,
        point_m: [uv[0], uv[1], 0.0],
        derivative_u_m: [1.0, 0.0, 0.0],
        derivative_v_m: [0.0, 1.0, 0.0],
        physical_metric: identity_metric(),
        sizing_metric: identity_metric(),
    }
}

fn deviation_samples(
    face_id: &PersistentEntityId,
    corners: [[f64; 2]; 3],
) -> [ExactFaceMetricEvaluation; 4] {
    let weights = [
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [1.0 / 3.0; 3],
    ];
    weights.map(|weights| {
        evaluation(
            face_id,
            std::array::from_fn(|axis| {
                corners[0][axis] * weights[0]
                    + corners[1][axis] * weights[1]
                    + corners[2][axis] * weights[2]
            }),
        )
    })
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
            source: ExactFacePslgSegmentSource::ExactTrim {
                source_coedge_id: id(PersistentEntityKind::Coedge, "coedge"),
                source_edge_id: id(PersistentEntityKind::Edge, "edge"),
            },
            vertex_indices: [0, 1],
            edge_parameters: Some([6.0, 2.0]),
        }],
        loops: Vec::new(),
    }
}

fn paired_cut_pslg(source_face_id: PersistentEntityId) -> ExactFacePslg {
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
            ExactFacePslgVertex {
                node_id: node(2),
                seam_image: None,
                uv: [1.0, 1.0],
            },
            ExactFacePslgVertex {
                node_id: node(1),
                seam_image: None,
                uv: [-1.0, 1.0],
            },
        ],
        segments: vec![
            ExactFacePslgSegment {
                source: ExactFacePslgSegmentSource::ChartCut { cut_id: node(9) },
                vertex_indices: [0, 1],
                edge_parameters: None,
            },
            ExactFacePslgSegment {
                source: ExactFacePslgSegmentSource::ChartCut { cut_id: node(9) },
                vertex_indices: [2, 3],
                edge_parameters: None,
            },
        ],
        loops: Vec::new(),
    }
}

fn acute_corner_geometry() -> (ExactFacePslg, ExactFaceGeometry) {
    let face_id = id(PersistentEntityKind::Face, "acute-face");
    let angle = 10.0_f64.to_radians();
    let uvs = [[0.0, 0.0], [angle.cos(), angle.sin()], [1.0, 0.0]];
    let pslg = ExactFacePslg {
        source_face_id: face_id.clone(),
        vertices: uvs
            .into_iter()
            .enumerate()
            .map(|(index, uv)| ExactFacePslgVertex {
                node_id: node(index as u8 + 1),
                seam_image: None,
                uv,
            })
            .collect(),
        segments: (0..3)
            .map(|index| ExactFacePslgSegment {
                source: ExactFacePslgSegmentSource::ExactTrim {
                    source_coedge_id: id(
                        PersistentEntityKind::Coedge,
                        &format!("acute-coedge-{index}"),
                    ),
                    source_edge_id: id(PersistentEntityKind::Edge, &format!("acute-edge-{index}")),
                },
                vertex_indices: [index, (index + 1) % 3],
                edge_parameters: Some([0.0, 1.0]),
            })
            .collect(),
        loops: vec![ExactFacePslgLoop {
            source: ExactFacePslgLoopSource::ExactWire {
                source_wire_id: id(PersistentEntityKind::Wire, "acute-wire"),
            },
            orientation: TopologicalOrientation::Forward,
            first_segment: 0,
            segment_count: 3,
        }],
    };
    let centroid_uv = [
        uvs.iter().map(|uv| uv[0]).sum::<f64>() / 3.0,
        uvs.iter().map(|uv| uv[1]).sum::<f64>() / 3.0,
    ];
    let geometry = ExactFaceGeometry {
        source_face_id: face_id.clone(),
        vertices: uvs
            .into_iter()
            .enumerate()
            .map(|(index, uv)| ExactFaceGeometryVertex {
                pslg_vertex_index: index as u32,
                point_m: evaluation(&face_id, uv).point_m,
                evaluation: evaluation(&face_id, uv),
                unit_normal: [0.0, 0.0, 1.0],
            })
            .collect(),
        triangles: vec![ExactFaceTriangleGeometry {
            triangle: ExactFaceDelaunayTriangle {
                vertex_indices: [0, 1, 2],
            },
            deviation_samples: deviation_samples(&face_id, uvs),
            unit_normal: [0.0, 0.0, 1.0],
            physical_area_m2: 0.1,
            metric_edge_lengths: [0.5; 3],
            minimum_metric_angle_rad: 5.0_f64.to_radians(),
            physical_aspect_ratio: 100.0,
            chordal_deviation_m: 0.0,
            chordal_deviation_uv: centroid_uv,
            normal_deviation_rad: 0.0,
            normal_deviation_uv: centroid_uv,
        }],
        maximum_metric_edge_length: 0.5,
        minimum_metric_angle_rad: 5.0_f64.to_radians(),
        maximum_physical_aspect_ratio: 100.0,
        maximum_chordal_deviation_m: runmat_types::physical::Length::new(0.0),
        maximum_normal_deviation_rad: 0.0,
    };
    (pslg, geometry)
}

fn triangle_pslg(geometry: &ExactFaceGeometry) -> ExactFacePslg {
    ExactFacePslg {
        source_face_id: geometry.source_face_id.clone(),
        vertices: geometry
            .vertices
            .iter()
            .enumerate()
            .map(|(index, vertex)| ExactFacePslgVertex {
                node_id: node(index as u8 + 1),
                seam_image: None,
                uv: vertex.evaluation.uv,
            })
            .collect(),
        segments: Vec::new(),
        loops: Vec::new(),
    }
}

fn empty_collars(geometry: &ExactFaceGeometry) -> ExactFaceFeatureCollars {
    ExactFaceFeatureCollars {
        source_face_id: geometry.source_face_id.clone(),
        collars: Vec::new(),
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
        minimum_metric_edge_length: 0.01,
        minimum_metric_angle_degrees: 10.0,
        maximum_physical_aspect_ratio: 10.0,
        maximum_chordal_deviation_m: runmat_types::physical::Length::new(0.1),
        maximum_normal_deviation_degrees: 30.0,
    }
}

fn metric_request() -> MetricFieldRequest {
    MetricFieldRequest {
        entity_supports: Vec::new(),
        located_sources: Vec::new(),
        combination: MetricCombinationRule::MostRestrictiveIntersection,
        global_metric: MetricTensor3::isotropic_length_m(1.0).unwrap(),
        maximum_grading_ratio: 2.0,
        contributions: Vec::new(),
        cells_across_gap: None,
    }
}

fn curve_options() -> SharedCurveDiscretizationOptions {
    SharedCurveDiscretizationOptions {
        resolution: CurveResolutionPolicy {
            maximum_chordal_deviation_m: runmat_types::physical::Length::new(0.01),
            maximum_tangent_change_rad: 0.2,
            minimum_metric_edge_length: 0.01,
            maximum_metric_edge_length: 1.0,
        },
        maximum_nodes_per_edge: 1_024,
        maximum_subdivision_depth: 20,
        geometry_absolute_error_m: runmat_types::physical::Length::new(1.0e-10),
        pcurve_absolute_error: 1.0e-10,
        arc_length_absolute_error_m: runmat_types::physical::Length::new(1.0e-10),
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
    PersistentEntityId::new(kind, name.into(), vec!["part".into()])
}

struct Control;

struct CancelledSignal;

impl MeshingCancellationSignal for CancelledSignal {
    fn is_cancelled(&self) -> bool {
        true
    }
}

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
