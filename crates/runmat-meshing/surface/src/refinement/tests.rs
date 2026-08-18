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
    exact_face_chart_cut_node_id, ExactFaceDelaunayTriangle, ExactFaceGeometry,
    ExactFaceGeometryVertex, ExactFaceMetricEvaluation, ExactFacePslg, ExactFacePslgLoop,
    ExactFacePslgLoopSource, ExactFacePslgSegment, ExactFacePslgSegmentSource, ExactFacePslgVertex,
    ExactFaceTriangleGeometry, ParametricMetricTensor,
};

use super::*;

#[test]
fn canonical_bad_triangle_selects_its_metric_circumcenter() {
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
    assert_eq!(candidate.uv, [1.0, 1.0]);

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
fn geometric_deviation_uses_the_exact_triangle_centroid() {
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
        ExactFaceRefinementErrorKind::Delaunay(crate::ExactFaceDelaunayErrorKind::ResourceLimit)
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
        combination: MetricCombinationRule::MostRestrictiveIntersection,
        global_metric: MetricTensor3::isotropic_length_m(100.0).unwrap(),
        maximum_grading_ratio: 2.0,
        contributions: Vec::new(),
    };
    let permissive_quality = SurfaceQualityTargets {
        minimum_metric_angle_degrees: 0.01,
        maximum_physical_aspect_ratio: 1.0e9,
        maximum_chordal_deviation_m: 1.0e9,
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
    assert!(acceptance.maximum_chordal_deviation_m < 1.0e-12);
    assert!(acceptance.maximum_normal_deviation_rad < 1.0e-7);
    let mut tampered_acceptance = acceptance.clone();
    tampered_acceptance.maximum_chordal_deviation_m = 1.0;
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
    let strict = refine_exact_face_until_blocked(
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
            delaunay: options,
            refinement: ExactFaceRefinementOptions {
                maximum_interior_insertions: 1,
            },
        },
    );
    let ExactFaceRefinementOutcome::RequiresCurveSplit {
        completed_interior_insertions,
        ..
    } = strict.unwrap()
    else {
        panic!("strict boundary-adjacent refinement must request a protected split")
    };
    assert_eq!(completed_interior_insertions, 0);

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
                evaluation: evaluation(&face_id, uv),
                unit_normal: [0.0, 0.0, 1.0],
            })
            .collect(),
        triangles: vec![ExactFaceTriangleGeometry {
            triangle: ExactFaceDelaunayTriangle {
                vertex_indices: [0, 1, 2],
            },
            centroid: evaluation(&face_id, centroid_uv),
            unit_normal: [0.0, 0.0, 1.0],
            physical_area_m2: 0.1,
            metric_edge_lengths: [0.5; 3],
            minimum_metric_angle_rad: 5.0_f64.to_radians(),
            physical_aspect_ratio: 100.0,
            chordal_deviation_m: 0.0,
            normal_deviation_rad: 0.0,
        }],
        maximum_metric_edge_length: 0.5,
        minimum_metric_angle_rad: 5.0_f64.to_radians(),
        maximum_physical_aspect_ratio: 100.0,
        maximum_chordal_deviation_m: 0.0,
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
