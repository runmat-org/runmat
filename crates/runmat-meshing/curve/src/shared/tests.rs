use std::f64::consts::{PI, TAU};
use std::sync::atomic::{AtomicU64, Ordering};

use runmat_geometry_core::ParameterRange;

use super::*;

#[test]
fn shared_curve_contract_covers_exact_edges_and_face_uses() {
    let (_, topology, _) = runmat_geometry_fixtures::exact_circle();
    let mesh = circle_mesh(&topology);
    mesh.validate_against(&topology).unwrap();

    let mut reordered = mesh.clone();
    reordered.edges[0].nodes.swap(0, 1);
    assert!(reordered.validate_against(&topology).is_err());

    let mut missing_use = mesh.clone();
    missing_use.edges[0].face_uses.clear();
    assert!(missing_use.validate_against(&topology).is_err());

    let mut tampered_identity = mesh.clone();
    tampered_identity.edges[0].nodes[1].node_id = tampered_identity.edges[0].nodes[0].node_id;
    assert!(tampered_identity.validate_against(&topology).is_err());

    let mut missed_bound = mesh;
    missed_bound.edges[0].achieved.maximum_chordal_deviation_m = 0.2;
    assert!(missed_bound.validate_against(&topology).is_err());
}

#[test]
fn shared_curve_codec_is_canonical_bounded_and_topology_admitted() {
    let (_, topology, _) = runmat_geometry_fixtures::exact_circle();
    let mesh = circle_mesh(&topology);
    let encoded = encode_shared_curve_mesh(&mesh, &topology).unwrap();
    let decoded = decode_shared_curve_mesh(&encoded, &topology).unwrap();
    assert_eq!(decoded, mesh);
    assert_eq!(
        encode_shared_curve_mesh(&decoded, &topology).unwrap(),
        encoded
    );

    let mut trailing = encoded.clone();
    trailing.push(0);
    assert!(decode_shared_curve_mesh(&trailing, &topology).is_err());

    let mut corrupt = encoded;
    corrupt[0] ^= 1;
    assert!(decode_shared_curve_mesh(&corrupt, &topology).is_err());

    let encoded = encode_shared_curve_mesh(&mesh, &topology).unwrap();
    assert!(super::codec::decode_with_byte_limit(&encoded, &topology, encoded.len() - 1).is_err());

    let mut different_topology = topology.clone();
    different_topology.edges.clear();
    different_topology.coedges.clear();
    assert!(decode_shared_curve_mesh(&encoded, &different_topology,).is_err());
}

#[test]
fn exact_circle_is_constructively_discretized_once_with_pcurve_images() {
    let (document, topology, registry) = runmat_geometry_fixtures::exact_circle();
    let runmat_geometry_core::GeometryModel::ExactBRep { model } = &document.model else {
        panic!("fixture must be exact");
    };
    let evaluator =
        runmat_geometry_core::PortableExactEvaluator::new(&registry, &topology, model).unwrap();
    let metric = UniformCurveMetric::from_target_size_m(0.25).unwrap();
    let options = shared_options();
    let mesh = discretize_shared_curves(
        &topology,
        &evaluator,
        &evaluator,
        &metric,
        &UnlimitedControl,
        options,
    )
    .unwrap();

    assert_eq!(mesh.edges.len(), 1);
    let curve = &mesh.edges[0];
    assert!(curve.nodes.len() > 8);
    assert_eq!(curve.face_uses.len(), 1);
    assert_eq!(curve.face_uses[0].node_uv.len(), curve.nodes.len());
    assert!((curve.nodes.last().unwrap().arc_length_m - TAU).abs() < 1.0e-10);
    assert!(
        curve.achieved.maximum_chordal_deviation_m
            <= options.resolution.maximum_chordal_deviation_m
    );
    assert!(
        curve.achieved.maximum_tangent_change_rad <= options.resolution.maximum_tangent_change_rad
    );
    assert!(
        curve.achieved.maximum_metric_edge_length <= options.resolution.maximum_metric_edge_length
    );
    assert_eq!(
        discretize_shared_curves(
            &topology,
            &evaluator,
            &evaluator,
            &metric,
            &UnlimitedControl,
            options,
        )
        .unwrap(),
        mesh
    );
    let report = validate_shared_curve_geometry(
        &mesh,
        &topology,
        &evaluator,
        &evaluator,
        &metric,
        &UnlimitedControl,
        options,
    )
    .unwrap();
    assert_eq!(report.edge_count, 1);
    assert_eq!(report.node_count, curve.nodes.len() as u64);
    let CurveMetricResolutionEvidence::Evaluated {
        evaluation_count, ..
    } = curve.metric_resolution
    else {
        panic!("ordinary edge must retain evaluated metric evidence")
    };
    assert_eq!(report.metric_evaluation_count, evaluation_count);
}

#[test]
fn independent_curve_validation_rejects_geometry_uv_arc_and_evidence_tampering() {
    let (document, topology, registry) = runmat_geometry_fixtures::exact_circle();
    let runmat_geometry_core::GeometryModel::ExactBRep { model } = &document.model else {
        panic!("fixture must be exact");
    };
    let evaluator =
        runmat_geometry_core::PortableExactEvaluator::new(&registry, &topology, model).unwrap();
    let metric = UniformCurveMetric::from_target_size_m(0.25).unwrap();
    let options = shared_options();
    let mesh = discretize_shared_curves(
        &topology,
        &evaluator,
        &evaluator,
        &metric,
        &UnlimitedControl,
        options,
    )
    .unwrap();
    let validate = |candidate: &SharedCurveMesh| {
        validate_shared_curve_geometry(
            candidate,
            &topology,
            &evaluator,
            &evaluator,
            &metric,
            &UnlimitedControl,
            options,
        )
    };

    let cancelled = validate_shared_curve_geometry(
        &mesh,
        &topology,
        &evaluator,
        &evaluator,
        &metric,
        &CancelledControl,
        options,
    )
    .unwrap_err();
    assert_eq!(
        cancelled.kind,
        SharedCurveErrorKind::GeometryEvaluation(
            runmat_geometry_core::GeometryEvaluationErrorKind::Cancelled
        )
    );

    let mut coordinates = mesh.clone();
    coordinates.edges[0].nodes[1].coordinates_m[0] += 1.0e-4;
    assert_eq!(
        validate(&coordinates).unwrap_err().kind,
        SharedCurveErrorKind::GeometricMismatch
    );

    let mut uv = mesh.clone();
    uv.edges[0].face_uses[0].node_uv[1][0] += 1.0e-4;
    assert_eq!(
        validate(&uv).unwrap_err().kind,
        SharedCurveErrorKind::GeometricMismatch
    );

    let mut arc = mesh.clone();
    arc.edges[0].nodes[1].arc_length_m += 1.0e-4;
    assert_eq!(
        validate(&arc).unwrap_err().kind,
        SharedCurveErrorKind::GeometricMismatch
    );

    let mut understated = mesh.clone();
    understated.edges[0].achieved.maximum_chordal_deviation_m = 1.0e-12;
    assert_eq!(
        validate(&understated).unwrap_err().kind,
        SharedCurveErrorKind::GeometricMismatch
    );

    let mut false_count = mesh.clone();
    let CurveMetricResolutionEvidence::Evaluated {
        applied_contribution_count,
        ..
    } = &mut false_count.edges[0].metric_resolution
    else {
        panic!("ordinary edge must retain evaluated metric evidence")
    };
    *applied_contribution_count += 1;
    assert_eq!(
        validate(&false_count).unwrap_err().kind,
        SharedCurveErrorKind::GeometricMismatch
    );

    let mut false_source = mesh;
    let CurveMetricResolutionEvidence::Evaluated { active_sources, .. } =
        &mut false_source.edges[0].metric_resolution
    else {
        panic!("ordinary edge must retain evaluated metric evidence")
    };
    *active_sources = vec![
        runmat_meshing_core::MetricSourceKind::Global,
        runmat_meshing_core::MetricSourceKind::Feature,
    ];
    assert_eq!(
        validate(&false_source).unwrap_err().kind,
        SharedCurveErrorKind::GeometricMismatch
    );
}

#[test]
fn constructive_curve_discretization_fails_at_its_hard_node_limit() {
    let (document, topology, registry) = runmat_geometry_fixtures::exact_circle();
    let runmat_geometry_core::GeometryModel::ExactBRep { model } = &document.model else {
        panic!("fixture must be exact");
    };
    let evaluator =
        runmat_geometry_core::PortableExactEvaluator::new(&registry, &topology, model).unwrap();
    let metric = UniformCurveMetric::from_target_size_m(0.25).unwrap();
    let mut options = shared_options();
    options.maximum_nodes_per_edge = 2;
    let error = discretize_shared_curves(
        &topology,
        &evaluator,
        &evaluator,
        &metric,
        &UnlimitedControl,
        options,
    )
    .unwrap_err();
    assert_eq!(error.kind, SharedCurveErrorKind::ResourceLimit);
    assert_eq!(error.edge_id, Some(topology.edges[0].id.clone()));
}

#[test]
fn constructive_curve_discretization_reports_incompatible_minimum_length() {
    let (document, topology, registry) = runmat_geometry_fixtures::exact_circle();
    let runmat_geometry_core::GeometryModel::ExactBRep { model } = &document.model else {
        panic!("fixture must be exact");
    };
    let evaluator =
        runmat_geometry_core::PortableExactEvaluator::new(&registry, &topology, model).unwrap();
    let metric = UniformCurveMetric::from_target_size_m(0.25).unwrap();
    let mut options = shared_options();
    options.resolution.minimum_metric_edge_length = 0.9;
    let error = discretize_shared_curves(
        &topology,
        &evaluator,
        &evaluator,
        &metric,
        &UnlimitedControl,
        options,
    )
    .unwrap_err();
    assert_eq!(error.kind, SharedCurveErrorKind::UnsatisfiedConstraint);
    assert_eq!(error.field, "minimum metric edge length");
}

#[test]
fn constructive_curves_apply_occurrence_scale_to_coordinates_and_arc_length() {
    let (document, mut topology, registry) = runmat_geometry_fixtures::exact_circle();
    topology.instances[0].transform = runmat_geometry_core::GeometryTransform([
        2.0, 0.0, 0.0, 3.0, 0.0, 2.0, 0.0, 4.0, 0.0, 0.0, 2.0, 5.0, 0.0, 0.0, 0.0, 1.0,
    ]);
    let runmat_geometry_core::GeometryModel::ExactBRep { model } = &document.model else {
        panic!("fixture must be exact");
    };
    let evaluator =
        runmat_geometry_core::PortableExactEvaluator::new(&registry, &topology, model).unwrap();
    let metric = UniformCurveMetric::from_target_size_m(0.5).unwrap();
    let mesh = discretize_shared_curves(
        &topology,
        &evaluator,
        &evaluator,
        &metric,
        &UnlimitedControl,
        shared_options(),
    )
    .unwrap();
    let curve = &mesh.edges[0];
    assert_eq!(curve.nodes[0].coordinates_m, [5.0, 4.0, 5.0]);
    assert!((curve.nodes.last().unwrap().arc_length_m - 2.0 * TAU).abs() < 1.0e-10);
}

#[test]
fn constructive_curves_integrate_arc_length_under_nonconformal_affine_placement() {
    let (document, mut topology, registry) = runmat_geometry_fixtures::exact_circle();
    topology.instances[0].transform = runmat_geometry_core::GeometryTransform([
        2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]);
    let runmat_geometry_core::GeometryModel::ExactBRep { model } = &document.model else {
        panic!("fixture must be exact");
    };
    let evaluator =
        runmat_geometry_core::PortableExactEvaluator::new(&registry, &topology, model).unwrap();
    let metric = UniformCurveMetric::from_target_size_m(0.5).unwrap();
    let mesh = discretize_shared_curves(
        &topology,
        &evaluator,
        &evaluator,
        &metric,
        &UnlimitedControl,
        shared_options(),
    )
    .unwrap();
    let circumference = mesh.edges[0].nodes.last().unwrap().arc_length_m;
    assert!((circumference - 9.688_448_220_547_675).abs() < 1.0e-8);
}

#[test]
fn constructive_curves_preserve_typed_evaluator_cancellation() {
    let (document, topology, registry) = runmat_geometry_fixtures::exact_circle();
    let runmat_geometry_core::GeometryModel::ExactBRep { model } = &document.model else {
        panic!("fixture must be exact");
    };
    let evaluator =
        runmat_geometry_core::PortableExactEvaluator::new(&registry, &topology, model).unwrap();
    let metric = UniformCurveMetric::from_target_size_m(0.25).unwrap();
    let error = discretize_shared_curves(
        &topology,
        &evaluator,
        &evaluator,
        &metric,
        &CancelledControl,
        shared_options(),
    )
    .unwrap_err();
    assert_eq!(
        error.kind,
        SharedCurveErrorKind::GeometryEvaluation(
            runmat_geometry_core::GeometryEvaluationErrorKind::Cancelled
        )
    );
}

#[test]
fn singular_edge_preserves_pcurve_interval_as_one_collapsed_mesh_node() {
    let (document, topology, registry) = runmat_geometry_fixtures::exact_singular_edge();
    let runmat_geometry_core::GeometryModel::ExactBRep { model } = &document.model else {
        panic!("fixture must be exact");
    };
    let evaluator =
        runmat_geometry_core::PortableExactEvaluator::new(&registry, &topology, model).unwrap();
    let metric = CountingMetric::new();
    let options = shared_options();
    let mesh = discretize_shared_curves(
        &topology,
        &evaluator,
        &evaluator,
        &metric,
        &UnlimitedControl,
        options,
    )
    .unwrap();
    let curve = &mesh.edges[0];
    assert_eq!(curve.nodes.len(), 2);
    assert_eq!(curve.nodes[0].node_id, curve.nodes[1].node_id);
    assert_eq!(curve.nodes[0].coordinates_m, [0.0, 0.0, 1.0]);
    assert_eq!(curve.nodes[0].coordinates_m, curve.nodes[1].coordinates_m);
    assert_eq!(curve.nodes[0].arc_length_m, 0.0);
    assert_eq!(curve.nodes[1].arc_length_m, 0.0);
    assert_eq!(curve.face_uses[0].node_uv[0], [0.0, PI / 2.0]);
    assert_eq!(curve.face_uses[0].node_uv[1], [TAU, PI / 2.0]);
    assert_eq!(
        curve.metric_resolution,
        CurveMetricResolutionEvidence::DegenerateTopologicalCollapse
    );
    assert_eq!(metric.calls.load(Ordering::Relaxed), 0);

    let encoded = encode_shared_curve_mesh(&mesh, &topology).unwrap();
    assert_eq!(decode_shared_curve_mesh(&encoded, &topology).unwrap(), mesh);
    let report = validate_shared_curve_geometry(
        &mesh,
        &topology,
        &evaluator,
        &evaluator,
        &metric,
        &UnlimitedControl,
        options,
    )
    .unwrap();
    assert_eq!(report.metric_evaluation_count, 0);
    assert_eq!(metric.calls.load(Ordering::Relaxed), 0);

    let mut fabricated_metric = mesh.clone();
    fabricated_metric.edges[0].metric_resolution = CurveMetricResolutionEvidence::Evaluated {
        active_sources: vec![runmat_meshing_core::MetricSourceKind::Global],
        evaluation_count: 2,
        applied_contribution_count: 0,
        minimum_tangent_target_size_m: 1.0,
        maximum_tangent_target_size_m: 1.0,
        clipped_contribution_count: 0,
        rejected_contribution_count: 0,
    };
    assert!(fabricated_metric.validate_against(&topology).is_err());

    let mut fabricated_length = mesh;
    fabricated_length.edges[0].nodes[1].arc_length_m = -0.0;
    assert!(fabricated_length.validate_against(&topology).is_err());
}

#[test]
fn declared_degenerate_edge_must_geometrically_collapse() {
    let (document, topology, registry) = runmat_geometry_fixtures::exact_circle();
    let runmat_geometry_core::GeometryModel::ExactBRep { model } = &document.model else {
        panic!("fixture must be exact");
    };
    let evaluator =
        runmat_geometry_core::PortableExactEvaluator::new(&registry, &topology, model).unwrap();
    let mut false_topology = topology.clone();
    false_topology.edges[0].is_degenerate = true;
    false_topology.edges[0].is_periodic = false;
    let error = discretize_shared_curves(
        &false_topology,
        &evaluator,
        &evaluator,
        &CountingMetric::new(),
        &UnlimitedControl,
        shared_options(),
    )
    .unwrap_err();
    assert_eq!(error.kind, SharedCurveErrorKind::GeometricMismatch);
    assert_eq!(error.field, "degenerate exact edge");
}

#[test]
fn resolved_curve_metric_combines_all_incident_typed_sources() {
    let (document, topology, registry) = runmat_geometry_fixtures::exact_circle();
    let runmat_geometry_core::GeometryModel::ExactBRep { model } = &document.model else {
        panic!("fixture must be exact");
    };
    let evaluator =
        runmat_geometry_core::PortableExactEvaluator::new(&registry, &topology, model).unwrap();
    let request = resolved_metric_request(&topology);
    let metric = ResolvedCurveMetricField::new(&topology, &request).unwrap();
    let edge = &topology.edges[0];
    let evaluation = metric
        .evaluate(CurveMetricQuery {
            edge_id: &edge.id,
            parameter: 0.0,
            point_m: [1.0, 0.0, 0.0],
            unit_tangent: [0.0, 1.0, 0.0],
        })
        .unwrap();
    assert_eq!(evaluation.metric.xx, 5.0);
    assert_eq!(evaluation.metric.yy, 5.0);
    assert_eq!(evaluation.metric.zz, 5.0);
    assert_eq!(evaluation.applied_contribution_count, 4);
    assert_eq!(
        evaluation.active_sources,
        vec![
            runmat_meshing_core::MetricSourceKind::Global,
            runmat_meshing_core::MetricSourceKind::Region,
            runmat_meshing_core::MetricSourceKind::Curve,
            runmat_meshing_core::MetricSourceKind::Face,
            runmat_meshing_core::MetricSourceKind::Feature,
        ]
    );

    let mesh = discretize_shared_curves(
        &topology,
        &evaluator,
        &evaluator,
        &metric,
        &UnlimitedControl,
        shared_options(),
    )
    .unwrap();
    let CurveMetricResolutionEvidence::Evaluated {
        active_sources,
        applied_contribution_count,
        ..
    } = &mesh.edges[0].metric_resolution
    else {
        panic!("ordinary edge must retain evaluated metric evidence")
    };
    assert_eq!(active_sources, &evaluation.active_sources);
    assert!(*applied_contribution_count > 4);

    let mut unknown = request;
    let runmat_meshing_core::MetricContributionScope::Entity { entity_id } =
        &mut unknown.contributions[1].scope
    else {
        panic!("curve contribution must remain entity-scoped")
    };
    entity_id.source_topology_id = "unknown-edge".into();
    assert_eq!(
        ResolvedCurveMetricField::new(&topology, &unknown)
            .unwrap_err()
            .kind,
        SharedCurveErrorKind::InvalidRequest
    );
}

fn shared_options() -> SharedCurveDiscretizationOptions {
    SharedCurveDiscretizationOptions {
        resolution: CurveResolutionPolicy {
            maximum_chordal_deviation_m: 0.01,
            maximum_tangent_change_rad: 0.2,
            minimum_metric_edge_length: 0.1,
            maximum_metric_edge_length: 1.0,
        },
        maximum_nodes_per_edge: 1_024,
        maximum_subdivision_depth: 20,
        geometry_absolute_error_m: 1.0e-10,
        pcurve_absolute_error: 1.0e-10,
        arc_length_absolute_error_m: 1.0e-10,
    }
}

fn resolved_metric_request(
    topology: &runmat_geometry_core::ExactBRepTopology,
) -> runmat_meshing_core::MetricFieldRequest {
    let unit = runmat_meshing_core::MetricTensor3::isotropic_length_m(1.0).unwrap();
    runmat_meshing_core::MetricFieldRequest {
        combination: runmat_meshing_core::MetricCombinationRule::MostRestrictiveIntersection,
        global_metric: unit,
        maximum_grading_ratio: 1.3,
        contributions: vec![
            runmat_meshing_core::MetricContribution {
                source: runmat_meshing_core::MetricSourceKind::Region,
                scope: runmat_meshing_core::MetricContributionScope::Region {
                    region_id: topology.regions[0].id.clone(),
                },
                metric: unit,
            },
            runmat_meshing_core::MetricContribution {
                source: runmat_meshing_core::MetricSourceKind::Curve,
                scope: runmat_meshing_core::MetricContributionScope::Entity {
                    entity_id: topology.edges[0].id.clone(),
                },
                metric: unit,
            },
            runmat_meshing_core::MetricContribution {
                source: runmat_meshing_core::MetricSourceKind::Face,
                scope: runmat_meshing_core::MetricContributionScope::Entity {
                    entity_id: topology.faces[0].id.clone(),
                },
                metric: unit,
            },
            runmat_meshing_core::MetricContribution {
                source: runmat_meshing_core::MetricSourceKind::Feature,
                scope: runmat_meshing_core::MetricContributionScope::Entity {
                    entity_id: topology.vertices[0].id.clone(),
                },
                metric: unit,
            },
        ],
    }
}

struct UnlimitedControl;

struct CountingMetric {
    calls: AtomicU64,
}

impl CountingMetric {
    fn new() -> Self {
        Self {
            calls: AtomicU64::new(0),
        }
    }
}

impl CurveMetricField for CountingMetric {
    fn evaluate(
        &self,
        _query: CurveMetricQuery<'_>,
    ) -> Result<CurveMetricEvaluation, SharedCurveError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        panic!("a degenerate edge must not request a tangent metric")
    }
}

impl runmat_geometry_core::GeometryEvaluationControl for UnlimitedControl {
    fn checkpoint(&self) -> Result<(), runmat_geometry_core::GeometryEvaluationError> {
        Ok(())
    }

    fn consume_iterations(
        &self,
        _count: u64,
    ) -> Result<(), runmat_geometry_core::GeometryEvaluationError> {
        Ok(())
    }

    fn consume_search_work(
        &self,
        _count: u64,
    ) -> Result<(), runmat_geometry_core::GeometryEvaluationError> {
        Ok(())
    }

    fn consume_allocation_bytes(
        &self,
        _count: u64,
    ) -> Result<(), runmat_geometry_core::GeometryEvaluationError> {
        Ok(())
    }
}

struct CancelledControl;

impl runmat_geometry_core::GeometryEvaluationControl for CancelledControl {
    fn checkpoint(&self) -> Result<(), runmat_geometry_core::GeometryEvaluationError> {
        Err(runmat_geometry_core::GeometryEvaluationError::new(
            runmat_geometry_core::GeometryEvaluationErrorKind::Cancelled,
            "cancelled by test",
        ))
    }

    fn consume_iterations(
        &self,
        _count: u64,
    ) -> Result<(), runmat_geometry_core::GeometryEvaluationError> {
        self.checkpoint()
    }

    fn consume_search_work(
        &self,
        _count: u64,
    ) -> Result<(), runmat_geometry_core::GeometryEvaluationError> {
        self.checkpoint()
    }

    fn consume_allocation_bytes(
        &self,
        _count: u64,
    ) -> Result<(), runmat_geometry_core::GeometryEvaluationError> {
        self.checkpoint()
    }
}

fn circle_mesh(topology: &runmat_geometry_core::ExactBRepTopology) -> SharedCurveMesh {
    let edge = &topology.edges[0];
    let coedge = &topology.coedges[0];
    let parameter_range = ParameterRange {
        start: 0.0,
        end: TAU,
    };
    let nodes = [
        (0.0, 0.0, [1.0, 0.0, 0.0], edge.start_vertex_id.clone()),
        (PI, PI, [-1.0, 0.0, 0.0], None),
        (TAU, TAU, [1.0, 0.0, 0.0], edge.end_vertex_id.clone()),
    ]
    .into_iter()
    .map(
        |(parameter, arc_length_m, coordinates_m, source_vertex_id)| SharedCurveNode {
            node_id: shared_curve_node_id(&edge.id, parameter),
            source_vertex_id,
            parameter,
            arc_length_m,
            coordinates_m,
        },
    )
    .collect();
    SharedCurveMesh {
        schema_version: SHARED_CURVE_MESH_SCHEMA_VERSION,
        edges: vec![SharedCurve {
            source_edge_id: edge.id.clone(),
            parameter_range,
            nodes,
            face_uses: vec![SharedCurveFaceUse {
                coedge_id: coedge.id.clone(),
                face_id: coedge.face_id.clone(),
                orientation: coedge.orientation,
                seam_image: coedge.seam_image,
                node_uv: vec![[0.0, 0.0], [PI, 0.0], [TAU, 0.0]],
            }],
            requested: CurveResolutionPolicy {
                maximum_chordal_deviation_m: 0.1,
                maximum_tangent_change_rad: PI,
                minimum_metric_edge_length: 0.5,
                maximum_metric_edge_length: 4.0,
            },
            achieved: CurveResolutionEvidence {
                maximum_chordal_deviation_m: 0.05,
                maximum_tangent_change_rad: PI,
                minimum_metric_edge_length: 2.0,
                maximum_metric_edge_length: 2.0,
            },
            metric_resolution: CurveMetricResolutionEvidence::Evaluated {
                active_sources: vec![runmat_meshing_core::MetricSourceKind::Global],
                evaluation_count: 3,
                applied_contribution_count: 0,
                minimum_tangent_target_size_m: 1.0,
                maximum_tangent_target_size_m: 1.0,
                clipped_contribution_count: 0,
                rejected_contribution_count: 0,
            },
        }],
    }
}
