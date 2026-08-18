use runmat_geometry_core::{
    GeometryEvaluationControl, GeometryEvaluationError, GeometryModel, PortableExactEvaluator,
};
use runmat_meshing_core::{
    MetricCombinationRule, MetricFieldRequest, MetricTensor3, NeverCancelled, SurfaceQualityTargets,
};
use runmat_meshing_curve::{
    canonicalize_shared_curve_splits, discretize_shared_curves, CurveResolutionPolicy,
    SharedCurveDiscretizationOptions, SharedCurveEvaluationContext, SharedCurveSegmentSplit,
    UniformCurveMetric,
};

use crate::{
    build_exact_face_partition_result, encode_exact_face_partition_result,
    face_partition_descriptors, mesh_exact_face_partition, resolve_exact_surface_pass,
    ExactFacePartitionContext, ExactFacePartitionOptions, ExactFacePartitionOutcome,
    ExactSurfaceConvergenceOutcome, ExactSurfaceJoinOptions,
};

#[test]
fn complete_face_partition_runs_the_exact_surface_pipeline() {
    let (document, topology, registry) = runmat_geometry_fixtures::exact_circle();
    let GeometryModel::ExactBRep { model } = &document.model else {
        panic!("fixture must be exact")
    };
    let evaluator = PortableExactEvaluator::new(&registry, &topology, model).unwrap();
    let curve_metric = UniformCurveMetric::from_target_size_m(0.5).unwrap();
    let curves = discretize_shared_curves(
        &topology,
        &evaluator,
        &evaluator,
        &curve_metric,
        &Control,
        curve_options(),
    )
    .unwrap();
    let metric_request = MetricFieldRequest {
        combination: MetricCombinationRule::MostRestrictiveIntersection,
        global_metric: MetricTensor3::isotropic_length_m(100.0).unwrap(),
        maximum_grading_ratio: 2.0,
        contributions: Vec::new(),
    };
    let partitions = face_partition_descriptors(&topology, 1).unwrap();
    let context = ExactFacePartitionContext {
        topology: &topology,
        curves: &curves,
        metric_request: &metric_request,
        quality: permissive_quality(),
        evaluator: &evaluator,
        geometry_control: &Control,
        cancellation: &NeverCancelled,
    };
    let outcome = mesh_exact_face_partition(
        partitions[0].clone(),
        context,
        ExactFacePartitionOptions::default(),
    )
    .unwrap();
    let ExactFacePartitionOutcome::Converged { faces } = &outcome.outcome else {
        panic!("permissive planar face must converge without a curve restart")
    };

    assert_eq!(outcome.partition, partitions[0]);
    assert_eq!(faces.len(), 1);
    assert_eq!(faces[0].source_face_id, topology.faces[0].id);
    assert!(!faces[0].triangles.is_empty());
    let mut sheet_topology = topology.clone();
    sheet_topology.bodies[0].is_sheet_body = true;
    sheet_topology.bodies[0].sheet_shell_ids = vec![sheet_topology.shells[0].id.clone()];
    sheet_topology.bodies[0].lump_ids.clear();
    sheet_topology.lumps.clear();
    sheet_topology.solids.clear();
    sheet_topology.regions.clear();
    let converged = resolve_exact_surface_pass(
        &curves,
        vec![outcome.clone()],
        SharedCurveEvaluationContext::new(
            &sheet_topology,
            &evaluator,
            &evaluator,
            &curve_metric,
            &Control,
        ),
        curve_options(),
        ExactSurfaceJoinOptions::default(),
    )
    .unwrap();
    let ExactSurfaceConvergenceOutcome::Converged(surface) = converged else {
        panic!("complete face results must converge to the exact sheet surface")
    };
    assert_eq!(surface.face_ids, vec![topology.faces[0].id.clone()]);
    assert_eq!(surface.shells[0].open_edge_count, 1);
    let encoded = encode_exact_face_partition_result(&outcome, &topology, &curves).unwrap();
    assert_eq!(
        crate::decode_exact_face_partition_result(&encoded, &topology, &curves).unwrap(),
        outcome
    );
    assert_eq!(
        crate::surface_mesh::decode_exact_face_partition_result_with_byte_limit(
            &encoded,
            &topology,
            &curves,
            encoded.len() - 1,
        )
        .unwrap_err()
        .kind,
        crate::ExactSurfaceMeshErrorKind::InvalidEncoding
    );

    let curve = &curves.edges[0];
    let split = SharedCurveSegmentSplit {
        source_edge_id: curve.source_edge_id.clone(),
        endpoint_node_ids: [curve.nodes[0].node_id, curve.nodes[1].node_id],
        edge_parameters: [curve.nodes[0].parameter, curve.nodes[1].parameter],
        split_parameter: (curve.nodes[0].parameter + curve.nodes[1].parameter) * 0.5,
    };
    let restart = build_exact_face_partition_result(
        &topology,
        &curves,
        partitions[0].clone(),
        ExactFacePartitionOutcome::RequiresCurveSplits {
            splits: vec![split.clone()],
        },
    )
    .unwrap();
    let restart_bytes = encode_exact_face_partition_result(&restart, &topology, &curves).unwrap();
    assert_eq!(
        crate::decode_exact_face_partition_result(&restart_bytes, &topology, &curves).unwrap(),
        restart
    );
    let refined = resolve_exact_surface_pass(
        &curves,
        vec![restart.clone()],
        SharedCurveEvaluationContext::new(
            &sheet_topology,
            &evaluator,
            &evaluator,
            &curve_metric,
            &Control,
        ),
        curve_options(),
        ExactSurfaceJoinOptions::default(),
    )
    .unwrap();
    let ExactSurfaceConvergenceOutcome::RefinedCurves(refined) = refined else {
        panic!("curve-restart result must refine the shared curve")
    };
    assert_eq!(
        refined.edges[0].nodes.len(),
        curves.edges[0].nodes.len() + 1
    );
    assert!(crate::validate_exact_face_partition_result(&outcome, &topology, &refined).is_err());
    let mut corrupted_restart = restart_bytes;
    corrupted_restart[0] ^= 1;
    assert_eq!(
        crate::decode_exact_face_partition_result(&corrupted_restart, &topology, &curves)
            .unwrap_err()
            .kind,
        crate::ExactSurfaceMeshErrorKind::InvalidEncoding
    );
    assert!(build_exact_face_partition_result(
        &topology,
        &curves,
        partitions[0].clone(),
        ExactFacePartitionOutcome::RequiresCurveSplits {
            splits: vec![split.clone(), split.clone()],
        },
    )
    .is_err());
    let mut unrelated = split;
    unrelated.source_edge_id.source_topology_id = "unrelated-edge".into();
    assert!(build_exact_face_partition_result(
        &topology,
        &curves,
        partitions[0].clone(),
        ExactFacePartitionOutcome::RequiresCurveSplits {
            splits: vec![unrelated],
        },
    )
    .is_err());

    let mut fabricated_range = partitions[0].clone();
    fabricated_range
        .entity_range
        .as_mut()
        .unwrap()
        .first
        .source_topology_id
        .clear();
    assert!(mesh_exact_face_partition(
        fabricated_range,
        context,
        ExactFacePartitionOptions::default(),
    )
    .is_err());
}

#[test]
fn curve_split_demands_have_one_canonical_order() {
    let (_, topology, _) = runmat_geometry_fixtures::exact_circle();
    let edge_id = topology.edges[0].id.clone();
    let endpoints = [
        runmat_meshing_core::StableDigest::from_bytes([1; 32]),
        runmat_meshing_core::StableDigest::from_bytes([2; 32]),
    ];
    let first = SharedCurveSegmentSplit {
        source_edge_id: edge_id.clone(),
        endpoint_node_ids: endpoints,
        edge_parameters: [0.0, 1.0],
        split_parameter: 0.25,
    };
    let second = SharedCurveSegmentSplit {
        source_edge_id: edge_id,
        endpoint_node_ids: endpoints,
        edge_parameters: [0.0, 1.0],
        split_parameter: 0.75,
    };
    let mut demands = vec![second.clone(), first.clone(), second.clone()];

    canonicalize_shared_curve_splits(&mut demands);

    assert_eq!(demands, vec![first, second]);
}

fn permissive_quality() -> SurfaceQualityTargets {
    SurfaceQualityTargets {
        minimum_metric_angle_degrees: 0.01,
        maximum_physical_aspect_ratio: 1.0e9,
        maximum_chordal_deviation_m: 1.0e9,
        maximum_normal_deviation_degrees: 180.0,
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
