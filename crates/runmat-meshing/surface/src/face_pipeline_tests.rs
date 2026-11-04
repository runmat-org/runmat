mod global_validation;
mod cooperative_codec;
mod cooperative_join;
mod prepared_metric;
mod publication_scale;

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
    build_exact_face_charts, build_exact_face_partition_result, build_exact_surface_boundary,
    decide_exact_surface_pass, encode_exact_face_partition_result,
    encode_exact_surface_pass_result, face_partition_descriptors, mesh_exact_face_partition,
    resolve_exact_surface_pass, ExactFaceChartErrorKind, ExactFaceChartOptions,
    ExactFacePartitionContext, ExactFacePartitionOptions, ExactFacePartitionOutcome,
    ExactSurfaceConvergenceOutcome, ExactSurfaceJoinOptions, ExactSurfacePassOutcome,
};

// witness: FEA-SHARED-RESOURCES-001/shared_immutable_inventories @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Move-based joining reproduces the original decision and canonical bytes. Independent
//   borrowed prerequisite checks reject substituted coordinates, quality, evaluator parameters,
//   face coverage and bounds. Construction-only accepted results cannot be restored from serialized
//   flags. This is a finite lineage witness, not whole-operator storage or performance
//   qualification.
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
        &evaluator,
        &curve_metric,
        &Control,
        curve_options(),
    )
    .unwrap();
    let metric_request = MetricFieldRequest {
        entity_supports: Vec::new(),
        combination: MetricCombinationRule::MostRestrictiveIntersection,
        global_metric: MetricTensor3::isotropic_length_m(100.0).unwrap(),
        maximum_grading_ratio: 2.0,
        contributions: Vec::new(),
        cells_across_gap: None,
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

    let mut streamed = Vec::new();
    let completion = crate::visit_exact_face_partition(
        partitions[0].clone(),
        context,
        ExactFacePartitionOptions::default(),
        &mut |accepted| {
            streamed.push(accepted.encode(&NeverCancelled)?);
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(completion.accepted_face_count, faces.len());
    assert!(completion.curve_splits.is_empty());
    assert_eq!(completion.partition, partitions[0]);
    assert_eq!(
        streamed,
        vec![crate::surface_mesh::encode_accepted_face(
            &faces[0],
            256 * 1024 * 1024,
            &NeverCancelled
        )
        .unwrap()]
    );

    let borrowed =
        crate::surface_mesh::encode_accepted_face(&faces[0], 256 * 1024 * 1024, &NeverCancelled)
            .unwrap();
    assert_eq!(
        borrowed,
        encode_exact_face_partition_result(&outcome, &topology, &curves).unwrap(),
        "borrowed single-face encoding preserves the complete existing canonical record"
    );
    assert!(
        crate::surface_mesh::profile_accepted_face(&faces[0], 16, &NeverCancelled)
            .unwrap()
            .is_none()
    );
    let boundary = build_exact_surface_boundary(&topology, &curves).unwrap();
    assert_eq!(
        crate::decode_exact_face_reuse_record_with_cancellation(
            &borrowed,
            &topology,
            &curves,
            &boundary.faces[0],
            &NeverCancelled,
        )
        .unwrap(),
        faces[0]
    );
    let mut changed = faces[0].clone();
    let boundary_node = boundary.faces[0].outer_loop.segments[0].node_ids[0];
    let node = changed
        .nodes
        .iter_mut()
        .find(|node| node.node_id == boundary_node)
        .unwrap();
    node.point_m[0] = f64::from_bits(node.point_m[0].to_bits() + 1);
    let foreign =
        crate::surface_mesh::encode_accepted_face(&changed, 256 * 1024 * 1024, &NeverCancelled)
            .unwrap();
    assert!(
        crate::decode_exact_face_reuse_record_with_cancellation(
            &foreign,
            &topology,
            &curves,
            &boundary.faces[0],
            &NeverCancelled,
        )
        .is_err(),
        "local cold admission rejects a one-ULP original-coordinate substitution"
    );
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
    let converged_results = [outcome.clone()];
    let pass = decide_exact_surface_pass(
        &curves,
        &converged_results,
        &sheet_topology,
        ExactSurfaceJoinOptions::default(),
    )
    .unwrap();
    assert!(matches!(
        &pass.outcome,
        ExactSurfacePassOutcome::Converged { .. }
    ));
    cooperative_join::verify(&pass, &sheet_topology, &curves, &converged_results);
    let pass_bytes = encode_exact_surface_pass_result(
        &pass,
        &sheet_topology,
        &curves,
        &converged_results,
        ExactSurfaceJoinOptions::default(),
    )
    .unwrap();
    assert_eq!(
        crate::decode_exact_surface_pass_result(
            &pass_bytes,
            &sheet_topology,
            &curves,
            &converged_results,
            ExactSurfaceJoinOptions::default(),
        )
        .unwrap(),
        pass
    );
    assert_eq!(
        crate::surface_mesh::decode_exact_surface_pass_result_with_byte_limit(
            &pass_bytes,
            &sheet_topology,
            &curves,
            &converged_results,
            ExactSurfaceJoinOptions::default(),
            pass_bytes.len() - 1,
        )
        .unwrap_err()
        .kind,
        crate::ExactSurfaceMeshErrorKind::InvalidEncoding
    );
    let mut tampered_pass = pass.clone();
    let ExactSurfacePassOutcome::Converged { surface } = &mut tampered_pass.outcome else {
        unreachable!()
    };
    surface.maximum_chordal_deviation_m = runmat_types::physical::Quantity::new(
        surface.maximum_chordal_deviation_m.canonical_value() + (1.0),
    );
    assert!(encode_exact_surface_pass_result(
        &tampered_pass,
        &sheet_topology,
        &curves,
        &converged_results,
        ExactSurfaceJoinOptions::default(),
    )
    .is_err());
    let converged = resolve_exact_surface_pass(
        &curves,
        &converged_results,
        SharedCurveEvaluationContext::new(
            &sheet_topology,
            &evaluator,
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
    cooperative_codec::verify(&outcome, &topology, &curves);
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
    cooperative_codec::verify(&restart, &topology, &curves);
    let restart_bytes = encode_exact_face_partition_result(&restart, &topology, &curves).unwrap();
    assert_eq!(
        crate::decode_exact_face_partition_result(&restart_bytes, &topology, &curves).unwrap(),
        restart
    );
    let restart_results = [restart.clone()];
    let restart_pass = decide_exact_surface_pass(
        &curves,
        &restart_results,
        &sheet_topology,
        ExactSurfaceJoinOptions::default(),
    )
    .unwrap();
    let ExactSurfacePassOutcome::RequiresCurveSplits { splits } = &restart_pass.outcome else {
        panic!("restart pass must retain canonical curve split demands")
    };
    assert_eq!(splits, std::slice::from_ref(&split));
    assert!(crate::encode_exact_surface_mesh_from_pass(
        &restart_pass,
        &sheet_topology,
        &curves,
        &restart_results,
        ExactSurfaceJoinOptions::default(),
    )
    .is_err());
    cooperative_join::verify(&restart_pass, &sheet_topology, &curves, &restart_results);
    let restart_pass_bytes = encode_exact_surface_pass_result(
        &restart_pass,
        &sheet_topology,
        &curves,
        &restart_results,
        ExactSurfaceJoinOptions::default(),
    )
    .unwrap();
    assert_eq!(
        crate::decode_exact_surface_pass_result(
            &restart_pass_bytes,
            &sheet_topology,
            &curves,
            &restart_results,
            ExactSurfaceJoinOptions::default(),
        )
        .unwrap(),
        restart_pass
    );
    let mut tampered_restart_pass = restart_pass.clone();
    let ExactSurfacePassOutcome::RequiresCurveSplits { splits } =
        &mut tampered_restart_pass.outcome
    else {
        unreachable!()
    };
    splits.clear();
    assert!(encode_exact_surface_pass_result(
        &tampered_restart_pass,
        &sheet_topology,
        &curves,
        &restart_results,
        ExactSurfaceJoinOptions::default(),
    )
    .is_err());
    let refined = resolve_exact_surface_pass(
        &curves,
        &restart_results,
        SharedCurveEvaluationContext::new(
            &sheet_topology,
            &evaluator,
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
fn authoritative_spherical_octant_runs_the_complete_face_partition() {
    let (document, topology, registry) = runmat_geometry_fixtures::exact_spherical_octant();
    let GeometryModel::ExactBRep { model } = &document.model else {
        panic!("fixture must be exact")
    };
    let evaluator = PortableExactEvaluator::new(&registry, &topology, model).unwrap();
    let curve_metric = UniformCurveMetric::from_target_size_m(1.0).unwrap();
    let curves = discretize_shared_curves(
        &topology,
        &evaluator,
        &evaluator,
        &evaluator,
        &curve_metric,
        &Control,
        spherical_octant_curve_options(),
    )
    .unwrap();
    assert_eq!(curves.edges.len(), 3);
    let boundary = build_exact_surface_boundary(&topology, &curves).unwrap();
    let mut noncoincident_pole = boundary.faces[0].clone();
    assert_eq!(
        noncoincident_pole.outer_loop.segments[0].node_ids[1],
        noncoincident_pole.outer_loop.segments[1].node_ids[0]
    );
    noncoincident_pole.outer_loop.segments[1].node_uv[0][1] -= 0.1;
    assert_eq!(
        build_exact_face_charts(
            &noncoincident_pole,
            &topology,
            &evaluator,
            &Control,
            ExactFaceChartOptions::default(),
        )
        .unwrap_err()
        .kind,
        ExactFaceChartErrorKind::InvalidInput
    );
    let metric_request = MetricFieldRequest {
        entity_supports: Vec::new(),
        combination: MetricCombinationRule::MostRestrictiveIntersection,
        global_metric: MetricTensor3::isotropic_length_m(100.0).unwrap(),
        maximum_grading_ratio: 2.0,
        contributions: Vec::new(),
        cells_across_gap: None,
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
        panic!("authoritative singular face must converge without a curve restart")
    };
    assert_eq!(faces.len(), 1);
    assert!(faces[0]
        .nodes
        .iter()
        .any(|node| (node.point_m[2] - 1.0).abs() < 1.0e-12));
    assert!(faces[0]
        .triangles
        .iter()
        .all(|triangle| triangle.source_face_id == topology.faces[0].id));

    let results = [outcome];
    let pass = decide_exact_surface_pass(
        &curves,
        &results,
        &topology,
        ExactSurfaceJoinOptions::default(),
    )
    .unwrap();
    let ExactSurfacePassOutcome::Converged { surface } = pass.outcome else {
        panic!("authoritative singular sheet must pass the surface join")
    };
    assert_eq!(surface.face_ids, vec![topology.faces[0].id.clone()]);
    assert_eq!(surface.shells.len(), 1);
    assert_eq!(surface.shells[0].open_edge_count, 3);
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
        minimum_metric_edge_length: 0.001,
        minimum_metric_angle_degrees: 0.01,
        maximum_physical_aspect_ratio: 1.0e9,
        maximum_chordal_deviation_m: runmat_types::physical::Length::new(1.0e9),
        maximum_normal_deviation_degrees: 180.0,
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

fn spherical_octant_curve_options() -> SharedCurveDiscretizationOptions {
    SharedCurveDiscretizationOptions {
        resolution: CurveResolutionPolicy {
            maximum_chordal_deviation_m: runmat_types::physical::Length::new(1.0),
            maximum_tangent_change_rad: 2.0,
            minimum_metric_edge_length: 0.001,
            maximum_metric_edge_length: 2.0,
        },
        maximum_nodes_per_edge: 16,
        maximum_subdivision_depth: 4,
        geometry_absolute_error_m: runmat_types::physical::Length::new(1.0e-10),
        pcurve_absolute_error: 1.0e-10,
        arc_length_absolute_error_m: runmat_types::physical::Length::new(1.0e-10),
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
