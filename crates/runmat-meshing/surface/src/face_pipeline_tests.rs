use runmat_geometry_core::{
    GeometryEvaluationControl, GeometryEvaluationError, GeometryModel, PortableExactEvaluator,
};
use runmat_meshing_core::{
    MetricCombinationRule, MetricFieldRequest, MetricTensor3, NeverCancelled, SurfaceQualityTargets,
};
use runmat_meshing_curve::{
    discretize_shared_curves, CurveResolutionPolicy, SharedCurveDiscretizationOptions,
    SharedCurveSegmentSplit, UniformCurveMetric,
};

use crate::{
    face_partition_descriptors, mesh_exact_face_partition, ExactFacePartitionContext,
    ExactFacePartitionOptions, ExactFacePartitionOutcome,
};

#[test]
fn complete_face_partition_runs_the_exact_surface_pipeline() {
    let (document, topology, registry) = runmat_geometry_fixtures::exact_circle();
    let GeometryModel::ExactBRep { model } = &document.model else {
        panic!("fixture must be exact")
    };
    let evaluator = PortableExactEvaluator::new(&registry, &topology, model).unwrap();
    let curves = discretize_shared_curves(
        &topology,
        &evaluator,
        &evaluator,
        &UniformCurveMetric::from_target_size_m(0.5).unwrap(),
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
    let ExactFacePartitionOutcome::Converged(batch) = outcome else {
        panic!("permissive planar face must converge without a curve restart")
    };

    assert_eq!(batch.partition, partitions[0]);
    assert_eq!(batch.faces.len(), 1);
    assert_eq!(batch.faces[0].source_face_id, topology.faces[0].id);
    assert!(!batch.faces[0].triangles.is_empty());

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

    super::face_pipeline::canonicalize_splits(&mut demands);

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
