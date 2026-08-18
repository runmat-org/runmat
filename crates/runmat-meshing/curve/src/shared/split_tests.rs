use runmat_geometry_core::{
    GeometryEvaluationControl, GeometryEvaluationError, GeometryEvaluationErrorKind, GeometryModel,
    PortableExactEvaluator,
};

use super::*;
use crate::{
    discretize_shared_curves, shared_curve_interior_node_id, CurveResolutionPolicy,
    SharedCurveDiscretizationOptions, SharedCurveEvaluationContext, SharedCurveSegmentSplit,
    UniformCurveMetric,
};

#[test]
fn exact_splits_are_deduplicated_and_order_independent() {
    let (document, topology, registry) = runmat_geometry_fixtures::exact_circle();
    let GeometryModel::ExactBRep { model } = &document.model else {
        panic!("fixture must be exact")
    };
    let evaluator = PortableExactEvaluator::new(&registry, &topology, model).unwrap();
    let metric = UniformCurveMetric::from_target_size_m(0.5).unwrap();
    let options = options();
    let curves = discretize_shared_curves(
        &topology, &evaluator, &evaluator, &metric, &Control, options,
    )
    .unwrap();
    let nodes = &curves.edges[0].nodes;
    let first = split(&curves, 0);
    let second = split(&curves, 2);
    let context =
        SharedCurveEvaluationContext::new(&topology, &evaluator, &evaluator, &metric, &Control);

    let refined = apply_shared_curve_splits(
        &curves,
        context,
        options,
        &[first.clone(), second.clone(), first.clone()],
    )
    .unwrap();
    let reversed =
        apply_shared_curve_splits(&curves, context, options, &[second, first.clone()]).unwrap();
    assert_eq!(refined, reversed);
    assert_eq!(refined.edges[0].nodes.len(), nodes.len() + 2);
    let inserted = refined.edges[0]
        .nodes
        .iter()
        .find(|node| node.parameter == first.split_parameter)
        .unwrap();
    assert_eq!(
        inserted.node_id,
        shared_curve_interior_node_id(&first.source_edge_id, first.split_parameter)
    );
    assert!(refined.edges[0]
        .face_uses
        .iter()
        .all(|face_use| face_use.node_uv.len() == refined.edges[0].nodes.len()));

    let reapplied = apply_shared_curve_splits(
        &refined,
        context,
        options,
        &[SharedCurveSegmentSplit {
            source_edge_id: first.source_edge_id,
            endpoint_node_ids: [nodes[0].node_id, inserted.node_id],
            edge_parameters: [nodes[0].parameter, inserted.parameter],
            split_parameter: nodes[0].parameter * 0.5 + inserted.parameter * 0.5,
        }],
    )
    .unwrap();
    assert_eq!(
        reapplied.edges[0].nodes.len(),
        refined.edges[0].nodes.len() + 1
    );
}

fn split(curves: &SharedCurveMesh, first_index: usize) -> SharedCurveSegmentSplit {
    let curve = &curves.edges[0];
    let first = &curve.nodes[first_index];
    let second = &curve.nodes[first_index + 1];
    SharedCurveSegmentSplit {
        source_edge_id: curve.source_edge_id.clone(),
        endpoint_node_ids: [first.node_id, second.node_id],
        edge_parameters: [first.parameter, second.parameter],
        split_parameter: first.parameter * 0.5 + second.parameter * 0.5,
    }
}

#[test]
fn split_rejects_stale_segment_identity_and_noninterior_parameters() {
    let (document, topology, registry) = runmat_geometry_fixtures::exact_circle();
    let GeometryModel::ExactBRep { model } = &document.model else {
        panic!("fixture must be exact")
    };
    let evaluator = PortableExactEvaluator::new(&registry, &topology, model).unwrap();
    let metric = UniformCurveMetric::from_target_size_m(0.5).unwrap();
    let options = options();
    let curves = discretize_shared_curves(
        &topology, &evaluator, &evaluator, &metric, &Control, options,
    )
    .unwrap();
    let nodes = &curves.edges[0].nodes;
    let context =
        SharedCurveEvaluationContext::new(&topology, &evaluator, &evaluator, &metric, &Control);
    let mut stale = SharedCurveSegmentSplit {
        source_edge_id: curves.edges[0].source_edge_id.clone(),
        endpoint_node_ids: [nodes[0].node_id, nodes[1].node_id],
        edge_parameters: [nodes[0].parameter, nodes[1].parameter],
        split_parameter: nodes[0].parameter * 0.5 + nodes[1].parameter * 0.5,
    };
    stale.endpoint_node_ids[0] = nodes[2].node_id;
    assert_eq!(
        apply_shared_curve_splits(&curves, context, options, &[stale])
            .unwrap_err()
            .kind,
        crate::SharedCurveErrorKind::InvalidRequest
    );

    let noninterior = SharedCurveSegmentSplit {
        source_edge_id: curves.edges[0].source_edge_id.clone(),
        endpoint_node_ids: [nodes[0].node_id, nodes[1].node_id],
        edge_parameters: [nodes[0].parameter, nodes[1].parameter],
        split_parameter: nodes[1].parameter,
    };
    assert_eq!(
        apply_shared_curve_splits(&curves, context, options, &[noninterior])
            .unwrap_err()
            .kind,
        crate::SharedCurveErrorKind::InvalidRequest
    );

    let mut limited = options;
    limited.maximum_nodes_per_edge = curves.edges[0].nodes.len() as u32;
    assert_eq!(
        apply_shared_curve_splits(&curves, context, limited, &[split(&curves, 0)])
            .unwrap_err()
            .kind,
        crate::SharedCurveErrorKind::ResourceLimit
    );

    let cancelled = SharedCurveEvaluationContext::new(
        &topology,
        &evaluator,
        &evaluator,
        &metric,
        &CancelledControl,
    );
    assert_eq!(
        apply_shared_curve_splits(&curves, cancelled, options, &[split(&curves, 0)])
            .unwrap_err()
            .kind,
        crate::SharedCurveErrorKind::GeometryEvaluation(GeometryEvaluationErrorKind::Cancelled)
    );
}

fn options() -> SharedCurveDiscretizationOptions {
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

struct CancelledControl;

impl GeometryEvaluationControl for CancelledControl {
    fn checkpoint(&self) -> Result<(), GeometryEvaluationError> {
        Err(GeometryEvaluationError::new(
            GeometryEvaluationErrorKind::Cancelled,
            "cancelled",
        ))
    }

    fn consume_iterations(&self, _count: u64) -> Result<(), GeometryEvaluationError> {
        self.checkpoint()
    }

    fn consume_search_work(&self, _count: u64) -> Result<(), GeometryEvaluationError> {
        self.checkpoint()
    }

    fn consume_allocation_bytes(&self, _count: u64) -> Result<(), GeometryEvaluationError> {
        self.checkpoint()
    }
}
