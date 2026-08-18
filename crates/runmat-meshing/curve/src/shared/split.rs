use std::collections::BTreeMap;

use runmat_geometry_core::PersistentEntityId;

use super::{
    discretize::{
        discretize_edge_with_parameters, SharedCurveDiscretizationOptions,
        SharedCurveEvaluationContext,
    },
    validate_shared_curve_geometry, SharedCurveError, SharedCurveErrorKind, SharedCurveMesh,
    SharedCurveSegmentSplit,
};

const MAX_SPLIT_REQUESTS: usize = 10_000_000;

pub fn canonicalize_shared_curve_splits(splits: &mut Vec<SharedCurveSegmentSplit>) {
    splits.sort_by(compare_splits);
    splits.dedup();
}

pub fn validate_shared_curve_split_set(
    existing: &SharedCurveMesh,
    topology: &runmat_geometry_core::ExactBRepTopology,
    splits: &[SharedCurveSegmentSplit],
) -> Result<(), SharedCurveError> {
    existing.validate_against(topology)?;
    if splits.is_empty() || splits.len() > MAX_SPLIT_REQUESTS {
        return Err(SharedCurveError::invalid_request(
            "protected curve splits",
            "split inventory must be nonempty and within its hard bound",
        ));
    }
    let curves_by_edge = existing
        .edges
        .iter()
        .map(|curve| (&curve.source_edge_id, curve))
        .collect::<BTreeMap<_, _>>();
    for split in splits {
        validate_split(split, &curves_by_edge)?;
    }
    if splits
        .windows(2)
        .any(|pair| compare_splits(&pair[0], &pair[1]) != std::cmp::Ordering::Less)
    {
        return Err(SharedCurveError::invalid_request(
            "protected curve splits",
            "split inventory is not in unique canonical order",
        ));
    }
    Ok(())
}

pub fn apply_shared_curve_splits(
    existing: &SharedCurveMesh,
    context: SharedCurveEvaluationContext<'_>,
    options: SharedCurveDiscretizationOptions,
    splits: &[SharedCurveSegmentSplit],
) -> Result<SharedCurveMesh, SharedCurveError> {
    existing.validate_against(context.topology)?;
    if splits.is_empty() || splits.len() > MAX_SPLIT_REQUESTS {
        return Err(SharedCurveError::invalid_request(
            "protected curve splits",
            "split inventory must be nonempty and within its hard bound",
        ));
    }
    let curves_by_edge = existing
        .edges
        .iter()
        .map(|curve| (&curve.source_edge_id, curve))
        .collect::<BTreeMap<_, _>>();
    let mut requested = BTreeMap::<PersistentEntityId, Vec<&SharedCurveSegmentSplit>>::new();
    for split in splits {
        validate_split(split, &curves_by_edge)?;
        requested
            .entry(split.source_edge_id.clone())
            .or_default()
            .push(split);
    }

    let mut refined = existing.clone();
    for (edge_id, mut edge_splits) in requested {
        edge_splits.sort_by(|left, right| {
            left.split_parameter
                .total_cmp(&right.split_parameter)
                .then_with(|| left.edge_parameters[0].total_cmp(&right.edge_parameters[0]))
                .then_with(|| left.edge_parameters[1].total_cmp(&right.edge_parameters[1]))
        });
        edge_splits.dedup_by(|left, right| *left == *right);
        let curve_index = refined
            .edges
            .binary_search_by(|curve| curve.source_edge_id.cmp(&edge_id))
            .map_err(|_| invalid_split(&edge_id, "source edge is absent from shared curves"))?;
        let edge = context
            .topology
            .edges
            .binary_search_by(|edge| edge.id.cmp(&edge_id))
            .ok()
            .map(|index| &context.topology.edges[index])
            .ok_or_else(|| invalid_split(&edge_id, "source edge is absent from exact topology"))?;
        let mut required = refined.edges[curve_index]
            .nodes
            .iter()
            .skip(1)
            .take(refined.edges[curve_index].nodes.len().saturating_sub(2))
            .map(|node| node.parameter)
            .chain(edge_splits.into_iter().map(|split| split.split_parameter))
            .collect::<Vec<_>>();
        required.sort_by(f64::total_cmp);
        required.dedup_by(|left, right| left.to_bits() == right.to_bits());
        refined.edges[curve_index] =
            discretize_edge_with_parameters(context, edge, &required, options)?;
    }
    validate_shared_curve_geometry(
        &refined,
        context.topology,
        context.curves,
        context.pcurves,
        context.metric_field,
        context.control,
        options,
    )?;
    Ok(refined)
}

fn validate_split(
    split: &SharedCurveSegmentSplit,
    curves: &BTreeMap<&PersistentEntityId, &super::SharedCurve>,
) -> Result<(), SharedCurveError> {
    let curve = curves
        .get(&split.source_edge_id)
        .ok_or_else(|| invalid_split(&split.source_edge_id, "source edge is absent"))?;
    if split.edge_parameters.iter().any(|value| !value.is_finite())
        || !split.split_parameter.is_finite()
        || split.edge_parameters[0] >= split.split_parameter
        || split.split_parameter >= split.edge_parameters[1]
    {
        return Err(invalid_split(
            &split.source_edge_id,
            "split parameters must form one finite strict interior interval",
        ));
    }
    let owns_segment = curve.nodes.windows(2).any(|nodes| {
        [nodes[0].node_id, nodes[1].node_id] == split.endpoint_node_ids
            && [nodes[0].parameter, nodes[1].parameter] == split.edge_parameters
    });
    if !owns_segment {
        return Err(invalid_split(
            &split.source_edge_id,
            "split interval is not one current canonical shared-curve segment",
        ));
    }
    Ok(())
}

fn compare_splits(
    left: &SharedCurveSegmentSplit,
    right: &SharedCurveSegmentSplit,
) -> std::cmp::Ordering {
    left.source_edge_id
        .cmp(&right.source_edge_id)
        .then_with(|| left.edge_parameters[0].total_cmp(&right.edge_parameters[0]))
        .then_with(|| left.edge_parameters[1].total_cmp(&right.edge_parameters[1]))
        .then_with(|| left.split_parameter.total_cmp(&right.split_parameter))
        .then_with(|| left.endpoint_node_ids.cmp(&right.endpoint_node_ids))
}

fn invalid_split(edge_id: &PersistentEntityId, reason: &str) -> SharedCurveError {
    SharedCurveError::new(
        SharedCurveErrorKind::InvalidRequest,
        "protected curve split",
        reason,
    )
    .for_edge(edge_id)
}

#[cfg(test)]
#[path = "split_tests.rs"]
mod tests;
