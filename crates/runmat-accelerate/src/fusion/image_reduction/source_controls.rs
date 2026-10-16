//! Exact compiler-folded image controls. A bound fold is structural evidence
//! for recognition, never a replacement for the live request carrier.

use super::super::ImageNormalizePattern;
use super::ImageReductionRoutes;
use crate::fusion::resolve_numeric_vector_constant;
use crate::graph::{
    AccelGraph, AccelNode, AccelNodeLabel, FusionCallOrigin, NodeId, ValueId, ValueOrigin,
};
use runmat_builtins::{MAX_CATALOG_ENTRY, MEAN_CATALOG_ENTRY};
use runmat_execution::Digest;
use runmat_value::Value;

pub(crate) fn is_canonical_mean_node(node: &AccelNode) -> bool {
    let AccelNodeLabel::Builtin {
        name,
        origin: FusionCallOrigin::Named { binding },
    } = &node.label
    else {
        return false;
    };
    let [variant] = MEAN_CATALOG_ENTRY.bindings else {
        return false;
    };
    name.eq_ignore_ascii_case(MEAN_CATALOG_ENTRY.identity.name)
        && *binding == MEAN_CATALOG_ENTRY.binding_identity(variant)
}

pub(crate) fn is_canonical_max_node(node: &AccelNode) -> bool {
    let AccelNodeLabel::Builtin {
        name,
        origin: FusionCallOrigin::Named { binding },
    } = &node.label
    else {
        return false;
    };
    name.eq_ignore_ascii_case(MAX_CATALOG_ENTRY.identity.name)
        && MAX_CATALOG_ENTRY
            .bindings
            .iter()
            .any(|variant| MAX_CATALOG_ENTRY.binding_identity(variant) == *binding)
}

fn has_live_binding(graph: &AccelGraph, id: ValueId) -> bool {
    graph.var_binding(id).is_some()
        || matches!(graph.value(id).map(|info| &info.origin),
            Some(ValueOrigin::NodeOutput { node, .. })
                if graph.node_binding(*node).is_some())
}

pub(crate) fn bound_folded_zero(graph: &AccelGraph, control: ValueId, before_pc: usize) -> bool {
    let Some(Value::Tensor(represented)) = graph.exact_folded_unary_scalar(control) else {
        return false;
    };
    let Some([zero]) = represented.as_f32_slice() else {
        return false;
    };
    let Some(ValueOrigin::NodeOutput {
        node: producer,
        output: 0,
    }) = graph.value(control).map(|info| &info.origin)
    else {
        return false;
    };
    let Some(source) = graph.node(*producer) else {
        return false;
    };
    zero.to_bits() == 0.0f32.to_bits()
        && source.span.end < before_pc
        && source
            .inputs
            .iter()
            .all(|input| graph.source_literal_constant(*input).is_some())
}

/// Inspect the represented fold only after proving its producer, source
/// literals, shape, type, and order. The bound value stays in plan inputs.
pub(crate) fn resolve_selected_mean_axes(
    graph: &AccelGraph,
    vid: ValueId,
    before_pc: usize,
) -> Option<Vec<f64>> {
    if let Some(axes) = resolve_numeric_vector_constant(graph, vid) {
        return Some(axes);
    }
    let ValueOrigin::NodeOutput { node, output: 0 } = graph.value(vid)?.origin else {
        return None;
    };
    let producer = graph.node(node)?;
    if !has_live_binding(graph, vid)
        || producer.span.end >= before_pc
        || producer
            .inputs
            .iter()
            .any(|input| graph.source_literal_constant(*input).is_none())
    {
        return None;
    }
    let Value::Tensor(tensor) = graph.exact_folded_literal_array(vid)? else {
        return None;
    };
    let [rows, columns] = tensor.shape() else {
        return None;
    };
    if tensor.is_empty() || (*rows != 1 && *columns != 1) {
        return None;
    }
    Some(tensor.as_f64_slice()?.to_vec())
}

pub(crate) fn collect_bound_mean_axes(
    graph: &AccelGraph,
    nodes: &[NodeId],
    routes: Option<&ImageReductionRoutes>,
) -> Option<Vec<(ValueId, Digest)>> {
    let mut source_bound_axes = nodes
        .iter()
        .filter_map(|id| graph.node(*id))
        .filter(|node| is_canonical_mean_node(node))
        .filter_map(|node| node.inputs.get(1).copied())
        .filter(|axis| has_live_binding(graph, *axis))
        .collect::<Vec<_>>();
    source_bound_axes.sort_unstable();
    source_bound_axes.dedup();
    if !source_bound_axes.is_empty() && routes.is_none() {
        return None;
    }
    let mut bound_axes_ids = Vec::new();
    if let Some(routes) = routes {
        routes.mu.for_each_axes_value(|id| bound_axes_ids.push(id));
        routes
            .variance
            .for_each_axes_value(|id| bound_axes_ids.push(id));
    }
    bound_axes_ids.sort_unstable();
    bound_axes_ids.dedup();
    let mut bound_mean_axes = Vec::new();
    for id in bound_axes_ids {
        if !has_live_binding(graph, id) {
            continue;
        }
        let represented = graph.exact_folded_literal_array(id)?;
        let digest = super::super::semantics::exact_constant_digest(represented).ok()?;
        bound_mean_axes.push((id, digest));
    }
    (source_bound_axes
        == bound_mean_axes
            .iter()
            .map(|(id, _)| *id)
            .collect::<Vec<_>>())
    .then_some(bound_mean_axes)
}

fn bound_single_zero(value: &Value) -> bool {
    matches!(value,
        Value::Tensor(tensor)
            if tensor.shape() == [1, 1]
                && tensor.as_f32_slice().is_some_and(|words|
                    matches!(words, [zero] if zero.to_bits() == 0.0f32.to_bits())))
}

/// Both image routes must check actual request carriers before provider work.
pub(crate) fn bound_image_controls_match<'a>(
    pattern: &ImageNormalizePattern,
    mut actual: impl FnMut(ValueId) -> Option<&'a Value>,
) -> bool {
    if let Some(id) = pattern.clamp_zero_value {
        if !pattern.clamp_zero || !actual(id).is_some_and(bound_single_zero) {
            return false;
        }
    }
    pattern.bound_mean_axes.iter().all(|(id, digest)| {
        actual(*id).is_some_and(|value| {
            super::super::semantics::exact_constant_digest(value)
                .ok()
                .as_ref()
                == Some(digest)
        })
    })
}
