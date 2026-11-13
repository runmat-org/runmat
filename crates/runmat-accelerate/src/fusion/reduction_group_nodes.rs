//! Exact in-span reduction dependency closure. VM instruction admission owns
//! the span; this owner never invents a producer from an adjacent graph node.

use std::collections::HashSet;

use crate::graph::{
    AccelGraph, AccelGraphTag, AccelNode, AccelNodeLabel, AccelOpCategory, NodeId, PrimitiveOp,
    ValueId, ValueOrigin,
};
use crate::reduction_meta::detect_reduction_signature;
use runmat_value::Value;

use super::FusionGroup;

pub(super) fn checked(graph: &AccelGraph, group: &FusionGroup) -> Option<Vec<NodeId>> {
    let contained = graph
        .nodes
        .iter()
        .filter(|node| node.span.start >= group.span.start && node.span.end <= group.span.end)
        .collect::<Vec<_>>();
    let roots = contained
        .iter()
        .copied()
        .filter(|node| node.is_reduction())
        .collect::<Vec<_>>();
    let [root] = roots.as_slice() else {
        return None;
    };
    if root.span.end != group.span.end {
        return None;
    }
    // Preserve the existing singleton reduction route. A widened group must
    // prove its exact data dependency rather than using a guessed input.
    if contained.len() == 1 && group.span.start == group.span.end {
        return Some(vec![root.id]);
    }
    let signature = detect_reduction_signature(graph, root)?;
    let mut selected = HashSet::from([root.id]);
    let mut frontier = vec![signature.data_input];
    let mut visited_values = HashSet::new();
    while let Some(value_id) = frontier.pop() {
        if !visited_values.insert(value_id) {
            continue;
        }
        let value = graph.value(value_id)?;
        let ValueOrigin::NodeOutput { node: node_id, .. } = &value.origin else {
            continue;
        };
        if *node_id == root.id {
            return None;
        }
        let node = graph.node(*node_id)?;
        if node.span.start < group.span.start
            || node.span.end > group.span.end
            || !matches!(
                node.category,
                AccelOpCategory::Elementwise | AccelOpCategory::Transpose
            )
            || node.outputs.as_slice() != std::slice::from_ref(&value_id)
            || graph.var_binding(value_id).is_some()
            || graph.node_binding(*node_id).is_some()
        {
            // An outside producer is an ordinary materialized data leaf. An
            // inside producer with observable binding is never elidable.
            if node.span.end < group.span.start {
                continue;
            }
            return None;
        }
        if selected.insert(*node_id) {
            frontier.extend(node.inputs.iter().copied());
        }
    }
    if selected.len() <= 1
        || contained.iter().any(|node| {
            !selected.contains(&node.id)
                && !unbound_empty_reduction_control(graph, root, signature.data_input, node)
        })
    {
        return None;
    }
    for node_id in selected.iter().copied().filter(|id| *id != root.id) {
        let node = graph.node(node_id)?;
        for &output in &node.outputs {
            let consumers = graph
                .nodes
                .iter()
                .filter(|consumer| consumer.inputs.contains(&output))
                .collect::<Vec<_>>();
            if consumers.is_empty()
                || consumers
                    .iter()
                    .any(|consumer| !selected.contains(&consumer.id))
            {
                return None;
            }
        }
    }
    let mut ordered = selected.into_iter().collect::<Vec<_>>();
    ordered.sort_unstable_by_key(|id| {
        graph
            .node(*id)
            .map(|node| (node.span.start, node.span.end, node.id))
            .unwrap_or((usize::MAX, usize::MAX, *id))
    });
    Some(ordered)
}

/// A control literal may occupy the selected instruction span without being
/// an executable shader producer. Only the VM's sealed private-temporary
/// projection can remove its store binding; a published or escaped literal
/// remains a hard barrier.
fn unbound_empty_reduction_control(
    graph: &AccelGraph,
    root: &AccelNode,
    data_input: ValueId,
    node: &AccelNode,
) -> bool {
    let [output] = node.outputs.as_slice() else {
        return false;
    };
    node.category == AccelOpCategory::Other
        && matches!(&node.label, AccelNodeLabel::Primitive(PrimitiveOp::UPlus))
        && node.tags.as_slice() == [AccelGraphTag::FoldedLiteralArray]
        && node.inputs.is_empty()
        && *output != data_input
        && root.inputs.contains(output)
        && graph.var_binding(*output).is_none()
        && graph.node_binding(node.id).is_none()
        && graph.value(*output).is_some_and(|value| {
            matches!(&value.origin, ValueOrigin::NodeOutput { node: producer, output: 0 }
                if *producer == node.id)
                && matches!(&value.constant, Some(Value::Tensor(tensor))
                    if tensor.shape() == [0, 0] && tensor.is_empty())
        })
        && graph
            .nodes
            .iter()
            .filter(|consumer| consumer.inputs.contains(output))
            .all(|consumer| consumer.id == root.id)
}
