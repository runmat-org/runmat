//! A single-output matmul executor cannot publish intermediate assignments or
//! discard branches. Keep those regions in their original bytecode execution.
use super::{AccelGraph, FusionGroup};

pub(super) fn preserves_observable_values(graph: &AccelGraph, group: &FusionGroup) -> bool {
    let Some(last) = group.nodes.last().and_then(|id| graph.node(*id)) else {
        return false;
    };
    if last.outputs.len() != 1 {
        return false;
    }
    for node in graph
        .nodes
        .iter()
        .filter(|node| node.span.start >= group.span.start && node.span.end <= group.span.end)
    {
        if !group.nodes.contains(&node.id) {
            return false;
        }
    }
    for (index, id) in group.nodes.iter().enumerate() {
        let Some(node) = graph.node(*id) else {
            return false;
        };
        if node.outputs.len() != 1 {
            return false;
        }
        if index + 1 == group.nodes.len() {
            continue;
        }
        let output = node.outputs[0];
        if graph.var_binding(output).is_some() {
            return false;
        }
        let next = group.nodes[index + 1];
        let mut consumers = graph
            .nodes
            .iter()
            .filter(|consumer| consumer.inputs.contains(&output));
        if consumers.next().map(|consumer| consumer.id) != Some(next) || consumers.next().is_some()
        {
            return false;
        }
    }
    true
}
