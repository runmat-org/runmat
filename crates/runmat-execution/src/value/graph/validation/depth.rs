use std::collections::VecDeque;

use super::*;

/// SCC condensation makes a cycle one bounded graph level. A back edge does
/// not turn a valid self-referential closure into infinite structural depth.
pub(super) fn graph_depths(
    graph: &PayloadGraph,
    edges: &[Vec<GraphNodeRef>],
    roots: &[GraphNodeRef],
    max_depth: u16,
) -> Result<Vec<u16>, ContractError> {
    let count = edges.len();
    let mut seen = vec![false; count];
    let mut order = Vec::with_capacity(count);
    for root in roots {
        let start = root.0 as usize;
        if seen[start] {
            continue;
        }
        seen[start] = true;
        let mut stack = vec![(start, 0usize)];
        while let Some((node, next)) = stack.last_mut() {
            if let Some(edge) = edges[*node].get(*next) {
                *next += 1;
                let child = edge.0 as usize;
                if !seen[child] {
                    seen[child] = true;
                    stack.push((child, 0));
                }
            } else {
                order.push(*node);
                stack.pop();
            }
        }
    }
    if seen.iter().any(|seen| !seen) {
        return Err(invalid("payload contains an unreachable node"));
    }
    let mut reverse = vec![Vec::new(); count];
    for (node, outgoing) in edges.iter().enumerate() {
        for edge in outgoing {
            reverse[edge.0 as usize].push(node);
        }
    }
    let mut component = vec![usize::MAX; count];
    let mut components = 0usize;
    for node in order.into_iter().rev() {
        if component[node] != usize::MAX {
            continue;
        }
        component[node] = components;
        let mut stack = vec![node];
        while let Some(current) = stack.pop() {
            for &parent in &reverse[current] {
                if component[parent] == usize::MAX {
                    component[parent] = components;
                    stack.push(parent);
                }
            }
        }
        components += 1;
    }
    let mut downstream = vec![Vec::new(); components];
    let mut member_count = vec![0usize; components];
    let mut contains_anchor = vec![false; components];
    for (node, &part) in component.iter().enumerate() {
        member_count[part] += 1;
        contains_anchor[part] |= matches!(
            &graph.nodes[node],
            GraphNode::Activation(_)
                | GraphNode::SelectedDynamicPropertyMetadata(_)
                | GraphNode::SelectedObject(GraphSelectedObject {
                    kind: GraphSelectedObjectKind::Handle,
                    ..
                })
        );
    }
    for (node, outgoing) in edges.iter().enumerate() {
        if !contains_anchor[component[node]]
            && (member_count[component[node]] > 1
                || outgoing.iter().any(|edge| edge.0 as usize == node))
        {
            return Err(invalid(
                "cycle has no constructible activation or handle owner",
            ));
        }
    }
    let mut indegree = vec![0usize; components];
    for (node, outgoing) in edges.iter().enumerate() {
        for edge in outgoing {
            let from = component[node];
            let to = component[edge.0 as usize];
            if from != to {
                downstream[from].push(to);
                indegree[to] += 1;
            }
        }
    }
    let mut queue = VecDeque::new();
    let mut depth = vec![0usize; components];
    for (index, degree) in indegree.iter().enumerate() {
        if *degree == 0 {
            queue.push_back(index);
        }
    }
    while let Some(current) = queue.pop_front() {
        for &child in &downstream[current] {
            depth[child] = depth[child].max(depth[current] + 1);
            indegree[child] -= 1;
            if indegree[child] == 0 {
                queue.push_back(child);
            }
        }
    }
    if depth.iter().any(|depth| *depth > max_depth as usize) {
        return Err(ContractError::Limit {
            field: "value depth",
            limit: max_depth.into(),
        });
    }
    Ok(component
        .into_iter()
        .map(|index| depth[index] as u16)
        .collect())
}
