use super::*;

fn value_child(node: &GraphNode, index: usize) -> Option<GraphNodeRef> {
    match node {
        GraphNode::Leaf { .. }
        | GraphNode::Activation(_)
        | GraphNode::SelectedDynamicPropertyMetadata(_) => None,
        GraphNode::Cell { values, .. } => values.get(index).copied(),
        GraphNode::Struct { fields } => fields.get(index).map(|field| field.value),
        GraphNode::Registered(registered) => registered.fields.get(index).map(|field| field.value),
        GraphNode::SelectedObject(object) => object.fields.get(index).map(|field| field.value),
        GraphNode::SelectedObjectArray(array) => {
            if index < array.values.len() {
                array.values.get(index).copied()
            } else {
                array
                    .empty_provenance
                    .get(index - array.values.len())
                    .copied()
            }
        }
        GraphNode::Callable(callable) => callable.captures.get(index).copied(),
    }
}

/// A Value is built from its child Values, while activation cells are filled
/// only after rooted activation shells and Values exist. Cutting just those
/// typed activation edges admits Closure↔Activation cycles, but never an
/// impossible by-value cycle hidden inside their larger SCC.
pub(super) fn value_construction_order(
    graph: &PayloadGraph,
) -> Result<Vec<GraphNodeRef>, ContractError> {
    let mut state = vec![0u8; graph.nodes.len()];
    let mut order = Vec::with_capacity(graph.nodes.len());
    for start in 0..graph.nodes.len() {
        if matches!(graph.nodes[start], GraphNode::Activation(_)) || state[start] != 0 {
            continue;
        }
        state[start] = 1;
        let mut stack = vec![(start, 0usize)];
        while let Some((index, next_child)) = stack.last_mut() {
            let Some(child) = value_child(&graph.nodes[*index], *next_child) else {
                let (finished, _) = stack.pop().expect("nonempty traversal stack");
                state[finished] = 2;
                order.push(GraphNodeRef(u32::try_from(finished).map_err(|_| {
                    invalid("value node ordinal exceeds wire capacity")
                })?));
                continue;
            };
            *next_child += 1;
            let child = child.0 as usize;
            if child >= graph.nodes.len() {
                return Err(invalid("node reference is outside the payload"));
            }
            if matches!(graph.nodes[child], GraphNode::Activation(_)) {
                return Err(invalid("value reference points to an activation"));
            }
            if matches!(
                graph.nodes[child],
                GraphNode::SelectedObject(GraphSelectedObject {
                    kind: GraphSelectedObjectKind::Handle,
                    ..
                })
            ) {
                // The decoder roots every selected handle shell before it
                // builds any by-value field. This is the only object edge we
                // may cut; a value-class cycle still has no construction.
                continue;
            }
            match state[child] {
                0 => {
                    state[child] = 1;
                    stack.push((child, 0));
                }
                1 => return Err(invalid("value-only cycle cannot be constructed")),
                _ => {}
            }
        }
    }
    Ok(order)
}

/// Count every by-value occurrence, not merely every distinct wire node. A
/// small DAG can otherwise understate the memory needed to clone a large
/// numeric subtree into many cells, structs, or callable captures.
pub(super) fn expanded_numeric_allocation_bytes(
    graph: &PayloadGraph,
) -> Result<u64, ContractError> {
    let order = value_construction_order(graph)?;
    let mut occurrences = vec![0u64; graph.nodes.len()];
    for root in &graph.roots {
        let Some(node) = graph.nodes.get(root.0 as usize) else {
            return Err(invalid("graph root is outside the payload"));
        };
        if matches!(node, GraphNode::Activation(_)) {
            return Err(invalid("graph root is an activation"));
        }
        occurrences[root.0 as usize] = occurrences[root.0 as usize]
            .checked_add(1)
            .ok_or_else(|| invalid("graph value fanout overflows"))?;
    }
    for node in &graph.nodes {
        if let GraphNode::Activation(activation) = node {
            for binding in &activation.bindings {
                let Some(reference) = binding.value else {
                    continue;
                };
                let Some(value) = graph.nodes.get(reference.0 as usize) else {
                    return Err(invalid("activation value is outside the payload"));
                };
                if matches!(value, GraphNode::Activation(_)) {
                    return Err(invalid("activation value points to an activation"));
                }
                occurrences[reference.0 as usize] = occurrences[reference.0 as usize]
                    .checked_add(1)
                    .ok_or_else(|| invalid("graph value fanout overflows"))?;
            }
        }
    }
    for reference in order.iter().rev() {
        let index = reference.0 as usize;
        let count = if matches!(
            graph.nodes[index],
            GraphNode::SelectedObject(GraphSelectedObject {
                kind: GraphSelectedObjectKind::Handle,
                ..
            })
        ) {
            1
        } else {
            occurrences[index]
        };
        let mut next = 0;
        while let Some(child) = value_child(&graph.nodes[index], next) {
            next += 1;
            let child_index = child.0 as usize;
            if matches!(
                graph.nodes[child_index],
                GraphNode::SelectedObject(GraphSelectedObject {
                    kind: GraphSelectedObjectKind::Handle,
                    ..
                })
            ) {
                continue;
            }
            occurrences[child_index] = occurrences[child_index]
                .checked_add(count)
                .ok_or_else(|| invalid("graph value fanout overflows"))?;
        }
    }
    let mut total = 0u64;
    for (index, node) in graph.nodes.iter().enumerate() {
        let bytes = match node {
            GraphNode::Leaf {
                value: ValuePayload::Inline(value),
            } => match value.as_ref() {
                InlineValue::Dense(value) => u64::try_from(value.little_endian_data.len())
                    .map_err(|_| invalid("graph dense allocation overflows"))?,
                InlineValue::Sparse(value) => value
                    .column_offsets
                    .len()
                    .checked_add(value.row_indices.len())
                    .and_then(|indices| indices.checked_mul(8))
                    .and_then(|indices| indices.checked_add(value.little_endian_data.len()))
                    .and_then(|bytes| u64::try_from(bytes).ok())
                    .ok_or_else(|| invalid("graph sparse allocation overflows"))?,
                _ => 0,
            },
            _ => 0,
        };
        total = total
            .checked_add(
                bytes
                    .checked_mul(occurrences[index])
                    .ok_or_else(|| invalid("graph numeric fanout allocation overflows"))?,
            )
            .ok_or_else(|| invalid("graph numeric fanout allocation overflows"))?;
    }
    Ok(total)
}
