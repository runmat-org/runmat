use super::*;

/// Activation nodes are execution owners, not Values. Only the two explicitly
/// typed activation links (callable activation and activation parent) may
/// reference them; every root and value-bearing edge must decode as a Value.
pub(super) fn validate_value_reference_roles(graph: &PayloadGraph) -> Result<(), ContractError> {
    fn value_target(graph: &PayloadGraph, reference: GraphNodeRef) -> Result<(), ContractError> {
        match graph.nodes.get(reference.0 as usize) {
            Some(GraphNode::Activation(_)) => {
                Err(invalid("value reference points to an activation"))
            }
            Some(_) => Ok(()),
            None => Err(invalid("node reference is outside the payload")),
        }
    }

    for root in &graph.roots {
        value_target(graph, *root)?;
    }
    for node in &graph.nodes {
        match node {
            GraphNode::Cell { values, .. } => {
                for value in values {
                    value_target(graph, *value)?;
                }
            }
            GraphNode::Struct { fields } => {
                for field in fields {
                    value_target(graph, field.value)?;
                }
            }
            GraphNode::Registered(registered) => {
                for field in &registered.fields {
                    value_target(graph, field.value)?;
                }
            }
            GraphNode::SelectedObject(object) => {
                for field in &object.fields {
                    value_target(graph, field.value)?;
                }
                for metadata in &object.dynamic_metadata {
                    value_target(graph, *metadata)?;
                }
            }
            GraphNode::SelectedObjectArray(array) => {
                for value in array.values.iter().chain(&array.empty_provenance) {
                    value_target(graph, *value)?;
                }
            }
            GraphNode::SelectedDynamicPropertyMetadata(metadata) => {
                value_target(graph, metadata.receiver)?;
            }
            GraphNode::Callable(callable) => {
                for capture in &callable.captures {
                    value_target(graph, *capture)?;
                }
            }
            GraphNode::Activation(activation) => {
                for binding in &activation.bindings {
                    if let Some(value) = binding.value {
                        value_target(graph, value)?;
                    }
                }
            }
            GraphNode::Leaf { .. } => {}
        }
    }
    Ok(())
}

pub(super) fn activation_chain_owns_from(
    graph: &PayloadGraph,
    first: Option<GraphNodeRef>,
    binding: BindingId,
) -> Result<bool, ContractError> {
    let mut current = first;
    while let Some(reference) = current {
        let Some(GraphNode::Activation(activation)) = graph.nodes.get(reference.0 as usize) else {
            return Err(invalid("activation parent is absent or not an activation"));
        };
        if activation
            .bindings
            .iter()
            .any(|cell| cell.binding == binding)
        {
            return Ok(true);
        }
        current = activation.parent;
    }
    Ok(false)
}

pub(super) fn validate_parent_chains(
    graph: &PayloadGraph,
    max_depth: u16,
) -> Result<(), ContractError> {
    let mut state = vec![0u8; graph.nodes.len()];
    let mut depth = vec![0usize; graph.nodes.len()];
    for start in 0..graph.nodes.len() {
        if !matches!(&graph.nodes[start], GraphNode::Activation(_)) || state[start] == 2 {
            continue;
        }
        let mut current = Some(start);
        let mut path = Vec::new();
        let mut base = 0usize;
        while let Some(index) = current {
            match state[index] {
                1 => return Err(invalid("activation parent chain is cyclic")),
                2 => {
                    base = depth[index];
                    break;
                }
                _ => {}
            }
            state[index] = 1;
            path.push(index);
            let GraphNode::Activation(activation) = &graph.nodes[index] else {
                return Err(invalid("activation parent is not an activation"));
            };
            current = activation.parent.map(|reference| reference.0 as usize);
        }
        for index in path.into_iter().rev() {
            base += 1;
            if base - 1 > max_depth as usize {
                return Err(ContractError::Limit {
                    field: "value depth",
                    limit: max_depth.into(),
                });
            }
            depth[index] = base;
            state[index] = 2;
        }
    }
    Ok(())
}

pub(super) fn shape_count(shape: &[u64]) -> Result<u64, ContractError> {
    shape.iter().try_fold(1u64, |count, extent| {
        count
            .checked_mul(*extent)
            .ok_or_else(|| invalid("shape element count overflows"))
    })
}

pub(super) fn sorted_fields(
    fields: &[GraphField],
    limits: ValueLimits,
) -> Result<(), ContractError> {
    if fields.len() > limits.max_fields as usize {
        return Err(ContractError::Limit {
            field: "struct fields",
            limit: limits.max_fields.into(),
        });
    }
    if fields.windows(2).any(|pair| pair[0].name >= pair[1].name) {
        return Err(invalid("field names must be unique and sorted"));
    }
    Ok(())
}

pub(super) fn validate_activation(
    graph: &PayloadGraph,
    activation: &GraphActivation,
    layouts: &HashMap<GraphBoundOwner, GraphLexicalLayout>,
) -> Result<(), ContractError> {
    let layout = layouts
        .get(&activation.owner)
        .ok_or_else(|| invalid("activation owner is not admitted"))?;
    let actual_parent =
        activation
            .parent
            .and_then(|reference| match graph.nodes.get(reference.0 as usize) {
                Some(GraphNode::Activation(parent)) => Some(parent.owner),
                _ => None,
            });
    if actual_parent != layout.activation_parent {
        return Err(invalid(
            "activation parent differs from admitted lexical ancestry",
        ));
    }
    if actual_parent.is_some_and(|parent| parent.origin.program != activation.owner.origin.program)
    {
        return Err(invalid("activation parent comes from a different program"));
    }
    if layout
        .bindings
        .windows(2)
        .any(|pair| pair[0].0 >= pair[1].0)
        || activation
            .bindings
            .windows(2)
            .any(|pair| pair[0].binding >= pair[1].binding)
    {
        return Err(invalid("lexical bindings must be unique and sorted"));
    }
    let expected_owned = layout
        .bindings
        .iter()
        .filter(|(_, kind)| *kind == GraphBindingKind::Owned)
        .map(|(binding, _)| *binding)
        .collect::<Vec<_>>();
    if activation
        .bindings
        .iter()
        .map(|entry| entry.binding)
        .collect::<Vec<_>>()
        != expected_owned
    {
        return Err(invalid(
            "activation cells differ from the admitted owned layout",
        ));
    }
    Ok(())
}
