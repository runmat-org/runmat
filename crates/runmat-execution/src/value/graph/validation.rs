use std::collections::HashMap;

use super::*;

mod depth;
mod order;
mod structure;
use depth::graph_depths;
use order::value_construction_order;

pub(super) fn expanded_numeric_allocation_bytes(
    graph: &PayloadGraph,
) -> Result<u64, ContractError> {
    order::expanded_numeric_allocation_bytes(graph)
}
use structure::{
    activation_chain_owns_from, shape_count, sorted_fields, validate_activation,
    validate_parent_chains, validate_value_reference_roles,
};

fn invalid(reason: &'static str) -> ContractError {
    ContractError::invalid("payload graph", reason)
}

pub(super) fn validate(
    graph: &PayloadGraph,
    limits: ValueLimits,
    source_map: Option<&ExecutableSourceMapV2>,
    layouts: &HashMap<GraphBoundOwner, GraphLexicalLayout>,
    selected_unit_count: usize,
) -> Result<Vec<GraphNodeRef>, ContractError> {
    graph.validate_selected_unit_references(selected_unit_count)?;
    let admitted = |owner: GraphBoundOwner| (owner.unit.0 as usize) < selected_unit_count;
    if layouts.keys().any(|owner| !admitted(*owner)) {
        return Err(invalid(
            "lexical layout selects a unit outside the admitted bundle",
        ));
    }
    if graph.nodes.len() as u64 > limits.max_nodes {
        return Err(ContractError::Limit {
            field: "value nodes",
            limit: limits.max_nodes,
        });
    }
    if graph.roots.is_empty() && !graph.nodes.is_empty() {
        return Err(invalid("a graph with nodes must have a root"));
    }
    let mut edges = vec![Vec::new(); graph.nodes.len()];
    let mut structural_nodes = 0u64;
    // Root ordinals are encoded references too, including repeated aliases.
    let mut bytes = (graph.roots.len() as u64).saturating_mul(4);
    let mut elements = graph.roots.len() as u64;
    let mut text = 0u64;
    let mut leaves = Vec::new();
    let mut metadata_seen = vec![false; graph.nodes.len()];
    for (index, node) in graph.nodes.iter().enumerate() {
        let outgoing = &mut edges[index];
        match node {
            GraphNode::Leaf { value } => {
                if !is_leaf(value) {
                    return Err(invalid(
                        "recursive or durable value is hidden in a graph leaf",
                    ));
                }
                leaves.push((index, value));
            }
            GraphNode::Cell { shape, values } => {
                structural_nodes += 1;
                let count = shape_count(shape)?;
                if count != values.len() as u64 {
                    return Err(invalid("cell shape does not match its values"));
                }
                elements = elements.saturating_add(count);
                bytes = bytes.saturating_add((shape.len() as u64).saturating_mul(8));
                outgoing.extend(values.iter().copied());
            }
            GraphNode::Struct { fields } => {
                structural_nodes += 1;
                sorted_fields(fields, limits)?;
                elements = elements.saturating_add(fields.len() as u64);
                text = text.saturating_add(
                    fields
                        .iter()
                        .map(|field| field.name.len() as u64)
                        .sum::<u64>(),
                );
                outgoing.extend(fields.iter().map(|field| field.value));
            }
            GraphNode::Registered(registered) => {
                structural_nodes += 1;
                super::super::RegisteredData::validate_identity_parts(
                    &registered.type_identity,
                    registered.schema_version,
                )?;
                sorted_fields(&registered.fields, limits)?;
                elements = elements.saturating_add(registered.fields.len() as u64);
                text = text
                    .saturating_add(registered.type_identity.len() as u64)
                    .saturating_add(
                        registered
                            .fields
                            .iter()
                            .map(|field| field.name.len() as u64)
                            .sum::<u64>(),
                    );
                outgoing.extend(registered.fields.iter().map(|field| field.value));
            }
            GraphNode::SelectedObject(object) => {
                structural_nodes += 1;
                if !admitted(object.constructor) || !layouts.contains_key(&object.constructor) {
                    return Err(invalid(
                        "selected object constructor is outside the admitted bundle",
                    ));
                }
                sorted_fields(&object.fields, limits)?;
                if object.dynamic_properties.len() > limits.max_fields as usize
                    || object
                        .dynamic_properties
                        .windows(2)
                        .any(|pair| pair[0].name >= pair[1].name)
                {
                    return Err(invalid(
                        "selected object dynamic properties must be unique and sorted",
                    ));
                }
                if object.dynamic_metadata.len() > object.dynamic_properties.len()
                    || (object.kind != GraphSelectedObjectKind::Handle
                        && !object.dynamic_metadata.is_empty())
                {
                    return Err(invalid(
                        "selected object metadata exceeds its dynamic properties",
                    ));
                }
                let mut previous_metadata_name = None;
                for reference in &object.dynamic_metadata {
                    let position = reference.0 as usize;
                    let Some(GraphNode::SelectedDynamicPropertyMetadata(metadata)) =
                        graph.nodes.get(position)
                    else {
                        return Err(invalid("selected object metadata has the wrong node kind"));
                    };
                    if metadata.receiver.0 as usize != index
                        || previous_metadata_name
                            .is_some_and(|name: &str| name >= metadata.property.as_str())
                        || object
                            .dynamic_properties
                            .binary_search_by(|property| {
                                property.name.as_str().cmp(metadata.property.as_str())
                            })
                            .is_err()
                        || std::mem::replace(&mut metadata_seen[position], true)
                    {
                        return Err(invalid(
                            "selected object metadata differs from its property",
                        ));
                    }
                    previous_metadata_name = Some(metadata.property.as_str());
                }
                elements = elements
                    .saturating_add(object.fields.len() as u64)
                    .saturating_add(object.dynamic_properties.len() as u64)
                    .saturating_add(object.dynamic_metadata.len() as u64);
                bytes = bytes.saturating_add(40);
                text = text
                    .saturating_add(object.class.to_string().len() as u64)
                    .saturating_add(
                        object
                            .fields
                            .iter()
                            .map(|field| field.name.len() as u64)
                            .sum::<u64>(),
                    )
                    .saturating_add(object.dynamic_properties.iter().fold(
                        0u64,
                        |total, property| {
                            total
                                .saturating_add(property.name.len() as u64)
                                .saturating_add(property.definition_name.len() as u64)
                                .saturating_add(property.defining_class.to_string().len() as u64)
                                .saturating_add(property.description.len() as u64)
                        },
                    ));
                outgoing.extend(object.fields.iter().map(|field| field.value));
                outgoing.extend(object.dynamic_metadata.iter().copied());
            }
            GraphNode::SelectedObjectArray(array) => {
                structural_nodes += 1;
                if array.shape.len() < 2 || shape_count(&array.shape)? != array.values.len() as u64
                {
                    return Err(invalid(
                        "selected object array shape differs from its values",
                    ));
                }
                if array.values.is_empty() {
                    if array.empty_provenance.is_empty() {
                        return Err(invalid(
                            "empty selected object array has no exact provenance",
                        ));
                    }
                } else if !array.empty_provenance.is_empty() || array.includes_unselected {
                    return Err(invalid(
                        "nonempty selected array carries redundant provenance",
                    ));
                }
                if u64::try_from(array.empty_provenance.len())
                    .map_err(|_| invalid("selected array provenance length overflows"))?
                    > limits.max_nodes
                {
                    return Err(invalid("selected array provenance exceeds graph nodes"));
                }
                for reference in &array.values {
                    let Some(GraphNode::SelectedObject(element)) =
                        graph.nodes.get(reference.0 as usize)
                    else {
                        return Err(invalid("selected array element is not an admitted object"));
                    };
                    if element.class != array.class || element.kind != array.kind {
                        return Err(invalid("selected array elements differ in class or kind"));
                    }
                }
                let mut owners = std::collections::HashSet::new();
                for reference in &array.empty_provenance {
                    let Some(GraphNode::Callable(anchor)) = graph.nodes.get(reference.0 as usize)
                    else {
                        return Err(invalid("selected array provenance is not a bound anchor"));
                    };
                    let GraphCallableOwner::Bound(owner) = anchor.owner else {
                        return Err(invalid("selected array provenance has no bound owner"));
                    };
                    if anchor.form != CallableForm::Handle
                        || !anchor.captures.is_empty()
                        || anchor.activation.is_some()
                        || !owners.insert(owner)
                    {
                        return Err(invalid("selected array provenance has an invalid anchor"));
                    }
                }
                elements = elements
                    .saturating_add(array.values.len() as u64)
                    .saturating_add(array.empty_provenance.len() as u64);
                bytes = bytes.saturating_add((array.shape.len() as u64).saturating_mul(8));
                text = text.saturating_add(array.class.to_string().len() as u64);
                outgoing.extend(array.values.iter().copied());
                outgoing.extend(array.empty_provenance.iter().copied());
            }
            GraphNode::SelectedDynamicPropertyMetadata(metadata) => {
                structural_nodes += 1;
                if metadata.property.is_empty()
                    || metadata.property.len() > limits.max_text_bytes as usize
                {
                    return Err(invalid("selected metadata property name is invalid"));
                }
                let Some(GraphNode::SelectedObject(receiver)) =
                    graph.nodes.get(metadata.receiver.0 as usize)
                else {
                    return Err(invalid("selected metadata receiver is absent"));
                };
                if receiver.kind != GraphSelectedObjectKind::Handle {
                    return Err(invalid("selected metadata receiver is not a handle"));
                }
                elements = elements.saturating_add(1);
                bytes = bytes.saturating_add(8);
                text = text.saturating_add(metadata.property.len() as u64);
                outgoing.push(metadata.receiver);
            }
            GraphNode::Callable(callable) => {
                structural_nodes += 1;
                super::super::CallableValue::validate_identity_parts(
                    callable.owner.portable_owner(),
                    &callable.qualified_name,
                    callable.callable_digest,
                    callable.form,
                    callable.source_role,
                    callable.source_file.as_deref(),
                    callable.captures.len(),
                )?;
                if callable.form == CallableForm::Handle && callable.activation.is_some() {
                    return Err(invalid("plain handle carries closure state"));
                }
                if let GraphCallableOwner::Bound(origin) = callable.owner {
                    if !admitted(origin) {
                        return Err(invalid(
                            "bound callable selects a unit outside the admitted bundle",
                        ));
                    }
                    if !layouts.contains_key(&origin) {
                        return Err(invalid("bound callable has no admitted lexical layout"));
                    }
                    if callable.activation.is_some() != layouts[&origin].activation_parent.is_some()
                    {
                        return Err(invalid(
                            "bound callable activation disagrees with admitted layout",
                        ));
                    }
                    if callable.activation.is_none()
                        && layouts[&origin]
                            .bindings
                            .iter()
                            .any(|(_, kind)| *kind != GraphBindingKind::Owned)
                    {
                        return Err(invalid("bound callable has no inherited activation"));
                    }
                } else if callable.activation.is_some() {
                    return Err(invalid("unbound callable carries an activation"));
                }
                if let Some(reference) = callable.activation {
                    let Some(GraphNode::Activation(selected_activation)) =
                        graph.nodes.get(reference.0 as usize)
                    else {
                        return Err(invalid(
                            "callable activation reference is not an activation",
                        ));
                    };
                    let GraphCallableOwner::Bound(origin) = callable.owner else {
                        return Err(invalid("callable activation has no bound owner"));
                    };
                    let layout = &layouts[&origin];
                    if layout.activation_parent != Some(selected_activation.owner) {
                        return Err(invalid(
                            "callable activation differs from admitted lexical parent",
                        ));
                    }
                }
                elements = elements.saturating_add(callable.captures.len() as u64);
                bytes = bytes.saturating_add(
                    if matches!(callable.owner, GraphCallableOwner::Bound(_)) {
                        39
                    } else {
                        1
                    },
                );
                text = text
                    .saturating_add(callable.qualified_name.len() as u64)
                    .saturating_add(
                        callable
                            .source_file
                            .as_ref()
                            .map_or(0, |path| path.len() as u64),
                    );
                outgoing.extend(callable.captures.iter().copied());
                if let Some(activation) = callable.activation {
                    outgoing.push(activation);
                }
            }
            GraphNode::Activation(activation) => {
                structural_nodes += 1;
                if !admitted(activation.owner) {
                    return Err(invalid(
                        "activation selects a unit outside the admitted bundle",
                    ));
                }
                validate_activation(graph, activation, layouts)?;
                elements = elements.saturating_add(activation.bindings.len() as u64);
                bytes = bytes
                    .saturating_add(36)
                    .saturating_add((activation.bindings.len() as u64).saturating_mul(8));
                if let Some(parent) = activation.parent {
                    outgoing.push(parent);
                }
                outgoing.extend(
                    activation
                        .bindings
                        .iter()
                        .filter_map(|binding| binding.value),
                );
            }
        }
        bytes = bytes.saturating_add((outgoing.len() as u64).saturating_mul(4));
    }
    for (index, node) in graph.nodes.iter().enumerate() {
        if matches!(node, GraphNode::SelectedDynamicPropertyMetadata(_)) && !metadata_seen[index] {
            return Err(invalid(
                "selected metadata handle is not owned by its receiver",
            ));
        }
    }
    for outgoing in &edges {
        for edge in outgoing {
            if edge.0 as usize >= graph.nodes.len() {
                return Err(invalid("node reference is outside the payload"));
            }
        }
    }
    for root in &graph.roots {
        if root.0 as usize >= graph.nodes.len() {
            return Err(invalid("root reference is outside the payload"));
        }
    }
    validate_value_reference_roles(graph)?;
    let value_order = value_construction_order(graph)?;
    validate_parent_chains(graph, limits.max_depth)?;
    for node in &graph.nodes {
        match node {
            GraphNode::Activation(activation) => {
                let layout = &layouts[&activation.owner];
                for (binding, kind) in &layout.bindings {
                    if *kind != GraphBindingKind::Owned
                        && !activation_chain_owns_from(graph, activation.parent, *binding)?
                    {
                        return Err(invalid("inherited lexical cell has no admitted ancestor"));
                    }
                }
            }
            GraphNode::Callable(callable) => {
                if let (GraphCallableOwner::Bound(origin), Some(first)) =
                    (callable.owner, callable.activation)
                {
                    for (binding, kind) in &layouts[&origin].bindings {
                        if *kind != GraphBindingKind::Owned
                            && !activation_chain_owns_from(graph, Some(first), *binding)?
                        {
                            return Err(invalid("callable activation lacks an inherited cell"));
                        }
                    }
                }
            }
            _ => {}
        }
    }
    let mut budget = super::super::ValueLimitBudget::new(limits);
    budget.nodes(structural_nodes)?;
    budget.bytes(bytes)?;
    budget.elements(elements)?;
    budget.text(text)?;
    let depths = graph_depths(graph, &edges, &graph.roots, limits.max_depth)?;
    let leaf_refs = leaves
        .iter()
        .map(|(index, value)| (*value, depths[*index]))
        .collect::<Vec<_>>();
    let remaining = budget.remaining();
    super::super::validation::validate_graph_leaves(&leaf_refs, remaining, source_map)?;
    Ok(value_order)
}
