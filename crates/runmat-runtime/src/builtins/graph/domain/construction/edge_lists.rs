use super::super::parsing::{
    is_numeric_weight_arg, numeric_vector, parse_string_vector, weights_from_value,
};
use super::*;
use std::collections::HashMap;

pub(super) fn graph_from_edges(
    name: &'static str,
    directed: bool,
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<Value> {
    if args.len() < 2 {
        return Err(graph_error(
            name,
            format!("{name}: expected source and target node lists"),
        ));
    }
    let source_nodes = node_tokens(&args[0], name)?;
    let target_nodes = node_tokens(&args[1], name)?;
    if source_nodes.len() != target_nodes.len() {
        return Err(graph_error(
            name,
            format!("{name}: source and target lists must have the same length"),
        ));
    }
    let weights = match args.get(2) {
        Some(value) if is_numeric_weight_arg(value) => {
            weights_from_value(value, source_nodes.len(), name)?
        }
        _ => vec![1.0; source_nodes.len()],
    };
    let names_arg = if args.len() >= 4 {
        if !args.get(2).is_some_and(is_numeric_weight_arg) {
            return Err(graph_error(
                name,
                format!(
                    "{name}: the third argument must contain edge weights when names are fourth"
                ),
            ));
        }
        args.get(3)
    } else if args.len() == 3 && !is_numeric_weight_arg(&args[2]) {
        args.get(2)
    } else {
        None
    };

    let mut node_names = names_arg
        .map(|value| parse_string_vector(value))
        .transpose()
        .map_err(|err| graph_error(name, format!("{name}: {err}")))?;
    let mut node_count = node_names.as_ref().map_or(0, Vec::len);
    let mut label_to_index = HashMap::<String, usize>::new();
    if let Some(names) = &node_names {
        for (idx, node_name) in names.iter().enumerate() {
            label_to_index.insert(node_name.clone(), idx);
        }
    }

    let mut edges = Vec::with_capacity(source_nodes.len());
    for ((source, target), weight) in source_nodes.into_iter().zip(target_nodes).zip(weights) {
        let source = resolve_node_token(
            source,
            &mut node_names,
            &mut label_to_index,
            &mut node_count,
            name,
        )?;
        let target = resolve_node_token(
            target,
            &mut node_names,
            &mut label_to_index,
            &mut node_count,
            name,
        )?;
        edges.push(Edge {
            source,
            target,
            weight,
        });
    }

    graph_object(GraphData {
        directed,
        node_names,
        edges,
        node_count,
    })
}

pub(in crate::builtins::graph::domain) fn node_tokens(
    value: &Value,
    name: &'static str,
) -> BuiltinResult<Vec<NodeToken>> {
    if matches!(
        value,
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Tensor(_) | Value::LogicalArray(_)
    ) {
        return Ok(numeric_vector(value, name)?
            .into_iter()
            .map(NodeToken::Index)
            .collect());
    }
    Ok(parse_string_vector(value)
        .map_err(|err| graph_error(name, format!("{name}: {err}")))?
        .into_iter()
        .map(NodeToken::Name)
        .collect())
}

#[derive(Debug)]
pub(in crate::builtins::graph::domain) enum NodeToken {
    Index(usize),
    Name(String),
}

pub(super) fn resolve_node_token(
    token: NodeToken,
    node_names: &mut Option<Vec<String>>,
    label_to_index: &mut HashMap<String, usize>,
    node_count: &mut usize,
    name: &'static str,
) -> BuiltinResult<usize> {
    match token {
        NodeToken::Index(index) => {
            if index == 0 {
                return Err(graph_error(
                    name,
                    format!("{name}: node indices are one-based"),
                ));
            }
            let zero = index - 1;
            *node_count = (*node_count).max(index);
            if let Some(names) = node_names {
                if zero >= names.len() {
                    return Err(graph_error(
                        name,
                        format!("{name}: numeric node index exceeds supplied node names"),
                    ));
                }
            }
            Ok(zero)
        }
        NodeToken::Name(label) => {
            if node_names.is_none() {
                *node_names = Some(Vec::new());
            }
            if let Some(&idx) = label_to_index.get(&label) {
                return Ok(idx);
            }
            let idx = node_names.as_ref().map_or(0, Vec::len);
            label_to_index.insert(label.clone(), idx);
            node_names.as_mut().expect("node names").push(label);
            *node_count = (*node_count).max(idx + 1);
            Ok(idx)
        }
    }
}
