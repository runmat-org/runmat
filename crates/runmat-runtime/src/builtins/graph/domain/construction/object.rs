use super::super::algorithms::tensor_value_raw;
use super::super::class_registry::ensure_graph_classes_registered;
use super::super::parsing::{
    node_index_from_tensor, parse_string_vector, positive_integer_scalar, weights_from_value,
};
use super::super::{internal_error, DIGRAPH_CLASS, GRAPH_CLASS, NUM_NODES_PROPERTY};
use super::*;
use crate::builtins::table::{table_from_columns, table_variables};
use runmat_value::{IntValue, ObjectInstance, StringArray};
use std::collections::HashMap;

pub(super) fn graph_object(graph: GraphData) -> BuiltinResult<Value> {
    let count = u64::try_from(graph.node_count).map_err(|error| {
        graph_error(
            "graph",
            format!("graph: node count is not representable: {error}"),
        )
    })?;
    if !crate::builtins::common::validation::integer_is_exact_f64(&IntValue::U64(count)) {
        return Err(graph_error(
            "graph",
            "graph: node count cannot be represented exactly as double",
        ));
    }
    ensure_graph_classes_registered();
    let class_name = if graph.directed {
        DIGRAPH_CLASS
    } else {
        GRAPH_CLASS
    };
    let fields = [
        (NUM_NODES_PROPERTY, Value::Num(graph.node_count as f64)),
        ("Edges", edges_table(&graph)?),
        ("Nodes", nodes_table(&graph)?),
    ];
    let context = crate::context::legacy::active();
    crate::execution::object::construct_object_from_borrowed_fields(
        context.as_ref(),
        class_name.display_name(),
        fields.iter().map(|(name, value)| (*name, value)),
    )
    .map(|value| value.with_owned(|value| value))
}

pub(super) fn edges_table(graph: &GraphData) -> BuiltinResult<Value> {
    let names = graph.node_names.as_ref();
    let end_nodes =
        if let Some(node_names) = names {
            let capacity =
                graph.edges.len().checked_mul(2).ok_or_else(|| {
                    graph_error("graph", "graph: edge endpoint dimensions overflow")
                })?;
            let mut values = Vec::new();
            values.try_reserve_exact(capacity).map_err(|error| {
                graph_error("graph", format!("graph: edge allocation failed: {error}"))
            })?;
            values.extend(
                graph
                    .edges
                    .iter()
                    .map(|edge| node_names[edge.source].clone()),
            );
            values.extend(
                graph
                    .edges
                    .iter()
                    .map(|edge| node_names[edge.target].clone()),
            );
            Value::StringArray(
                StringArray::new(values, vec![graph.edges.len(), 2])
                    .map_err(|err| internal_error("graph", err))?,
            )
        } else {
            let capacity =
                graph.edges.len().checked_mul(2).ok_or_else(|| {
                    graph_error("graph", "graph: edge endpoint dimensions overflow")
                })?;
            let mut values = Vec::new();
            values.try_reserve_exact(capacity).map_err(|error| {
                graph_error("graph", format!("graph: edge allocation failed: {error}"))
            })?;
            values.extend(graph.edges.iter().map(|edge| (edge.source + 1) as f64));
            values.extend(graph.edges.iter().map(|edge| (edge.target + 1) as f64));
            tensor_value_raw(values, vec![graph.edges.len(), 2], "graph")?
        };
    let all_unit = graph.edges.iter().all(|edge| edge.weight == 1.0);
    if all_unit {
        table_from_columns(vec!["EndNodes".to_string()], vec![end_nodes])
    } else {
        let weights: Vec<f64> = graph.edges.iter().map(|edge| edge.weight).collect();
        table_from_columns(
            vec!["EndNodes".to_string(), "Weight".to_string()],
            vec![
                end_nodes,
                tensor_value_raw(weights, vec![graph.edges.len(), 1], "graph")?,
            ],
        )
    }
}

pub(super) fn nodes_table(graph: &GraphData) -> BuiltinResult<Value> {
    if let Some(names) = &graph.node_names {
        table_from_columns(
            vec!["Name".to_string()],
            vec![Value::StringArray(
                StringArray::new(names.clone(), vec![graph.node_count, 1])
                    .map_err(|err| internal_error("graph", err))?,
            )],
        )
    } else {
        table_from_columns(
            vec!["Index".to_string()],
            vec![tensor_value_raw(
                node_index_column(graph.node_count)?,
                vec![graph.node_count, 1],
                "graph",
            )?],
        )
    }
}

fn node_index_column(count: usize) -> BuiltinResult<Vec<f64>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|error| graph_error("graph", format!("graph: node allocation failed: {error}")))?;
    values.extend((1..=count).map(|idx| idx as f64));
    Ok(values)
}

pub(in crate::builtins::graph::domain) fn graph_data(
    value: &Value,
    name: &'static str,
) -> BuiltinResult<GraphData> {
    let object = match value {
        Value::Object(object) if object.is_class(GRAPH_CLASS) || object.is_class(DIGRAPH_CLASS) => {
            object
        }
        other => {
            return Err(graph_error(
                name,
                format!("{name}: expected graph or digraph object, got {other:?}"),
            ))
        }
    };
    let directed = object.is_class(DIGRAPH_CLASS);
    let node_count = match object.properties().get(NUM_NODES_PROPERTY) {
        Some(value) => positive_integer_scalar(value, name)?,
        None => infer_node_count_from_nodes(object, name)?,
    };
    let node_names = node_names_from_object(object, node_count, name)?;
    let edges = edges_from_object(object, node_names.as_deref(), directed, node_count, name)?;
    Ok(GraphData {
        directed,
        node_names,
        edges,
        node_count,
    })
}

pub(super) fn node_names_from_object(
    object: &ObjectInstance,
    node_count: usize,
    name: &'static str,
) -> BuiltinResult<Option<Vec<String>>> {
    let Some(Value::Object(nodes)) = object.properties().get("Nodes") else {
        return Ok(None);
    };
    let variables = table_variables(nodes)
        .map_err(|err| graph_error(name, format!("{name}: invalid Nodes table: {err}")))?;
    let Some(value) = variables.fields.get("Name") else {
        return Ok(None);
    };
    let names = parse_string_vector(value).map_err(|err| graph_error(name, err))?;
    if names.len() != node_count {
        return Err(graph_error(
            name,
            format!("{name}: Nodes.Name length does not match graph node count"),
        ));
    }
    Ok(Some(names))
}

pub(in crate::builtins::graph::domain) fn infer_node_count_from_nodes(
    object: &ObjectInstance,
    name: &'static str,
) -> BuiltinResult<usize> {
    let Some(Value::Object(nodes)) = object.properties().get("Nodes") else {
        return Ok(0);
    };
    crate::builtins::table::table_height(nodes)
        .map_err(|err| graph_error(name, format!("{name}: invalid Nodes table: {err}")))
}

pub(super) fn edges_from_object(
    object: &ObjectInstance,
    node_names: Option<&[String]>,
    directed: bool,
    node_count: usize,
    name: &'static str,
) -> BuiltinResult<Vec<Edge>> {
    let Some(Value::Object(edges_table)) = object.properties().get("Edges") else {
        return Ok(Vec::new());
    };
    let variables = table_variables(edges_table)
        .map_err(|err| graph_error(name, format!("{name}: invalid Edges table: {err}")))?;
    let Some(endnodes) = variables.fields.get("EndNodes") else {
        return Ok(Vec::new());
    };
    let endpoint_pairs = edge_pairs_from_value(endnodes, node_names, node_count, name)?;
    let weights = match variables.fields.get("Weight") {
        Some(value) => weights_from_value(value, endpoint_pairs.len(), name)?,
        None => vec![1.0; endpoint_pairs.len()],
    };
    let mut edges = Vec::with_capacity(endpoint_pairs.len());
    for ((source, target), weight) in endpoint_pairs.into_iter().zip(weights) {
        if source >= node_count || target >= node_count {
            return Err(graph_error(
                name,
                format!("{name}: edge endpoint exceeds graph node count"),
            ));
        }
        edges.push(Edge {
            source,
            target,
            weight,
        });
    }
    if !directed {
        for edge in &edges {
            if edge.source >= node_count || edge.target >= node_count {
                return Err(graph_error(name, format!("{name}: invalid edge endpoint")));
            }
        }
    }
    Ok(edges)
}

pub(super) fn edge_pairs_from_value(
    value: &Value,
    node_names: Option<&[String]>,
    node_count: usize,
    name: &'static str,
) -> BuiltinResult<Vec<(usize, usize)>> {
    match value {
        Value::Tensor(tensor) if tensor.shape().len() == 2 && tensor.shape()[1] == 2 => {
            let rows = tensor.shape()[0];
            let mut out = Vec::with_capacity(rows);
            for row in 0..rows {
                let source = node_index_from_tensor(tensor, row, node_count, name)?;
                let target = node_index_from_tensor(tensor, row + rows, node_count, name)?;
                out.push((source, target));
            }
            Ok(out)
        }
        Value::StringArray(array) if array.shape.len() == 2 && array.cols == 2 => {
            let Some(names) = node_names else {
                return Err(graph_error(
                    name,
                    format!("{name}: string EndNodes require Nodes.Name values"),
                ));
            };
            let lookup: HashMap<&str, usize> = names
                .iter()
                .enumerate()
                .map(|(idx, text)| (text.as_str(), idx))
                .collect();
            let mut out = Vec::with_capacity(array.rows);
            for row in 0..array.rows {
                let source = *lookup
                    .get(array.data()[row].as_str().ok_or_else(|| {
                        graph_error(
                            name,
                            format!("{name}: source node names must be nonmissing"),
                        )
                    })?)
                    .ok_or_else(|| {
                        graph_error(
                            name,
                            format!(
                                "{name}: unknown source node '{}'",
                                array.data()[row].display()
                            ),
                        )
                    })?;
                let target = *lookup
                    .get(array.data()[row + array.rows].as_str().ok_or_else(|| {
                        graph_error(
                            name,
                            format!("{name}: target node names must be nonmissing"),
                        )
                    })?)
                    .ok_or_else(|| {
                        graph_error(
                            name,
                            format!(
                                "{name}: unknown target node '{}'",
                                array.data()[row + array.rows].display()
                            ),
                        )
                    })?;
                out.push((source, target));
            }
            Ok(out)
        }
        other => Err(graph_error(
            name,
            format!(
                "{name}: Edges.EndNodes must be an m-by-2 numeric or string array, got {other:?}"
            ),
        )),
    }
}
