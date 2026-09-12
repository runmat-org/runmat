use super::algorithms::tensor_value_raw;
use super::class_registry::{ensure_graph_classes_registered, TriangleMode};
use super::parsing::{
    canonical, looks_like_weights, node_index_from_tensor, numeric_vector, parse_node_names,
    parse_string_vector, positive_integer_scalar, scalar_text, weights_from_value,
};
use super::*;

pub(super) fn construct_graph(
    name: &'static str,
    directed: bool,
    args: Vec<Value>,
) -> BuiltinResult<Value> {
    if args.is_empty() {
        return graph_object(GraphData {
            directed,
            node_names: None,
            edges: Vec::new(),
            node_count: 0,
        });
    }
    if args.len() == 1 || is_adjacency_form(&args) {
        return graph_from_adjacency(name, directed, &args);
    }
    graph_from_edges(name, directed, &args)
}

pub(super) fn is_adjacency_form(args: &[Value]) -> bool {
    matches!(args.first(), Some(Value::Tensor(t)) if t.rows == t.cols)
        && (args.len() == 1
            || args
                .get(1)
                .is_some_and(|arg| parse_string_vector(arg).is_ok() || scalar_text(arg).is_some()))
}

pub(super) fn graph_from_adjacency(
    name: &'static str,
    directed: bool,
    args: &[Value],
) -> BuiltinResult<Value> {
    let Value::Tensor(matrix) = &args[0] else {
        return Err(graph_error(
            name,
            format!("{name}: expected adjacency matrix"),
        ));
    };
    if matrix.rows != matrix.cols {
        return Err(graph_error(
            name,
            format!("{name}: adjacency matrix must be square"),
        ));
    }
    let node_count = matrix.rows;
    let mut mode = TriangleMode::Full;
    let mut node_names = None;
    if let Some(arg) = args.get(1) {
        if let Some(text) = scalar_text(arg) {
            match canonical(&text).as_str() {
                "upper" => mode = TriangleMode::Upper,
                "lower" => mode = TriangleMode::Lower,
                _ => node_names = Some(parse_node_names(arg, node_count, name)?),
            }
        } else {
            node_names = Some(parse_node_names(arg, node_count, name)?);
        }
    }
    let mut edges = Vec::new();
    for col in 0..node_count {
        for row in 0..node_count {
            let weight = if directed {
                tensor::tensor_value_f64(matrix, row + col * matrix.rows)
            } else {
                match mode {
                    TriangleMode::Full => {
                        if row > col {
                            continue;
                        }
                        let upper = tensor::tensor_value_f64(matrix, row + col * matrix.rows);
                        if upper != 0.0 {
                            upper
                        } else if row == col {
                            0.0
                        } else {
                            tensor::tensor_value_f64(matrix, col + row * matrix.rows)
                        }
                    }
                    TriangleMode::Upper => {
                        if row > col {
                            continue;
                        }
                        tensor::tensor_value_f64(matrix, row + col * matrix.rows)
                    }
                    TriangleMode::Lower => {
                        if row < col {
                            continue;
                        }
                        tensor::tensor_value_f64(matrix, row + col * matrix.rows)
                    }
                }
            };
            if weight != 0.0 {
                edges.push(Edge {
                    source: row,
                    target: col,
                    weight,
                });
            }
        }
    }
    graph_object(GraphData {
        directed,
        node_names,
        edges,
        node_count,
    })
}

pub(super) fn graph_from_edges(
    name: &'static str,
    directed: bool,
    args: &[Value],
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
        Some(value) if looks_like_weights(value, source_nodes.len()) => {
            weights_from_value(value, source_nodes.len(), name)?
        }
        _ => vec![1.0; source_nodes.len()],
    };
    let names_arg = if args.len() >= 4 {
        args.get(3)
    } else if args.len() == 3 && !looks_like_weights(&args[2], source_nodes.len()) {
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

pub(super) fn graph_object(graph: GraphData) -> BuiltinResult<Value> {
    ensure_graph_classes_registered();
    let class_name = if graph.directed {
        DIGRAPH_CLASS
    } else {
        GRAPH_CLASS
    };
    let mut object = ObjectInstance::new(class_name.to_string());
    object.properties.insert(
        NUM_NODES_PROPERTY.to_string(),
        Value::Num(graph.node_count as f64),
    );
    object
        .properties
        .insert("Edges".to_string(), edges_table(&graph)?);
    object
        .properties
        .insert("Nodes".to_string(), nodes_table(&graph)?);
    Ok(Value::Object(object))
}

pub(super) fn edges_table(graph: &GraphData) -> BuiltinResult<Value> {
    let names = graph.node_names.as_ref();
    let end_nodes = if let Some(node_names) = names {
        let mut values = Vec::with_capacity(graph.edges.len() * 2);
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
        let mut values = Vec::with_capacity(graph.edges.len() * 2);
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
                (1..=graph.node_count).map(|idx| idx as f64).collect(),
                vec![graph.node_count, 1],
                "graph",
            )?],
        )
    }
}

pub(super) fn graph_data(value: &Value, name: &'static str) -> BuiltinResult<GraphData> {
    let object = match value {
        Value::Object(object)
            if object.class_name == GRAPH_CLASS || object.class_name == DIGRAPH_CLASS =>
        {
            object
        }
        other => {
            return Err(graph_error(
                name,
                format!("{name}: expected graph or digraph object, got {other:?}"),
            ))
        }
    };
    let directed = object.class_name == DIGRAPH_CLASS;
    let node_count = match object.properties.get(NUM_NODES_PROPERTY) {
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
    let Some(Value::Object(nodes)) = object.properties.get("Nodes") else {
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

pub(super) fn infer_node_count_from_nodes(
    object: &ObjectInstance,
    name: &'static str,
) -> BuiltinResult<usize> {
    let Some(Value::Object(nodes)) = object.properties.get("Nodes") else {
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
    let Some(Value::Object(edges_table)) = object.properties.get("Edges") else {
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
        Value::Tensor(tensor) if tensor.cols == 2 => {
            let mut out = Vec::with_capacity(tensor.rows);
            for row in 0..tensor.rows {
                let source = node_index_from_tensor(tensor, row, node_count, name)?;
                let target = node_index_from_tensor(tensor, row + tensor.rows, node_count, name)?;
                out.push((source, target));
            }
            Ok(out)
        }
        Value::StringArray(array) if array.cols == 2 => {
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
                let source = *lookup.get(array.data[row].as_str()).ok_or_else(|| {
                    graph_error(
                        name,
                        format!("{name}: unknown source node '{}'", array.data[row]),
                    )
                })?;
                let target = *lookup
                    .get(array.data[row + array.rows].as_str())
                    .ok_or_else(|| {
                        graph_error(
                            name,
                            format!(
                                "{name}: unknown target node '{}'",
                                array.data[row + array.rows]
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

pub(super) fn node_tokens(value: &Value, name: &'static str) -> BuiltinResult<Vec<NodeToken>> {
    if let Ok(values) = numeric_vector(value, name) {
        return Ok(values.into_iter().map(NodeToken::Index).collect());
    }
    Ok(parse_string_vector(value)
        .map_err(|err| graph_error(name, format!("{name}: {err}")))?
        .into_iter()
        .map(NodeToken::Name)
        .collect())
}

#[derive(Debug)]
pub(super) enum NodeToken {
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
