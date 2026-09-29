use super::super::class_registry::TriangleMode;
use super::super::parsing::{canonical, parse_node_names, parse_string_vector, scalar_text};
use super::*;
use crate::builtins::common::tensor;

pub(in crate::builtins::graph::domain) fn is_adjacency_form(args: &[Value]) -> bool {
    matches!(args.first(), Some(Value::Tensor(t)) if t.shape.len() == 2 && t.rows == t.cols)
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
    if matrix.shape.len() != 2 || matrix.rows != matrix.cols {
        return Err(graph_error(
            name,
            format!("{name}: adjacency matrix must be square"),
        ));
    }
    if !crate::builtins::common::validation::native_integer_value_is_exact_f64(&args[0]) {
        return Err(graph_error(
            name,
            format!("{name}: adjacency weights cannot be represented exactly as double"),
        ));
    }
    for index in 0..tensor::tensor_element_len(matrix) {
        let weight = tensor::tensor_value_f64(matrix, index);
        if !weight.is_finite() || weight < 0.0 {
            return Err(graph_error(
                name,
                format!("{name}: adjacency weights must be finite nonnegative values"),
            ));
        }
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
