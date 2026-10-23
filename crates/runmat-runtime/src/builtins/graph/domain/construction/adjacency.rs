use super::super::class_registry::TriangleMode;
use super::super::parsing::{canonical, parse_node_names, scalar_text};
use super::*;
use crate::builtins::common::tensor;

pub(in crate::builtins::graph::domain) fn is_adjacency_form(
    args: crate::call::FunctionArgumentView<'_>,
) -> bool {
    args.first().is_some_and(|value| match value {
        Value::Tensor(tensor) => {
            tensor.shape().len() == 2 && tensor.shape()[0] == tensor.shape()[1]
        }
        Value::LogicalArray(array) => array.shape.len() == 2 && array.shape[0] == array.shape[1],
        Value::GpuTensor(handle) => {
            handle.shape().len() == 2 && handle.shape()[0] == handle.shape()[1]
        }
        _ => false,
    }) && (args.len() == 1
        || args.get(1).is_some_and(|arg| {
            matches!(arg, Value::String(_) | Value::CharArray(_))
                || matches!(arg,Value::StringArray(a) if super::super::parsing::is_vector_shape(&a.shape))
                || matches!(arg,Value::Cell(a) if super::super::parsing::is_vector_shape(&a.shape))
        }))
}

pub(super) fn graph_from_adjacency(
    name: &'static str,
    directed: bool,
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<Value> {
    let node_count = match &args[0] {
        Value::Tensor(matrix) => {
            let shape = matrix.shape();
            if shape.len() != 2 || shape[0] != shape[1] {
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
            shape[0]
        }
        Value::LogicalArray(array) => {
            if array.shape.len() != 2 || array.shape[0] != array.shape[1] {
                return Err(graph_error(
                    name,
                    format!("{name}: adjacency matrix must be square"),
                ));
            }
            array.shape[0]
        }
        _ => {
            return Err(graph_error(
                name,
                format!("{name}: expected adjacency matrix"),
            ))
        }
    };
    let mut mode = TriangleMode::Full;
    let mut node_names = None;
    if let Some(arg) = args.get(1) {
        if let Some(text) = scalar_text(arg).map_err(|source| graph_error(name, source))? {
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
                adjacency_weight(&args[0], row + col * node_count)
            } else {
                match mode {
                    TriangleMode::Full => {
                        if row > col {
                            continue;
                        }
                        let upper = adjacency_weight(&args[0], row + col * node_count);
                        if upper != 0.0 {
                            upper
                        } else if row == col {
                            0.0
                        } else {
                            adjacency_weight(&args[0], col + row * node_count)
                        }
                    }
                    TriangleMode::Upper => {
                        if row > col {
                            continue;
                        }
                        adjacency_weight(&args[0], row + col * node_count)
                    }
                    TriangleMode::Lower => {
                        if row < col {
                            continue;
                        }
                        adjacency_weight(&args[0], row + col * node_count)
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

fn adjacency_weight(value: &Value, index: usize) -> f64 {
    match value {
        Value::Tensor(matrix) => tensor::tensor_value_f64(matrix, index),
        Value::LogicalArray(array) => {
            if array.data[index] != 0 {
                1.0
            } else {
                0.0
            }
        }
        _ => unreachable!("adjacency storage was validated before edge construction"),
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::entrypoints::graph_builtin;
    use futures::executor::block_on;
    use runmat_value::{Tensor, Value};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn graph_adjacency_requires_exact_rank_two_geometry() {
        let rank_zero = Tensor::new(vec![0.0], Vec::new()).expect("rank-zero tensor");
        let rank_one = Tensor::new(vec![0.0, 0.0], vec![2]).expect("rank-one tensor");
        for tensor in [rank_zero, rank_one] {
            let error = block_on(graph_builtin(
                crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(
                    tensor,
                )]),
            ))
            .expect_err("non-matrix adjacency must be rejected");
            assert_eq!(error.identifier(), Some("RunMat:graph:InvalidArgument"));
            assert!(error.message().contains("adjacency matrix must be square"));
        }

        let matrix =
            Tensor::new(vec![0.0, 1.0, 1.0, 0.0], vec![2, 2]).expect("two-by-two adjacency");
        block_on(graph_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(matrix)]),
        ))
        .expect("rank-two square adjacency remains accepted");
    }
}
