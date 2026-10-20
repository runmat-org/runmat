//! Ordered graph argument admission at the provider boundary.

use super::construction::{graph_data, is_adjacency_form, node_tokens};
use super::parsing::{
    node_index_from_tensor, parse_string_vector, positive_integer_scalar, weights_from_value,
};
use super::{
    gather_if_needed_async, graph_error, BuiltinResult, GraphData, ObjectInstance, Value,
    DIGRAPH_CLASS, GRAPH_CLASS, NUM_NODES_PROPERTY,
};
use crate::builtins::table::{table_replace_variables_like, table_variables, value_row_count};
use runmat_builtins::{
    DIGRAPH_INTEGER_ADJACENCY_EXTENSION, DIGRAPH_INTEGER_WEIGHTS_EXTENSION,
    GRAPH_INTEGER_ADJACENCY_EXTENSION, GRAPH_INTEGER_WEIGHTS_EXTENSION,
};

mod nodes;
pub(super) use nodes::{
    admit_distance_selectors, admit_scalar_selectors, resolve_distance_selectors,
    resolve_scalar_selectors,
};

fn admit_constructor_integer_forms(
    args: crate::call::FunctionArgumentView<'_>,
    directed: bool,
    name: &'static str,
) -> BuiltinResult<()> {
    let adjacency = args.len() == 1 || is_adjacency_form(args);
    let extension = if adjacency {
        args.first()
            .filter(|value| {
                crate::builtins::common::validation::value_has_native_integer_class(value)
            })
            .map(|_| {
                if directed {
                    &DIGRAPH_INTEGER_ADJACENCY_EXTENSION
                } else {
                    &GRAPH_INTEGER_ADJACENCY_EXTENSION
                }
            })
    } else {
        args.get(2)
            .filter(|value| {
                super::parsing::is_numeric_weight_arg(value)
                    && crate::builtins::common::validation::value_has_native_integer_class(value)
            })
            .map(|_| {
                if directed {
                    &DIGRAPH_INTEGER_WEIGHTS_EXTENSION
                } else {
                    &GRAPH_INTEGER_WEIGHTS_EXTENSION
                }
            })
    };
    if let Some(extension) = extension {
        crate::compatibility::ensure_builtin_extension_enabled(extension, name)?;
    }
    Ok(())
}

pub(super) async fn graph_node_count(value: &Value, name: &'static str) -> BuiltinResult<usize> {
    let object = expect_graph_object(value, name)?;
    if let Some(count) = object.properties().get(NUM_NODES_PROPERTY) {
        if is_host_value(count) {
            return positive_integer_scalar(count, name);
        }
        let gathered = gather_if_needed_async(count).await?;
        return positive_integer_scalar(&gathered, name);
    }
    super::construction::infer_node_count_from_nodes(object, name)
}

pub(super) fn graph_edge_count(value: &Value, name: &'static str) -> BuiltinResult<usize> {
    let object = expect_graph_object(value, name)?;
    let Some(Value::Object(edges)) = object.properties().get("Edges") else {
        return Ok(0);
    };
    let variables = table_variables(edges)
        .map_err(|error| graph_error(name, format!("{name}: invalid Edges table: {error}")))?;
    let Some(endpoints) = variables.fields.get("EndNodes") else {
        return Ok(0);
    };
    value_row_count(endpoints)
        .map_err(|error| graph_error(name, format!("{name}: invalid Edges.EndNodes: {error}")))
}

pub(super) fn count_as_double(count: usize, name: &'static str) -> BuiltinResult<f64> {
    let wide = u64::try_from(count).map_err(|error| {
        graph_error(name, format!("{name}: count is not representable: {error}"))
    })?;
    if !crate::builtins::common::validation::integer_is_exact_f64(&runmat_value::IntValue::U64(
        wide,
    )) {
        return Err(graph_error(
            name,
            format!("{name}: count cannot be represented exactly as double"),
        ));
    }
    Ok(count as f64)
}

fn expect_graph_object<'a>(
    value: &'a Value,
    name: &'static str,
) -> BuiltinResult<&'a ObjectInstance> {
    match value {
        Value::Object(object) if object.is_class(GRAPH_CLASS) || object.is_class(DIGRAPH_CLASS) => {
            Ok(object)
        }
        other => Err(graph_error(
            name,
            format!("{name}: expected graph or digraph object, got {other:?}"),
        )),
    }
}

pub(super) async fn constructor_arguments(
    args: crate::call::BuiltinArguments,
    runtime: Option<&crate::context::RuntimeContext>,
    directed: bool,
    name: &'static str,
) -> BuiltinResult<crate::call::BuiltinArguments> {
    admit_constructor_integer_forms(args.view(), directed, name)?;
    if args.len() > 4 {
        return Err(graph_error(
            name,
            format!("{name}: expected at most four constructor arguments"),
        ));
    }
    if is_adjacency_form(args.view()) && args.len() > 2 {
        return Err(graph_error(
            name,
            format!("{name}: adjacency construction accepts at most one option or name list"),
        ));
    }
    // Validate all locally decidable roles before downloading the first
    // resident input. A resident operand's contents are validated afterward by
    // the constructor itself.
    if args.len() == 1 || is_adjacency_form(args.view()) {
        if let Some(Value::Tensor(matrix)) = args.first() {
            let shape = matrix.shape();
            if shape.len() != 2 || shape[0] != shape[1] {
                return Err(graph_error(
                    name,
                    format!("{name}: adjacency matrix must be square"),
                ));
            }
        } else if let Some(Value::LogicalArray(array)) = args.first() {
            if array.shape.len() != 2 || array.shape[0] != array.shape[1] {
                return Err(graph_error(
                    name,
                    format!("{name}: adjacency matrix must be square"),
                ));
            }
        } else if args.first().is_some_and(is_host_value) {
            return Err(graph_error(
                name,
                format!("{name}: expected adjacency matrix"),
            ));
        }
        if let Some(option) = args.get(1).filter(|value| is_host_value(value)) {
            if parse_string_vector(option).is_err() {
                return Err(graph_error(
                    name,
                    format!("{name}: invalid adjacency option or node names"),
                ));
            }
        }
    } else if args.len() >= 2 {
        let source = args
            .first()
            .filter(|value| is_host_value(value))
            .map(|value| node_tokens(value, name))
            .transpose()?;
        let target = args
            .get(1)
            .filter(|value| is_host_value(value))
            .map(|value| node_tokens(value, name))
            .transpose()?;
        if let (Some(source), Some(target)) = (&source, &target) {
            if source.len() != target.len() {
                return Err(graph_error(
                    name,
                    format!("{name}: source and target lists must have the same length"),
                ));
            }
        }
        if let Some(value) = args.get(2).filter(|value| is_host_value(value)) {
            if super::parsing::is_numeric_weight_arg(value) {
                if let Some(source) = &source {
                    weights_from_value(value, source.len(), name)?;
                } else {
                    let weights = super::parsing::numeric_f64_vector(value, name)?;
                    if weights
                        .iter()
                        .any(|weight| !weight.is_finite() || *weight < 0.0)
                    {
                        return Err(graph_error(
                            name,
                            format!("{name}: edge weights must be finite nonnegative values"),
                        ));
                    }
                }
            } else {
                parse_string_vector(value)
                    .map_err(|message| graph_error(name, format!("{name}: {message}")))?;
            }
        }
        if args.len() == 4
            && args.get(2).is_some_and(|value| {
                is_host_value(value) && !super::parsing::is_numeric_weight_arg(value)
            })
        {
            return Err(graph_error(
                name,
                format!(
                    "{name}: the third argument must contain edge weights when names are fourth"
                ),
            ));
        }
        if let Some(value) = args.get(3).filter(|value| is_host_value(value)) {
            parse_string_vector(value)
                .map_err(|message| graph_error(name, format!("{name}: {message}")))?;
        }
    }
    crate::dispatcher::gather_builtin_arguments_async(args, runtime).await
}

/// Materialize only the graph representation fields consumed by graph queries.
/// Host-invalid fields are checked before any owner download.
pub(super) async fn graph_data_async(
    value: &Value,
    name: &'static str,
) -> BuiltinResult<GraphData> {
    let Value::Object(object) = value else {
        return graph_data(value, name);
    };
    if !object.is_class(GRAPH_CLASS) && !object.is_class(DIGRAPH_CLASS) {
        return graph_data(value, name);
    }
    preflight_graph_host_fields(object, name)?;
    let mut object = object.clone();
    if let Some(value) = object.property(NUM_NODES_PROPERTY) {
        let context = crate::context::legacy::active();
        let _source = crate::execution::structure::replacement::lease(
            &Value::Object(object.clone()),
            context.as_ref(),
        )?;
        let gathered = gather_if_needed_async(value).await?;
        let gathered = crate::execution::structure::replacement::root(gathered, context.as_ref())?;
        gathered.with_owned(|value| {
            crate::execution::object::replacement::replace_property(
                context.as_ref(),
                &mut object,
                NUM_NODES_PROPERTY,
                value,
                |message, source| {
                    crate::context::host_allocation::host_allocation_runtime_error(
                        context.as_ref(),
                        "RunMat:graph:ObjectAllocation",
                        message,
                        source,
                    )
                },
                |message| graph_error(name, message),
            )
        })?;
    }
    let node_columns: &[&str] = if object.properties().contains_key(NUM_NODES_PROPERTY) {
        &["Name"]
    } else {
        &["Name", "Index"]
    };
    gather_table_columns(&mut object, "Nodes", node_columns, name).await?;
    gather_table_columns(&mut object, "Edges", &["EndNodes", "Weight"], name).await?;
    graph_data(&Value::Object(object), name)
}

fn preflight_graph_host_fields(object: &ObjectInstance, name: &'static str) -> BuiltinResult<()> {
    let count = object
        .properties()
        .get(NUM_NODES_PROPERTY)
        .filter(|value| is_host_value(value))
        .map(|value| positive_integer_scalar(value, name))
        .transpose()?;
    if let Some(Value::Object(nodes)) = object.properties().get("Nodes") {
        let vars = table_variables(nodes)
            .map_err(|error| graph_error(name, format!("{name}: invalid Nodes table: {error}")))?;
        if let Some(value) = vars.fields.get("Name").filter(|value| is_host_value(value)) {
            let names = parse_string_vector(value)
                .map_err(|message| graph_error(name, format!("{name}: {message}")))?;
            if count.is_some_and(|count| names.len() != count) {
                return Err(graph_error(
                    name,
                    format!("{name}: Nodes.Name length does not match graph node count"),
                ));
            }
        }
    }
    if let Some(Value::Object(edges)) = object.properties().get("Edges") {
        let vars = table_variables(edges)
            .map_err(|error| graph_error(name, format!("{name}: invalid Edges table: {error}")))?;
        let endpoints = vars.fields.get("EndNodes");
        let endpoint_rows = match endpoints {
            Some(Value::Tensor(tensor)) if tensor.shape().len() == 2 && tensor.shape()[1] == 2 => {
                Some(tensor.shape()[0])
            }
            Some(Value::StringArray(array)) if array.shape.len() == 2 && array.cols == 2 => {
                Some(array.rows)
            }
            Some(Value::GpuTensor(handle))
                if handle.shape().len() == 2 && handle.shape()[1] == 2 =>
            {
                Some(handle.shape()[0])
            }
            Some(_) => {
                return Err(graph_error(
                    name,
                    format!("{name}: Edges.EndNodes must be an m-by-2 numeric or string array"),
                ))
            }
            None => None,
        };
        if let Some(Value::Tensor(tensor)) = endpoints {
            for index in 0..super::tensor::tensor_element_len(tensor) {
                node_index_from_tensor(tensor, index, count.unwrap_or(usize::MAX), name)?;
            }
        }
        if let Some(value) = vars
            .fields
            .get("Weight")
            .filter(|value| is_host_value(value))
        {
            if let Some(expected) = endpoint_rows {
                weights_from_value(value, expected, name)?;
            } else {
                let weights = super::parsing::numeric_f64_vector(value, name)?;
                if weights
                    .iter()
                    .any(|weight| !weight.is_finite() || *weight < 0.0)
                {
                    return Err(graph_error(
                        name,
                        format!("{name}: edge weights must be finite nonnegative values"),
                    ));
                }
            }
        }
    }
    Ok(())
}

async fn gather_table_columns(
    object: &mut ObjectInstance,
    property: &str,
    columns: &[&str],
    name: &'static str,
) -> BuiltinResult<()> {
    let Some(Value::Object(table)) = object.property(property) else {
        return Ok(());
    };
    let variables = table_variables(table)
        .map_err(|error| graph_error(name, format!("{name}: invalid {property} table: {error}")))?;
    let context = crate::context::legacy::active();
    let source =
        crate::execution::structure::replacement::root(Value::Struct(variables), context.as_ref())?;
    let Value::Struct(source_variables) = source.value() else {
        unreachable!("variable structure")
    };
    let prepared_variables =
        crate::execution::structure::replacement::record(source_variables, context.as_ref())?;
    let _roots = prepared_variables.1;
    let mut variables = prepared_variables.0;
    let mut changed = false;
    for column in columns {
        if let Some(value) = variables.fields.get(column) {
            if !is_host_value(value) {
                let gathered = gather_if_needed_async(value).await?;
                let gathered =
                    crate::execution::structure::replacement::root(gathered, context.as_ref())?;
                variables
                    .try_replace_unique(column, gathered.value().clone())
                    .expect("new admitted map contains the selected column");
                changed = true;
            }
        }
    }
    if changed {
        let replacement = table_replace_variables_like(table, variables)?;
        let replacement =
            crate::execution::structure::replacement::root(replacement, context.as_ref())?;
        replacement.with_owned(|replacement| {
            crate::execution::object::replacement::replace_property(
                context.as_ref(),
                object,
                property,
                replacement,
                |message, source| {
                    crate::context::host_allocation::host_allocation_runtime_error(
                        context.as_ref(),
                        "RunMat:graph:ObjectAllocation",
                        message,
                        source,
                    )
                },
                |message| graph_error(name, message),
            )
        })?;
    }
    Ok(())
}

fn is_host_value(value: &Value) -> bool {
    !crate::dispatcher::value_contains_gpu(value)
}
