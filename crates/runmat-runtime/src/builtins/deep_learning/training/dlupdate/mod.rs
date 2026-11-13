//! Callback traversal across matching deep-learning parameter trees.

use super::super::{deep_learning_error, model, unsupported_error};
use crate::execution::{RootedValue, RootedValueLease, RootedValues, RootedValuesBuilder};
use crate::BuiltinResult;
use runmat_macros::runtime_builtin;
use runmat_value::{CellArray, ObjectInstance, StructArray, StructValue, Value};

mod callback;
mod containers;
mod learnables;
mod preflight;

use callback::*;
use containers::*;
use learnables::*;

const DLUPDATE_MAX_DEPTH: usize = 256;

fn into_owned_outputs(
    values: runmat_value::OutputListStorage,
) -> BuiltinResult<runmat_value::OwnedOutputListValues> {
    values.try_into_owned().map_err(|_| {
        crate::sequence::sequence_error_to_runtime(
            runmat_value::ValueSequenceError::SharedOutputStorage,
        )
    })
}

#[runtime_builtin(
    name = "dlupdate",
    binding_variant = "default",
    context = true,
    requested_outputs = true,
    builtin_path = "crate::builtins::deep_learning::training"
)]
async fn dlupdate_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    requested_outputs: Option<usize>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    dlupdate_in_runtime(runtime.as_ref(), requested_outputs, args).await
}

#[cfg(test)]
pub(in crate::builtins::deep_learning) async fn dlupdate_builtin(
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    dlupdate_in_runtime(None, crate::output_count::current_output_count(), args).await
}

async fn dlupdate_in_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
    requested_outputs: Option<usize>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<RootedValue> {
    let _argument_roots =
        RootedValueLease::for_values(args.iter()).map_err(|error| error.into_runtime_error())?;
    let mut args = args.into_iter();
    let Some(function) = args.next() else {
        return Err(deep_learning_error(
            "dlupdate",
            "dlupdate: expected a function handle followed by one or more parameter trees",
        ));
    };
    if !is_function_handle(&function) {
        return Err(deep_learning_error(
            "dlupdate",
            format!("dlupdate: expected a function handle, got {function:?}"),
        ));
    }
    let trees = args.collect::<Vec<_>>();
    if trees.is_empty() {
        return Err(deep_learning_error(
            "dlupdate",
            "dlupdate: expected at least one parameter tree",
        ));
    }

    let requested_outputs = requested_outputs.unwrap_or(1);
    preflight::matching_trees(&trees)?;
    let outputs = dlupdate_node(runtime, &function, &trees, requested_outputs, 0).await?;
    outputs.with_owned_output_storage(|outputs| {
        let result = if requested_outputs == 0 {
            Value::OutputList(outputs)
        } else if requested_outputs == 1 {
            outputs
                .try_into_first()
                .map_err(|_| {
                    crate::sequence::sequence_error_to_runtime(
                        runmat_value::ValueSequenceError::SharedOutputStorage,
                    )
                })?
                .expect("one output")
        } else {
            Value::OutputList(outputs)
        };
        RootedValue::new(result).map_err(|error| error.into_runtime_error())
    })
}

fn is_function_handle(value: &Value) -> bool {
    matches!(
        value,
        Value::ScopedFunctionHandle(_)
            | Value::FunctionHandle(_)
            | Value::ExternalFunctionHandle(_)
            | Value::MethodFunctionHandle(_)
            | Value::BoundFunctionHandle(_)
            | Value::Closure(_)
    )
}

#[async_recursion::async_recursion(?Send)]
async fn dlupdate_node(
    runtime: Option<&crate::context::RuntimeContext>,
    function: &Value,
    values: &[Value],
    requested_outputs: usize,
    depth: usize,
) -> BuiltinResult<RootedValues> {
    if depth > DLUPDATE_MAX_DEPTH {
        return Err(deep_learning_error(
            "dlupdate",
            "dlupdate: parameter tree nesting exceeds supported recursion depth",
        ));
    }
    let Some(first) = values.first() else {
        return Err(deep_learning_error(
            "dlupdate",
            "dlupdate: internal empty parameter-tree node",
        ));
    };

    match first {
        Value::Struct(reference) => {
            if is_learnables_struct(reference) {
                dlupdate_learnables_struct(runtime, function, reference, values, requested_outputs, depth + 1)
                    .await
            } else {
                dlupdate_struct(runtime, function, reference, values, requested_outputs, depth + 1).await
            }
        }
        Value::Cell(reference) => {
            dlupdate_cell(runtime, function, reference, values, requested_outputs, depth + 1).await
        }
        Value::StructArray(reference) => {
            dlupdate_struct_array(runtime, function, reference, values, requested_outputs, depth + 1).await
        }
        Value::Object(object) if crate::builtins::table::is_tabular_object(object) => {
            dlupdate_table(runtime, function, object, values, requested_outputs, depth + 1).await
        }
        Value::Object(object) if model::is_deep_learning_network_object(object) => {
            dlupdate_network(runtime, function, object, values, requested_outputs, depth + 1).await
        }
        Value::Object(object) if is_leaf_object(object) => {
            if values.iter().any(is_tree_container) {
                return Err(deep_learning_error(
                    "dlupdate",
                    "dlupdate: all parameter trees must have matching container structure",
                ));
            }
            dlupdate_leaf(runtime, function, values, requested_outputs).await
        }
        Value::Object(object) => Err(unsupported_error(
            "dlupdate",
            format!(
                "dlupdate: traversal for '{}' objects requires dlnetwork/model metadata infrastructure",
                object.class_name()
            ),
        )),
        _ => {
            if values.iter().any(is_tree_container) {
                return Err(deep_learning_error(
                    "dlupdate",
                    "dlupdate: all parameter trees must have matching container structure",
                ));
            }
            dlupdate_leaf(runtime, function, values, requested_outputs).await
        }
    }
}

fn is_leaf_object(object: &ObjectInstance) -> bool {
    object.class_name().is(super::super::DLARRAY_CLASS)
}

fn is_tree_container(value: &Value) -> bool {
    match value {
        Value::Struct(_) | Value::StructArray(_) | Value::Cell(_) => true,
        Value::Object(object) => {
            crate::builtins::table::is_tabular_object(object)
                || model::is_deep_learning_network_object(object)
        }
        _ => false,
    }
}
