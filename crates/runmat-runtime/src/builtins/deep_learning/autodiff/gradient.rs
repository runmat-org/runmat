use super::*;
use runmat_builtins::DeepLearningParameterTableColumn as Column;
use std::collections::HashMap;

mod parameter_table;
pub(in crate::builtins::deep_learning::autodiff) mod resident;

enum GradientSource<'a> {
    Host(&'a HashMap<usize, Tensor>),
    Resident(&'a HashMap<usize, runmat_accelerate_api::GpuTensorHandle>),
}

#[runtime_builtin(
    name = "dlgradient",
    binding_variant = "default",
    context = true,
    requested_outputs = true,
    builtin_path = "crate::builtins::deep_learning::autodiff::gradient"
)]
pub(crate) async fn dlgradient_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    requested_outputs: Option<usize>,
    loss: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    dlgradient_in_runtime(runtime.as_ref(), requested_outputs, loss, rest).await
}

#[cfg(test)]
pub(in crate::builtins::deep_learning) async fn dlgradient_direct(
    loss: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    dlgradient_in_runtime(
        None,
        crate::output_count::current_output_count(),
        loss,
        rest,
    )
    .await
}

async fn dlgradient_in_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
    requested_outputs: Option<usize>,
    loss: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    if rest.is_empty() {
        return Err(deep_learning_error(
            "dlgradient",
            "dlgradient: expected at least one dlarray, learnables, or dlnetwork target",
        ));
    }
    let requested = requested_outputs.unwrap_or(1);
    if requested > rest.len() {
        return Err(deep_learning_error(
            "dlgradient",
            format!(
                "dlgradient: requested {requested} outputs for {} gradient targets",
                rest.len()
            ),
        ));
    }
    validate_tape_dlarray_data(&loss)?;
    let Some(loss_node) = dlarray_node_id(&loss) else {
        return Err(deep_learning_error(
            "dlgradient",
            "dlgradient: loss must be a traced dlarray value produced inside dlfeval",
        ));
    };
    let loss_value = super::tape::node_value(loss_node)?;
    if loss_value
        .shape()
        .iter()
        .try_fold(1usize, |n, &v| n.checked_mul(v))
        != Some(1)
    {
        return Err(deep_learning_error(
            "dlgradient",
            "dlgradient: loss must be scalar",
        ));
    }
    let mut requested_nodes = Vec::new();
    for target in rest.iter().take(requested) {
        requested_nodes.extend(targets::node_ids(target)?);
    }
    resident::admit_cancellation(runtime)?;
    let host_gradients;
    let resident_gradients;
    let gradients = match loss_value {
        super::tape::NodeValue::Host(_) => {
            host_gradients = backward(loss_node)?;
            GradientSource::Host(&host_gradients)
        }
        super::tape::NodeValue::Resident(_) => {
            resident_gradients = resident::backward(runtime, loss_node, &requested_nodes).await?;
            GradientSource::Resident(&resident_gradients)
        }
    };
    let mut outputs = Vec::with_capacity(requested);
    resident::admit_cancellation(runtime)?;
    for target in rest.iter().take(requested) {
        outputs.push(gradient_tree_for_target_impl(target, &gradients)?);
        resident::admit_cancellation(runtime)?;
    }
    if requested == 1 {
        Ok(outputs.remove(0))
    } else {
        Ok(Value::output_list(outputs))
    }
}

mod backward;
mod targets;
use backward::backward;

#[cfg(test)]
mod tests;

#[cfg(test)]
pub(super) fn gradient_tree_for_target(
    target: &Value,
    gradients: &HashMap<usize, Tensor>,
) -> BuiltinResult<Value> {
    gradient_tree_for_target_impl(target, &GradientSource::Host(gradients))
}

fn gradient_tree_for_target_impl(
    target: &Value,
    gradients: &GradientSource<'_>,
) -> BuiltinResult<Value> {
    let context = crate::context::legacy::active();
    match target {
        Value::Object(object) if object.class_name().is(DLARRAY_CLASS) => {
            validate_tape_dlarray_data(target)?;
            let Some(id) = dlarray_node_id(target) else {
                return Err(deep_learning_error(
                    "dlgradient",
                    "dlgradient: target dlarray was not traced inside the active dlfeval",
                ));
            };
            let data = match gradients {
                GradientSource::Host(values) => Value::Tensor(if let Some(tensor) = values.get(&id).cloned() {
                    tensor
                } else {
                    let target_tensor = host_tensor_from_value(target, "dlgradient", "target")?;
                    floating_tensor_from_f64(
                        vec![0.0; tensor::tensor_element_len(&target_tensor)],
                        target_tensor.shape().to_vec(),
                        target_tensor.numeric_dtype(),
                        "dlgradient",
                    )?
                }),
                GradientSource::Resident(values) => {
                    if let Some(handle) = values.get(&id) {
                        Value::GpuTensor(handle.clone())
                    } else {
                        zero_for_unconnected_resident_target(target)?
                    }
                }
            };
            let (format, labels) = dlarray_format_and_labels(target);
            Ok(super::super::object_with_identity(
                DLARRAY_CLASS,
                vec![
                    ("Data", data),
                    ("Format", format),
                    ("Labels", labels),
                ],
            )?)
        }
        Value::Object(object) if model::is_deep_learning_network_object(object) => {
            let Value::Object(table) = model::project_learnables(object, "dlgradient")? else {
                unreachable!("network learnables projection is a table")
            };
            gradient_network_table(&table, gradients)
        }
        Value::Object(object) if crate::builtins::table::is_tabular_object(object) => {
            if parameter_table::values(object, "dlgradient")?.is_some() {
                return gradient_network_table(object, gradients);
            }
            let variables = crate::builtins::table::table_variables(object)
                .map_err(|err| deep_learning_error("dlgradient", err.to_string()))?;
let prepared_updated = crate::execution::structure::replacement::record(&variables, context.as_ref())?;
            let _roots = prepared_updated.1;
            let mut updated = prepared_updated.0;
            for value in updated.fields.try_values_mut_unique().expect("private destination") {
                *value = gradient_tree_for_target_impl(value, gradients)?;
            }
            crate::builtins::table::table_replace_variables_like(object, updated)
                .map_err(|err| deep_learning_error("dlgradient", err.to_string()))
        }
        Value::Struct(st) => {
            if is_learnables_struct(st) {
let prepared_out = crate::execution::structure::replacement::record(st, context.as_ref())?;
                let _roots = prepared_out.1;
                let mut out = prepared_out.0;
                let value = st.fields.get(Column::Value.name()).ok_or_else(|| {
                    deep_learning_error("dlgradient", "dlgradient: learnables tree is missing Value")
                })?;
                out.try_replace_unique(Column::Value.name(), gradient_tree_for_target_impl(value, gradients)?)
                    .expect("learnables schema contains Value");
                return Ok(Value::Struct(out));
            }
let prepared_out = crate::execution::structure::replacement::record(st, context.as_ref())?;
            let _roots = prepared_out.1;
            let mut out = prepared_out.0;
            for value in out.fields.try_values_mut_unique().expect("private destination") {
                *value = gradient_tree_for_target_impl(value, gradients)?;
            }
            Ok(Value::Struct(out))
        }
        Value::StructArray(array) => {
let prepared_out = crate::execution::structure::replacement::array(array, context.as_ref())?;
            let _roots = prepared_out.1;
            let mut out = prepared_out.0;
            for name in array.field_names() {
                for value in out.try_column_mut_unique(name).expect("private destination") {
                    *value = gradient_tree_for_target_impl(value, gradients)?;
                }
            }
            Ok(Value::StructArray(out))
        }
        Value::Cell(cell) => {
let prepared_out = crate::execution::structure::replacement::cell(cell, context.as_ref())?;
            let _roots = prepared_out.1;
            let mut out = prepared_out.0;
            for value in out.try_data_mut_unique().expect("private destination") {
                *value = gradient_tree_for_target_impl(value, gradients)?;
            }
            Ok(Value::Cell(out))
        }
        other => Err(deep_learning_error(
            "dlgradient",
            format!("dlgradient: target must be a traced dlarray, learnables tree, or dlnetwork, got {other:?}"),
        )),
    }
}

fn gradient_network_table(
    table: &ObjectInstance,
    gradients: &GradientSource<'_>,
) -> BuiltinResult<Value> {
    let values = parameter_table::values(table, "dlgradient")?.expect("parameter table schema");
    let gradients = gradient_tree_for_target_impl(&Value::Cell(values.clone()), gradients)?;
    let variables = crate::builtins::table::table_variables(table)
        .map_err(|error| deep_learning_error("dlgradient", error.to_string()))?;
    let context = crate::context::legacy::active();
    let gradients = crate::execution::structure::replacement::root(gradients, context.as_ref())?;
    let prepared_variables =
        crate::execution::structure::replacement::record(&variables, context.as_ref())?;
    let _roots = prepared_variables.1;
    let mut variables = prepared_variables.0;
    variables
        .try_replace_unique("Value", gradients.value().clone())
        .expect("learnables schema contains Value");
    crate::builtins::table::table_replace_variables_like(table, variables)
        .map_err(|error| deep_learning_error("dlgradient", error.to_string()))
}

fn zero_for_unconnected_resident_target(target: &Value) -> BuiltinResult<Value> {
    let Value::Object(object) = target else {
        unreachable!("gradient target admission requires a dlarray")
    };
    match object.properties().get("Data") {
        Some(Value::GpuTensor(handle)) => {
            let provider = crate::builtins::common::gpu_helpers::exact_provider_for_handle(handle)
                .ok_or_else(|| {
                    crate::builtins::common::gpu_helpers::ResidentComputeDecline::InvalidOwner
                        .with_domain_error(deep_learning_error(
                            "dlgradient",
                            "dlgradient: target resident owner retired",
                        ))
                })?;
            let mut zero = provider
                .fill_like(handle, 0.0)
                .map_err(|error| {
                    crate::runtime_error::with_provider_execution_source(
                        deep_learning_error(
                            "dlgradient",
                            "dlgradient: resident zero allocation failed",
                        ),
                        error.with_unproven_effects(),
                    )
                })?
                .ok_or_else(|| {
                    let mut error = deep_learning_error(
                        "dlgradient",
                        "dlgradient: resident zero constructor unavailable",
                    );
                    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
                    error
                })?;
            use crate::builtins::common::gpu_helpers::{
                reject_gpu_output, unary_gpu_output_matches, GpuOutputAliasPolicy,
                UnaryGpuOutputContract,
            };
            if !unary_gpu_output_matches(
                &zero,
                handle,
                provider,
                UnaryGpuOutputContract {
                    storage: runmat_accelerate_api::GpuTensorStorage::Real,
                    precision: runmat_accelerate_api::handle_precision(handle),
                    integer: None,
                    logical: false,
                    alias: GpuOutputAliasPolicy::RequireDistinctAllocation,
                },
            ) {
                return Err(reject_gpu_output(
                    zero,
                    deep_learning_error(
                        "dlgradient",
                        "dlgradient: invalid resident zero descriptor",
                    ),
                ));
            }
            crate::builtins::common::gpu_helpers::propagate_output_provenance(
                &mut zero,
                std::iter::once(handle),
            )?;
            Ok(Value::GpuTensor(zero))
        }
        _ => {
            let tensor = host_tensor_from_value(target, "dlgradient", "target")?;
            Ok(Value::Tensor(floating_tensor_from_f64(
                vec![0.0; tensor::tensor_element_len(&tensor)],
                tensor.shape().to_vec(),
                tensor.numeric_dtype(),
                "dlgradient",
            )?))
        }
    }
}

fn validate_tape_dlarray_data(value: &Value) -> BuiltinResult<()> {
    let Value::Object(object) = value else {
        return Ok(());
    };
    if object.class_name().is(DLARRAY_CLASS) {
        match object.properties().get("Data") {
            Some(Value::GpuTensor(handle))
                if runmat_accelerate_api::handle_storage(handle)
                    == runmat_accelerate_api::GpuTensorStorage::Real
                    && runmat_accelerate_api::handle_precision(handle).is_some() => {}
            Some(Value::SparseTensor(_)) => {
                return Err(deep_learning_error(
                    "dlgradient",
                    "dlgradient: sparse dlarray Data are not supported by the host tape",
                ));
            }
            Some(Value::Num(_) | Value::Bool(_) | Value::LogicalArray(_)) => {}
            Some(Value::Tensor(tensor))
                if matches!(
                    tensor.numeric_dtype(),
                    NumericDType::F64 | NumericDType::F32
                ) => {}
            Some(_) => {
                return Err(deep_learning_error(
                    "dlgradient",
                    "dlgradient: dlarray Data must be real double, single, or logical numerical storage",
                ));
            }
            None => {
                return Err(deep_learning_error(
                    "dlgradient",
                    "dlgradient: traced dlarray is missing Data",
                ));
            }
        }
    }
    Ok(())
}

fn is_learnables_struct(value: &StructValue) -> bool {
    Column::ALL
        .iter()
        .all(|column| value.fields.contains_key(column.name()))
}
