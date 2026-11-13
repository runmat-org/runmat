//! One-shot network initialization. Structural admission precedes all random
//! draws and user callbacks; generated values are published only as a set.
use super::parameters::{ParameterRole, ParameterShape};
use super::*;
use crate::builtins::deep_learning::arguments::DeepLearningObjectEdit as _;
use crate::execution::{RootedValue, RootedValueLease};

mod generate;
mod plan;

#[cfg(test)]
mod tests;

struct StagedParameter {
    layer_index: usize,
    role: ParameterRole,
    value: RootedValue,
    generated: bool,
    existing_precision: Option<NumericDType>,
}

#[cfg(test)]
pub(in crate::builtins) async fn initialise_layers(
    layers: &mut [Value],
    function: &'static str,
) -> BuiltinResult<RootedValueLease> {
    initialise_layers_in_runtime(None, layers, function).await
}

pub(in crate::builtins) async fn initialise_layers_in_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
    layers: &mut [Value],
    function: &'static str,
) -> BuiltinResult<RootedValueLease> {
    let active = crate::context::legacy::active();
    let runtime = runtime.or(active.as_ref());
    let _source_roots = match runtime {
        Some(runtime) => RootedValueLease::for_values_in_context(layers, runtime)?,
        None => {
            require_ordinary_roots()?;
            RootedValueLease::for_values(layers.iter())
                .map_err(|error| error.into_runtime_error())?
        }
    };
    let tasks = plan::tasks(layers, function)?;
    let mut replacements = Vec::new();
    replacements.try_reserve_exact(tasks.len()).map_err(|_| {
        deep_learning_error(
            function,
            format!("{function}: cannot reserve initialized parameters"),
        )
    })?;
    for task in tasks {
        let generated = matches!(&task.initializer, plan::Initializer::Builtin(_));
        let value = match task.initializer {
            plan::Initializer::Builtin(kind) => {
                let mut turn =
                    crate::builtins::common::random::acquire_turn(runtime, function).await?;
                RootedValue::new(generate::builtin(
                    &mut turn,
                    kind,
                    task.kind,
                    &task.shape,
                    task.precision.unwrap_or(NumericDType::F64),
                    function,
                )?)
                .map_err(|error| error.into_runtime_error())?
            }
            plan::Initializer::Callback(callback) => {
                let size = tensor_value(
                    task.shape.iter().map(|size| *size as f64).collect(),
                    vec![1, task.shape.len()],
                    function,
                )?;
                let result = match runtime {
                    Some(runtime) => {
                        crate::call_feval_async_with_outputs_in_context(
                            runtime,
                            callback,
                            &[size],
                            1,
                        )
                        .await
                    }
                    None => crate::call_feval_async_with_outputs(callback, &[size], 1).await,
                }
                .map_err(|source| {
                    crate::build_runtime_error(format!(
                        "{function}: initializer callback failed for {}",
                        task.role.name()
                    ))
                    .with_builtin(function)
                    .with_identifier("RunMat:deepLearning:InvalidInput")
                    .with_source(source)
                    .build()
                })?;
                result.try_map_owned(
                    |result| match result {
                        Value::OutputList(values) if values.len() == 1 => {
                            Ok(crate::sequence::into_first_output(values)?
                                .expect("one initializer output"))
                        }
                        Value::OutputList(values) => Err(deep_learning_error(
                            function,
                            format!(
                                "{function}: {} initializer returned {} outputs, expected one",
                                task.role.name(),
                                values.len()
                            ),
                        )),
                        value => Ok(value),
                    },
                    |error| error.into_runtime_error(),
                )?
            }
        };
        parameters::validate_initialized_parameter(
            &value,
            &task.expected_shape,
            task.role,
            function,
        )?;
        if parameters::numeric_payload_shape(&value, task.role.name(), function)?.contains(&0) {
            return Err(deep_learning_error(
                function,
                format!(
                    "{function}: {} initializer returned an empty parameter",
                    task.role.name()
                ),
            ));
        }
        replacements.push(StagedParameter {
            layer_index: task.layer_index,
            role: task.role,
            value,
            generated,
            existing_precision: task.precision,
        });
    }
    let mut callback_precision = vec![None; layers.len()];
    for staged in &replacements {
        if staged.generated {
            continue;
        }
        if let Some(dtype @ (NumericDType::F32 | NumericDType::F64)) =
            parameters::floating_precision(&staged.value)
        {
            let prior = &mut callback_precision[staged.layer_index];
            if staged
                .existing_precision
                .is_some_and(|known| known != dtype)
                || prior.is_some_and(|known| known != dtype)
            {
                return Err(deep_learning_error(function, format!(
                    "{function}: initializer callbacks returned mixed learnable precision for one layer"
                )));
            }
            *prior = Some(dtype);
        }
    }
    for staged in &mut replacements {
        if staged.generated
            && staged.existing_precision.is_none()
            && callback_precision[staged.layer_index] == Some(NumericDType::F32)
        {
            let Value::Tensor(tensor) = staged.value.value() else {
                unreachable!("built-in initializer produced a host tensor")
            };
            staged.value = RootedValue::new(
                Tensor::new_with_dtype(
                    tensor.materialize_f64(),
                    tensor.shape().to_vec(),
                    NumericDType::F32,
                )
                .map(Value::Tensor)
                .map_err(|error| deep_learning_error(function, error))?,
            )
            .map_err(|error| error.into_runtime_error())?;
        }
    }
    publish_staged_parameters(runtime, layers, replacements)
}

fn publish_staged_parameters(
    runtime: Option<&crate::context::RuntimeContext>,
    layers: &mut [Value],
    replacements: Vec<StagedParameter>,
) -> BuiltinResult<RootedValueLease> {
    // Register the complete old/new graph before the first mutation. This
    // registration cannot fail after a partial publication, and it survives
    // each staged owner's retirement until the caller roots the final network.
    let inputs = layers
        .iter()
        .chain(replacements.iter().map(|staged| staged.value.value()));
    let publication_roots = match runtime {
        Some(runtime) => RootedValueLease::for_values_iterator_in_context(inputs, runtime)?,
        None => {
            require_ordinary_roots()?;
            RootedValueLease::for_values(inputs).map_err(|error| error.into_runtime_error())?
        }
    };
    for staged in replacements {
        let Value::Object(layer) = &mut layers[staged.layer_index] else {
            unreachable!("preflight established layer objects")
        };
        staged
            .value
            .with_owned(|value| layer.replace_dl_property(staged.role.name(), value))?;
    }
    Ok(publication_roots)
}

fn require_ordinary_roots() -> BuiltinResult<()> {
    use crate::context::host_allocation::{
        host_allocation_runtime_error, HostAllocationError, HostAllocationPolicy,
        HostAllocationReceipt,
    };
    let error = |source| {
        host_allocation_runtime_error(
            None,
            "RunMat:deepLearning:InitializationRoots",
            "initializer root policy is unavailable",
            source,
        )
    };
    match HostAllocationReceipt::policy(None).map_err(error)? {
        HostAllocationPolicy::Unbounded => Ok(()),
        HostAllocationPolicy::Managed => Err(error(HostAllocationError::MissingManagedOwner)),
    }
}

/// Reuse the initializer's effect-free structural plan when training needs
/// output geometry before the first random draw or user callback. Only small
/// shape metadata escapes; no parameter payload or initializer is run here.
pub(in crate::builtins::deep_learning) fn planned_parameter_shapes(
    layers: &[Value],
    function: &'static str,
) -> BuiltinResult<Vec<(usize, ParameterRole, Vec<usize>)>> {
    plan::planned_shapes(layers, function)
}
