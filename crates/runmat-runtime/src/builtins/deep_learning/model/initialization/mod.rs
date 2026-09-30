//! One-shot network initialization. Structural admission precedes all random
//! draws and user callbacks; generated values are published only as a set.
use super::parameters::{ParameterRole, ParameterShape};
use super::*;

mod generate;
mod plan;

#[cfg(test)]
mod tests;

struct StagedParameter {
    layer_index: usize,
    role: ParameterRole,
    value: Value,
    generated: bool,
    existing_precision: Option<NumericDType>,
}

pub(in crate::builtins) async fn initialise_layers(
    layers: &mut [Value],
    function: &'static str,
) -> BuiltinResult<()> {
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
            plan::Initializer::Builtin(kind) => generate::builtin(
                kind,
                task.kind,
                &task.shape,
                task.precision.unwrap_or(NumericDType::F64),
                function,
            )?,
            plan::Initializer::Callback(callback) => {
                let size = tensor_value(
                    task.shape.iter().map(|size| *size as f64).collect(),
                    vec![1, task.shape.len()],
                    function,
                )?;
                let result = crate::call_feval_async_with_outputs(callback, &[size], 1)
                    .await
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
                match result {
                    Value::OutputList(mut values) if values.len() == 1 => values.remove(0),
                    Value::OutputList(values) => {
                        return Err(deep_learning_error(
                            function,
                            format!(
                                "{function}: {} initializer returned {} outputs, expected one",
                                task.role.name(),
                                values.len()
                            ),
                        ));
                    }
                    value => value,
                }
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
            let Value::Tensor(tensor) = &staged.value else {
                unreachable!("built-in initializer produced a host tensor")
            };
            staged.value = Tensor::new_with_dtype(
                tensor.materialize_f64(),
                tensor.shape.clone(),
                NumericDType::F32,
            )
            .map(Value::Tensor)
            .map_err(|error| deep_learning_error(function, error))?;
        }
    }
    for staged in replacements {
        let Value::Object(layer) = &mut layers[staged.layer_index] else {
            unreachable!("preflight established layer objects")
        };
        layer
            .properties
            .insert(staged.role.name().into(), staged.value);
    }
    Ok(())
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
