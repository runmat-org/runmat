//! Scatterhist admission domain behavior.
use super::*;
use runmat_accelerate_api::GpuTensorStorage;

pub(super) fn admit_data_compatibility(
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<()> {
    if args.iter().any(crate::value_contains_gpu) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &SCATTERHIST_GPU_INPUT_EXTENSION,
            NAME,
        )?;
    }
    if args
        .iter()
        .any(crate::builtins::common::validation::value_has_native_integer_class)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &SCATTERHIST_INTEGER_DATA_EXTENSION,
            NAME,
        )?;
    }
    Ok(())
}

pub(super) fn admit_host_visible(
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<usize> {
    let mut lengths = [0usize; 2];
    for (index, value) in args.iter().take(2).enumerate() {
        let name = if index == 0 { "x" } else { "y" };
        let (shape, count): (Option<&[usize]>, usize) = match value {
            Value::Num(_) | Value::Int(_) | Value::Bool(_) => (None, 1),
            Value::Tensor(tensor) => (Some(tensor.shape()), tensor.len()),
            Value::LogicalArray(array) => (Some(&array.shape), array.len()),
            Value::GpuTensor(handle) => {
                if handle.physical().storage() != GpuTensorStorage::Real {
                    return Err(invalid(format!(
                        "scatterhist: {name} must be real numeric data"
                    )));
                }
                let count = handle
                    .shape()
                    .iter()
                    .try_fold(1usize, |n, &dim| n.checked_mul(dim))
                    .ok_or_else(|| invalid("scatterhist: observation shape overflows"))?;
                (Some(handle.shape()), count)
            }
            _ => return Err(invalid(format!("scatterhist: {name} must be numeric"))),
        };
        if shape.is_some_and(|shape| shape.iter().filter(|&&dim| dim > 1).count() > 1) {
            return Err(invalid(format!("scatterhist: {name} must be a vector")));
        }
        if !crate::value_contains_gpu(value)
            && crate::builtins::common::validation::value_has_native_integer_class(value)
            && !crate::builtins::common::validation::native_integer_value_is_exact_f64(value)
        {
            return Err(inexact_observation());
        }
        lengths[index] = count;
    }
    if lengths[0] != lengths[1] {
        return Err(invalid(
            "scatterhist: x and y must contain the same number of elements",
        ));
    }
    if lengths[0] == 0 {
        return Err(invalid("scatterhist: x and y must not be empty"));
    }
    Ok(lengths[0])
}

fn inexact_observation() -> RuntimeError {
    build_runtime_error(
        "scatterhist: integer observation values must be exactly representable as double",
    )
    .with_builtin(NAME)
    .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
    .build()
}

pub(super) async fn gather_values(
    args: crate::call::BuiltinArguments,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<crate::call::BuiltinArguments> {
    args.try_map_preserving_roots_async(args.len(), runtime, async |index| {
        let value = &args[index];
        if !crate::value_contains_gpu(value) {
            return Ok(value.clone());
        }
        let resident_integer = matches!(value, Value::GpuTensor(_))
            && crate::builtins::common::validation::value_has_native_integer_class(value);
        let gathered = gather_if_needed_async(value)
            .await
            .map_err(|err| invalid_with_source(format!("scatterhist: {err}"), err))?;
        if resident_integer
            && !crate::builtins::common::validation::native_integer_value_is_exact_f64(&gathered)
        {
            return Err(inexact_observation());
        }
        Ok(gathered)
    })
    .await
}

pub(super) fn parse_xy(
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<(Vec<f64>, Vec<f64>)> {
    let x = numeric_vector(args[0].clone(), "x")?;
    let y = numeric_vector(args[1].clone(), "y")?;
    if x.len() != y.len() {
        return Err(invalid(
            "scatterhist: x and y must contain the same number of elements",
        ));
    }
    if x.is_empty() {
        return Err(invalid("scatterhist: x and y must not be empty"));
    }
    Ok((x, y))
}

pub(super) fn numeric_vector(value: Value, name: &str) -> BuiltinResult<Vec<f64>> {
    let tensor = tensor::value_into_tensor_for(NAME, value)
        .map_err(|_| invalid(format!("scatterhist: {name} must be numeric")))?;
    if tensor
        .shape()
        .iter()
        .copied()
        .filter(|dim| *dim > 1)
        .count()
        > 1
    {
        return Err(invalid(format!("scatterhist: {name} must be a vector")));
    }
    Ok(tensor_values_f64(&tensor))
}
