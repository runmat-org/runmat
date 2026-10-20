use super::*;
pub(super) async fn parse_args(
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<ParsedExprnd> {
    if args.is_empty() {
        return Err(exprnd_error_with(
            &EXPRND_ERROR_INVALID_ARGUMENT,
            "exprnd: requires at least one argument (mu)",
        ));
    }
    let output_precision = output_precision(&args[0]);
    let borrowed_gpu = args
        .iter()
        .filter_map(|arg| match arg {
            Value::GpuTensor(handle) => Some(handle.clone()),
            _ => None,
        })
        .collect();
    let gpu_source = match &args[0] {
        Value::GpuTensor(handle) => Some(handle.clone()),
        _ => None,
    };
    let gathered = gather_if_needed_async(&args[0]).await.map_err(|err| {
        exprnd_error_with(&EXPRND_ERROR_INVALID_ARGUMENT, format!("exprnd: {err}"))
    })?;
    let mu_tensor = value_to_mu_tensor(gathered)?;
    let mu_shape = normalize_shape(mu_tensor.shape().to_vec());
    let mu = tensor::tensor_into_values_f64(mu_tensor);
    let shape = if args.len() == 1 {
        mu_shape.clone()
    } else {
        parse_shape_args(args.view().range(1..)?).await?
    };
    if mu.len() != 1 && mu_shape != shape {
        return Err(exprnd_error_with(
            &EXPRND_ERROR_INVALID_ARGUMENT,
            "exprnd: requested size must match nonscalar mu",
        ));
    }
    Ok(ParsedExprnd {
        mu,
        shape,
        output_precision,
        gpu_source,
        borrowed_gpu,
    })
}

async fn parse_shape_args(
    rest: crate::call::FunctionArgumentView<'_>,
) -> crate::BuiltinResult<Vec<usize>> {
    let mut dims = Vec::new();
    for value in rest {
        let gathered = gather_if_needed_async(value).await.map_err(|err| {
            exprnd_error_with(&EXPRND_ERROR_INVALID_ARGUMENT, format!("exprnd: {err}"))
        })?;
        dims.extend(dimensions_from_host_value(&gathered)?);
    }
    Ok(normalize_dims(dims))
}

pub(super) fn normalize_dims(dims: Vec<usize>) -> Vec<usize> {
    if dims.is_empty() {
        vec![0, 0]
    } else if dims.len() == 1 {
        vec![dims[0], dims[0]]
    } else {
        normalize_shape(dims)
    }
}

pub(super) fn normalize_shape(mut shape: Vec<usize>) -> Vec<usize> {
    if shape.is_empty() {
        shape = vec![1, 1];
    } else if shape.len() == 1 {
        shape.push(1);
    }
    while shape.len() > 2 && shape.last() == Some(&1) {
        shape.pop();
    }
    shape
}

pub(super) fn value_to_mu_tensor(value: Value) -> BuiltinResult<Tensor> {
    let result = match value {
        Value::Num(value) => Tensor::new(vec![value], vec![1, 1]),
        Value::Int(value) => Tensor::new(vec![value.to_f64()], vec![1, 1]),
        Value::Tensor(tensor) => return Ok(tensor),
        other => {
            return Err(exprnd_error_with(
                &EXPRND_ERROR_INVALID_ARGUMENT,
                format!("exprnd: expected real single or double mu, got {other:?}"),
            ))
        }
    };
    result.map_err(|err| exprnd_internal_error(format!("exprnd: {err}")))
}

pub(super) fn reject_unsupported_classes(
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<()> {
    let Some(mu) = args.first() else {
        return Ok(());
    };
    let invalid = |value: &Value| {
        matches!(
            value,
            Value::Bool(_)
                | Value::LogicalArray(_)
                | Value::Complex(_, _)
                | Value::ComplexTensor(_)
        ) || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle) || runmat_accelerate_api::handle_storage(handle) == GpuTensorStorage::ComplexInterleaved)
    };
    if invalid(mu) {
        return Err(exprnd_error_with(
            &EXPRND_ERROR_INVALID_ARGUMENT,
            "exprnd: mu must be a real single or double array",
        ));
    }
    if args.iter().skip(1).any(invalid) {
        return Err(exprnd_error_with(
            &EXPRND_ERROR_INVALID_ARGUMENT,
            "exprnd: size arguments must be real integer-valued arrays",
        ));
    }
    Ok(())
}

pub(super) fn is_typed_integer_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())
}

pub(super) fn dimensions_from_host_value(value: &Value) -> BuiltinResult<Vec<usize>> {
    match value {
        Value::Num(value) => Ok(vec![floating_dimension(*value)?]),
        Value::Int(value) => Ok(vec![integer_dimension(value)?]),
        Value::Tensor(tensor) => {
            let vector = tensor.len() <= 1
                || tensor.shape().iter().filter(|extent| **extent != 1).count() <= 1;
            if !vector {
                return Err(exprnd_error_with(
                    &EXPRND_ERROR_INVALID_ARGUMENT,
                    "exprnd: size must be a scalar or vector",
                ));
            }
            if let Some(storage) = tensor.integer_storage() {
                return (0..storage.len())
                    .map(|index| {
                        integer_dimension(&storage.value_at(index).expect("integer size value"))
                    })
                    .collect();
            }
            tensor
                .materialize_f64()
                .iter()
                .map(|value| floating_dimension(*value))
                .collect()
        }
        other => Err(exprnd_error_with(
            &EXPRND_ERROR_INVALID_ARGUMENT,
            format!("exprnd: invalid size argument: {other:?}"),
        )),
    }
}

pub(super) fn integer_dimension(value: &IntValue) -> BuiltinResult<usize> {
    if value.to_f64() < 0.0 {
        return Ok(0);
    }
    value.try_to_usize().ok_or_else(|| {
        exprnd_error_with(
            &EXPRND_ERROR_INVALID_ARGUMENT,
            "exprnd: size is outside the supported range",
        )
    })
}

pub(super) fn floating_dimension(value: f64) -> BuiltinResult<usize> {
    if !value.is_finite() || value.fract() != 0.0 {
        return Err(exprnd_error_with(
            &EXPRND_ERROR_INVALID_ARGUMENT,
            "exprnd: dimensions must be finite integers",
        ));
    }
    if value <= 0.0 {
        return Ok(0);
    }
    if value >= usize::MAX as f64 {
        return Err(exprnd_error_with(
            &EXPRND_ERROR_INVALID_ARGUMENT,
            "exprnd: size is outside the supported range",
        ));
    }
    Ok(value as usize)
}
