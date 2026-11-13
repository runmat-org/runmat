use super::*;
use runmat_accelerate_api::ProviderPrecision;
use runmat_value::NumericDType;

pub(super) async fn try_pol2cart_gpu(
    inputs: &[Value],
    final_shape: &[usize],
    requested_outputs: Option<usize>,
) -> BuiltinResult<Option<RootedValue>> {
    let Some(anchor) = inputs.iter().find_map(|value| match value {
        Value::GpuTensor(handle) => Some(handle),
        _ => None,
    }) else {
        return Ok(None);
    };
    let provider = gpu_helpers::exact_provider_for_handle(anchor)
        .ok_or_else(|| pol2cart_error(&ERROR_INTERNAL, "GPU input has no owning provider"))?;
    let Some(precision) = runmat_accelerate_api::handle_precision(anchor) else {
        return crate::context::provider_operation::declined_provider_route(
            crate::context::legacy::active().as_ref(),
        );
    };
    // Mixed physical classes retain the host formula's individual unary and
    // multiplication precision. Never approximate integer storage on device.
    if inputs
        .iter()
        .any(|value| !same_float_class(value, provider, precision))
    {
        return crate::context::provider_operation::declined_provider_route(
            crate::context::legacy::active().as_ref(),
        );
    }
    let mut outputs = owned::Outputs::new(inputs);
    let mut operands = Vec::with_capacity(inputs.len());
    for input in inputs {
        let handle = match input {
            Value::GpuTensor(handle) => handle.clone(),
            value => {
                let mut tensor = value_into_real_tensor(value.clone())?;
                if tensor.integer_storage().is_some() {
                    // Admission proved every authored integer exactly representable.
                    tensor = Tensor::new(tensor.materialize_f64(), tensor.shape().to_vec())
                        .map_err(|error| pol2cart_error(&ERROR_INTERNAL, error))?;
                }
                let uploaded =
                    gpu_helpers::upload_tensor_protecting(provider, &tensor, &outputs.protected())
                        .map_err(upload_error)?;
                outputs.adopt(provider, uploaded, anchor, tensor.shape(), precision)?
            }
        };
        operands.push(handle);
    }
    let result = evaluate(
        provider,
        &operands,
        final_shape,
        requested_outputs,
        precision,
        &mut outputs,
    )
    .await;
    match result {
        Ok(handles) => {
            let values = handles
                .into_iter()
                .map(|handle| outputs.publish(handle))
                .collect::<BuiltinResult<Vec<_>>>()?;
            let published = if requested_outputs.is_none() {
                values
                    .into_iter()
                    .next()
                    .expect("implicit first coordinate")
            } else {
                Value::output_list(values)
            };
            // Register the exact coordinate/list graph before Outputs retires
            // its remaining temporary and borrowed allocation authorities.
            RootedValue::new(published)
                .map(Some)
                .map_err(|cause| cause.into_runtime_error())
        }
        Err(AttemptError::Unsupported) => Ok(None),
        Err(AttemptError::Failure(error)) => Err(*error),
    }
}

fn same_float_class(
    value: &Value,
    provider: &'static dyn AccelProvider,
    precision: ProviderPrecision,
) -> bool {
    match value {
        Value::GpuTensor(handle) => {
            gpu_helpers::exact_provider_for_handle(handle)
                .is_some_and(|owner| runmat_accelerate_api::same_provider_instance(owner, provider))
                && runmat_accelerate_api::handle_precision(handle) == Some(precision)
                && runmat_accelerate_api::handle_integer_type(handle).is_none()
        }
        Value::Num(_) | Value::Int(_) => precision == ProviderPrecision::F64,
        Value::Tensor(tensor) => match precision {
            ProviderPrecision::F64 => {
                tensor.numeric_dtype() == NumericDType::F64 || tensor.integer_storage().is_some()
            }
            ProviderPrecision::F32 => tensor.numeric_dtype() == NumericDType::F32,
        },
        _ => false,
    }
}

enum AttemptError {
    Unsupported,
    Failure(Box<RuntimeError>),
}

impl From<anyhow::Error> for AttemptError {
    fn from(error: anyhow::Error) -> Self {
        if gpu_helpers::provider_hook_allows_host_fallback(&error) {
            Self::Unsupported
        } else {
            let contextual = pol2cart_error(&ERROR_INTERNAL, &error);
            Self::Failure(Box::new(
                crate::runtime_error::with_provider_execution_source(contextual, error.into()),
            ))
        }
    }
}
impl From<runmat_accelerate_api::ProviderExecutionError> for AttemptError {
    fn from(error: runmat_accelerate_api::ProviderExecutionError) -> Self {
        if error.is_unsupported() {
            Self::Unsupported
        } else {
            let contextual = pol2cart_error(&ERROR_INTERNAL, error.to_string());
            Self::Failure(Box::new(
                crate::runtime_error::with_provider_execution_source(contextual, error),
            ))
        }
    }
}
impl From<RuntimeError> for AttemptError {
    fn from(error: RuntimeError) -> Self {
        Self::Failure(Box::new(error))
    }
}

async fn evaluate(
    provider: &'static dyn AccelProvider,
    operands: &[GpuTensorHandle],
    shape: &[usize],
    requested: Option<usize>,
    precision: ProviderPrecision,
    outputs: &mut owned::Outputs,
) -> Result<Vec<GpuTensorHandle>, AttemptError> {
    let theta = &operands[0];
    let rho = &operands[1];
    let pair_shape = matlab_broadcast_shape(theta.shape(), rho.shape())?;
    let cos = outputs.run_protected(provider.unary_cos(theta)).await?;
    let cos = outputs.adopt(provider, cos, theta, theta.shape(), precision)?;
    let sin = outputs.run_protected(provider.unary_sin(theta)).await?;
    let sin = outputs.adopt(provider, sin, theta, theta.shape(), precision)?;
    let x = outputs.run_protected(provider.elem_mul(rho, &cos)).await?;
    let mut x = outputs.adopt(provider, x, theta, &pair_shape, precision)?;
    let y = outputs.run_protected(provider.elem_mul(rho, &sin)).await?;
    let mut y = outputs.adopt(provider, y, theta, &pair_shape, precision)?;
    let count = requested.unwrap_or(1);
    let needs_xy_expansion = pair_shape != shape;
    let needs_z_expansion = count >= 3 && operands[2].shape() != shape;
    let ones = if needs_xy_expansion || needs_z_expansion {
        // Typed upload preserves physical precision even when the provider's
        // default fill precision differs from the inputs.
        let len = if shape.contains(&0) {
            0
        } else {
            shape.iter().product()
        };
        let tensor = match precision {
            ProviderPrecision::F32 => Tensor::from_f32(vec![1.0; len], shape.to_vec()),
            ProviderPrecision::F64 => Tensor::new(vec![1.0; len], shape.to_vec()),
        }
        .map_err(|error| pol2cart_error(&ERROR_INTERNAL, error))?;
        let handle = gpu_helpers::upload_tensor_protecting(provider, &tensor, &outputs.protected())
            .map_err(upload_error)?;
        Some(outputs.adopt(provider, handle, theta, shape, precision)?)
    } else {
        None
    };
    if needs_xy_expansion {
        let ones = ones.as_ref().expect("expansion multiplier");
        let expanded = outputs.run_protected(provider.elem_mul(&x, ones)).await?;
        x = outputs.adopt(provider, expanded, theta, shape, precision)?;
        let expanded = outputs.run_protected(provider.elem_mul(&y, ones)).await?;
        y = outputs.adopt(provider, expanded, theta, shape, precision)?;
    }
    let mut result = Vec::with_capacity(count);
    if count >= 1 {
        result.push(x);
    }
    if count >= 2 {
        result.push(y);
    }
    if count >= 3 {
        // Always construct a fresh z result: returning a borrowed input would
        // require a different ownership/alias contract.
        let z = &operands[2];
        let multiplier = match ones {
            Some(handle) => handle,
            None => {
                let tensor = match precision {
                    ProviderPrecision::F32 => Tensor::from_f32(vec![1.0], vec![1, 1]),
                    ProviderPrecision::F64 => Tensor::new(vec![1.0], vec![1, 1]),
                }
                .map_err(|error| pol2cart_error(&ERROR_INTERNAL, error))?;
                let handle =
                    gpu_helpers::upload_tensor_protecting(provider, &tensor, &outputs.protected())
                        .map_err(upload_error)?;
                outputs.adopt(provider, handle, theta, &[1, 1], precision)?
            }
        };
        let output = outputs
            .run_protected(provider.elem_mul(z, &multiplier))
            .await?;
        result.push(outputs.adopt(provider, output, theta, shape, precision)?);
    }
    Ok(result)
}

fn upload_error(error: RuntimeError) -> RuntimeError {
    let mut contextual = pol2cart_error(&ERROR_INTERNAL, error.message());
    contextual.source = Some(Box::new(error));
    contextual
}
