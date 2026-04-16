#[cfg(feature = "native-accel")]
use runmat_accelerate::fusion_residency;
use runmat_builtins::Value;
use runmat_runtime::RuntimeError;

pub async fn execute_stochastic_evolution(
    state: Value,
    drift: Value,
    scale: Value,
    steps: Value,
) -> Result<Value, RuntimeError> {
    let steps_u32 = parse_steps_value(&steps).await?;
    if steps_u32 == 0 {
        return Ok(state);
    }

    #[cfg(feature = "native-accel")]
    {
        if let Some(provider) = runmat_accelerate_api::provider() {
            let (state_handle, state_owned) =
                ensure_gpu_tensor_for_stochastic(provider, &state).await?;
            let drift_scalar =
                scalar_from_value_scalar(&drift, "stochastic_evolution drift").await?;
            let scale_scalar =
                scalar_from_value_scalar(&scale, "stochastic_evolution scale").await?;
            match provider.stochastic_evolution(
                &state_handle,
                drift_scalar,
                scale_scalar,
                steps_u32,
            ) {
                Ok(output) => {
                    if let Some(temp) = state_owned {
                        let _ = provider.free(&temp);
                    }
                    fusion_residency::mark(&output);
                    return Ok(Value::GpuTensor(output));
                }
                Err(err) => {
                    log::debug!("stochastic_evolution provider fallback to host: {}", err);
                    if let Some(temp) = state_owned {
                        let _ = provider.free(&temp);
                    }
                }
            }
        }
    }

    let gathered_state = runmat_runtime::dispatcher::gather_if_needed_async(&state)
        .await
        .map_err(|e| format!("stochastic_evolution: {e}"))?;
    let mut tensor_value = match gathered_state {
        Value::Tensor(t) => t,
        other => runmat_runtime::builtins::common::tensor::value_into_tensor_for(
            "stochastic_evolution",
            other,
        )?,
    };
    let drift_scalar = scalar_from_value_scalar(&drift, "stochastic_evolution drift").await?;
    let scale_scalar = scalar_from_value_scalar(&scale, "stochastic_evolution scale").await?;
    runmat_runtime::builtins::stats::random::stochastic_evolution::stochastic_evolution_host(
        &mut tensor_value,
        drift_scalar,
        scale_scalar,
        steps_u32,
    )?;
    Ok(Value::Tensor(tensor_value))
}

async fn scalar_from_value_scalar(value: &Value, label: &str) -> Result<f64, RuntimeError> {
    match value {
        Value::Num(n) => Ok(*n),
        Value::Int(i) => Ok(i.to_f64()),
        Value::Tensor(t) if t.data.len() == 1 => Ok(t.data[0]),
        Value::Tensor(t) => Err(format!(
            "{label}: expected scalar tensor, got {} elements",
            t.data.len()
        )
        .into()),
        Value::GpuTensor(_) => {
            let gathered = runmat_runtime::dispatcher::gather_if_needed_async(value)
                .await
                .map_err(|e| format!("{label}: {e}"))?;
            match gathered {
                Value::Num(n) => Ok(n),
                Value::Int(i) => Ok(i.to_f64()),
                Value::Tensor(t) if t.data.len() == 1 => Ok(t.data[0]),
                Value::Tensor(t) => Err(format!(
                    "{label}: expected scalar tensor, got {} elements",
                    t.data.len()
                )
                .into()),
                other => Err(format!("{label}: expected numeric scalar, got {:?}", other).into()),
            }
        }
        other => Err(format!("{label}: expected numeric scalar, got {:?}", other).into()),
    }
}

async fn parse_steps_value(value: &Value) -> Result<u32, RuntimeError> {
    let raw = scalar_from_value_scalar(value, "stochastic_evolution steps").await?;
    if !raw.is_finite() || raw < 0.0 {
        return Err(crate::interpreter::errors::mex(
            "InvalidSteps",
            "stochastic_evolution: steps must be a non-negative scalar",
        ));
    }
    Ok(raw.round() as u32)
}

#[cfg(feature = "native-accel")]
async fn ensure_gpu_tensor_for_stochastic(
    provider: &dyn runmat_accelerate_api::AccelProvider,
    value: &Value,
) -> Result<
    (
        runmat_accelerate_api::GpuTensorHandle,
        Option<runmat_accelerate_api::GpuTensorHandle>,
    ),
    RuntimeError,
> {
    match value {
        Value::GpuTensor(handle) => Ok((handle.clone(), None)),
        Value::Tensor(tensor) => {
            let handle = upload_tensor_view(provider, tensor)?;
            Ok((handle.clone(), Some(handle)))
        }
        _ => {
            let gathered = runmat_runtime::dispatcher::gather_if_needed_async(value)
                .await
                .map_err(|e| format!("stochastic_evolution: {e}"))?;
            match gathered {
                Value::Tensor(t) => {
                    let handle = upload_tensor_view(provider, &t)?;
                    Ok((handle.clone(), Some(handle)))
                }
                other => {
                    let tensor = runmat_runtime::builtins::common::tensor::value_into_tensor_for(
                        "stochastic_evolution",
                        other,
                    )?;
                    let handle = upload_tensor_view(provider, &tensor)?;
                    Ok((handle.clone(), Some(handle)))
                }
            }
        }
    }
}

#[cfg(feature = "native-accel")]
fn upload_tensor_view(
    provider: &dyn runmat_accelerate_api::AccelProvider,
    tensor: &runmat_builtins::Tensor,
) -> Result<runmat_accelerate_api::GpuTensorHandle, RuntimeError> {
    let view = runmat_accelerate_api::HostTensorView {
        data: &tensor.data,
        shape: &tensor.shape,
    };
    provider
        .upload(&view)
        .map_err(|e| crate::interpreter::errors::mex("UploadFailed", &e.to_string()))
}
