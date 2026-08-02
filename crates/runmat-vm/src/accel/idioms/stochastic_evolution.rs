#[cfg(feature = "native-accel")]
use runmat_accelerate::fusion_residency;
use runmat_builtins::Value;
use runmat_runtime::builtins::common::tensor::{scalar_integer_value, tensor_value_f64};
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
        Value::Tensor(t) if t.len() == 1 => Ok(tensor_value_f64(t, 0)),
        Value::Tensor(t) => {
            Err(format!("{label}: expected scalar tensor, got {} elements", t.len()).into())
        }
        Value::GpuTensor(_) => {
            let gathered = runmat_runtime::dispatcher::gather_if_needed_async(value)
                .await
                .map_err(|e| format!("{label}: {e}"))?;
            match gathered {
                Value::Num(n) => Ok(n),
                Value::Int(i) => Ok(i.to_f64()),
                Value::Tensor(t) if t.len() == 1 => Ok(tensor_value_f64(&t, 0)),
                Value::Tensor(t) => {
                    Err(format!("{label}: expected scalar tensor, got {} elements", t.len()).into())
                }
                other => Err(format!("{label}: expected numeric scalar, got {:?}", other).into()),
            }
        }
        other => Err(format!("{label}: expected numeric scalar, got {:?}", other).into()),
    }
}

async fn parse_steps_value(value: &Value) -> Result<u32, RuntimeError> {
    if let Some(result) = parse_integer_steps(value) {
        return result;
    }
    let gathered;
    let scalar_value = if matches!(value, Value::GpuTensor(_)) {
        gathered = runmat_runtime::dispatcher::gather_if_needed_async(value)
            .await
            .map_err(|e| format!("stochastic_evolution steps: {e}"))?;
        if let Some(result) = parse_integer_steps(&gathered) {
            return result;
        }
        &gathered
    } else {
        value
    };
    let raw = scalar_from_value_scalar(scalar_value, "stochastic_evolution steps").await?;
    if !raw.is_finite() || raw < 0.0 {
        return Err(crate::interpreter::errors::mex(
            "InvalidSteps",
            "stochastic_evolution: steps must be a non-negative scalar",
        ));
    }
    Ok(raw.round() as u32)
}

fn parse_integer_steps(value: &Value) -> Option<Result<u32, RuntimeError>> {
    scalar_integer_value(value).map(|integer| {
        integer
            .try_to_u64()
            .and_then(|raw| u32::try_from(raw).ok())
            .ok_or_else(|| {
                crate::interpreter::errors::mex(
                    "InvalidSteps",
                    "stochastic_evolution: steps must be a non-negative uint32 scalar",
                )
            })
    })
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
        Value::GpuTensor(handle) => {
            if runmat_accelerate_api::handle_integer_type(handle).is_some()
                || runmat_accelerate_api::handle_is_logical(handle)
            {
                return Err(crate::interpreter::errors::mex(
                    "UnsupportedType",
                    "stochastic_evolution: optimized evolution requires single or double state",
                ));
            }
            Ok((handle.clone(), None))
        }
        Value::Tensor(tensor) => {
            ensure_floating_state_tensor(tensor)?;
            let handle = upload_tensor_view(provider, tensor)?;
            Ok((handle.clone(), Some(handle)))
        }
        _ => {
            let gathered = runmat_runtime::dispatcher::gather_if_needed_async(value)
                .await
                .map_err(|e| format!("stochastic_evolution: {e}"))?;
            match gathered {
                Value::Tensor(t) => {
                    ensure_floating_state_tensor(&t)?;
                    let handle = upload_tensor_view(provider, &t)?;
                    Ok((handle.clone(), Some(handle)))
                }
                other => {
                    let tensor = runmat_runtime::builtins::common::tensor::value_into_tensor_for(
                        "stochastic_evolution",
                        other,
                    )?;
                    ensure_floating_state_tensor(&tensor)?;
                    let handle = upload_tensor_view(provider, &tensor)?;
                    Ok((handle.clone(), Some(handle)))
                }
            }
        }
    }
}

#[cfg(feature = "native-accel")]
fn ensure_floating_state_tensor(tensor: &runmat_builtins::Tensor) -> Result<(), RuntimeError> {
    if matches!(
        tensor.numeric_dtype(),
        runmat_builtins::NumericDType::F64 | runmat_builtins::NumericDType::F32
    ) {
        return Ok(());
    }
    Err(crate::interpreter::errors::mex(
        "UnsupportedType",
        &format!(
            "stochastic_evolution: optimized evolution requires single or double state, got {}",
            tensor.numeric_dtype().class_name()
        ),
    ))
}

#[cfg(feature = "native-accel")]
fn upload_tensor_view(
    provider: &dyn runmat_accelerate_api::AccelProvider,
    tensor: &runmat_builtins::Tensor,
) -> Result<runmat_accelerate_api::GpuTensorHandle, RuntimeError> {
    if let Some(storage) = tensor.integer_storage() {
        let data = match storage {
            runmat_builtins::IntegerStorage::I8(values) => {
                runmat_accelerate_api::HostIntegerDataView::I8(values)
            }
            runmat_builtins::IntegerStorage::I16(values) => {
                runmat_accelerate_api::HostIntegerDataView::I16(values)
            }
            runmat_builtins::IntegerStorage::I32(values) => {
                runmat_accelerate_api::HostIntegerDataView::I32(values)
            }
            runmat_builtins::IntegerStorage::I64(values) => {
                runmat_accelerate_api::HostIntegerDataView::I64(values)
            }
            runmat_builtins::IntegerStorage::U8(values) => {
                runmat_accelerate_api::HostIntegerDataView::U8(values)
            }
            runmat_builtins::IntegerStorage::U16(values) => {
                runmat_accelerate_api::HostIntegerDataView::U16(values)
            }
            runmat_builtins::IntegerStorage::U32(values) => {
                runmat_accelerate_api::HostIntegerDataView::U32(values)
            }
            runmat_builtins::IntegerStorage::U64(values) => {
                runmat_accelerate_api::HostIntegerDataView::U64(values)
            }
        };
        return provider
            .upload_integer(&runmat_accelerate_api::HostIntegerTensorView {
                data,
                shape: &tensor.shape,
            })
            .map_err(|e| crate::interpreter::errors::mex("UploadFailed", &e.to_string()));
    }
    let view = runmat_accelerate_api::HostTensorView {
        data: &tensor.data,
        shape: &tensor.shape,
    };
    provider
        .upload(&view)
        .map_err(|e| crate::interpreter::errors::mex("UploadFailed", &e.to_string()))
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "native-accel")]
    use super::ensure_floating_state_tensor;
    use super::{parse_integer_steps, scalar_from_value_scalar};
    use futures::executor::block_on;
    use runmat_builtins::{IntegerStorage, Tensor, Value};

    #[test]
    fn scalar_from_value_scalar_reads_typed_integer_storage_exactly() {
        let tensor =
            Tensor::new_integer(IntegerStorage::I16(vec![-3]), vec![1, 1]).expect("scalar tensor");

        assert_eq!(
            block_on(scalar_from_value_scalar(&Value::Tensor(tensor), "drift")).unwrap(),
            -3.0
        );
    }

    #[test]
    fn typed_integer_steps_accept_all_integer_classes() {
        macro_rules! assert_steps {
            ($storage:expr) => {{
                let tensor = Tensor::new_integer($storage, vec![1, 1]).expect("steps");
                assert_eq!(
                    parse_integer_steps(&Value::Tensor(tensor))
                        .expect("typed integer steps")
                        .expect("valid steps"),
                    7
                );
            }};
        }

        assert_steps!(IntegerStorage::I8(vec![7]));
        assert_steps!(IntegerStorage::I16(vec![7]));
        assert_steps!(IntegerStorage::I32(vec![7]));
        assert_steps!(IntegerStorage::I64(vec![7]));
        assert_steps!(IntegerStorage::U8(vec![7]));
        assert_steps!(IntegerStorage::U16(vec![7]));
        assert_steps!(IntegerStorage::U32(vec![7]));
        assert_steps!(IntegerStorage::U64(vec![7]));
    }

    #[cfg(feature = "native-accel")]
    #[test]
    fn optimized_provider_state_rejects_integer_host_storage() {
        let integer =
            Tensor::new_integer(IntegerStorage::U64(vec![7]), vec![1, 1]).expect("integer state");
        let err = ensure_floating_state_tensor(&integer).expect_err("integer state");
        assert!(err.to_string().contains("requires single or double state"));

        let single = Tensor::from_f32(vec![1.0], vec![1, 1]).expect("single state");
        ensure_floating_state_tensor(&single).expect("single state");
    }
}
