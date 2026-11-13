use super::*;

pub(super) async fn diff_gpu(
    handle: GpuTensorHandle,
    order: usize,
    dim: Option<usize>,
    call: &super::super::numerical_execution::NumericalCall<'_>,
) -> BuiltinResult<Value> {
    let working_dim = dim.unwrap_or_else(|| default_dimension(handle.shape()));
    if working_dim == 0 {
        return Err(diff_invalid_argument("diff: dimension must be >= 1"));
    }

    let provider = gpu_helpers::require_resident_compute_owner(&handle, || {
        diff_internal_error("diff: GPU input has no owning provider")
    })?;
    let exact_or_nonreal = runmat_accelerate_api::handle_integer_type(&handle).is_some()
        || runmat_accelerate_api::handle_is_logical(&handle)
        || runmat_accelerate_api::handle_storage(&handle)
            == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved;
    if exact_or_nonreal {
        call.fallback([&Value::GpuTensor(handle.clone())])?;
        let gathered =
            gpu_helpers::download_value_preserving_residency_async(provider, &handle).await?;
        call.check()?;
        let output = match gathered {
            Value::ComplexTensor(tensor) => {
                Value::ComplexTensor(diff_complex_tensor(tensor, order, Some(working_dim))?)
            }
            Value::Complex(re, im) => {
                let tensor =
                    ComplexTensor::new(vec![(re, im)], vec![1, 1]).map_err(diff_internal_error)?;
                complex_tensor_into_value(diff_complex_tensor(tensor, order, Some(working_dim))?)
            }
            Value::LogicalArray(logical) => diff_logical_array(logical, order, Some(working_dim))?,
            Value::Bool(flag) => {
                let tensor = tensor::value_into_tensor_for(NAME, Value::Bool(flag))
                    .map_err(diff_invalid_input)?;
                tensor::tensor_into_value(diff_tensor_host(tensor, order, Some(working_dim))?)
            }
            Value::Tensor(tensor) => {
                tensor::tensor_into_value(diff_tensor_host(tensor, order, Some(working_dim))?)
            }
            other => {
                return Err(diff_invalid_input(format!(
                    "unsupported gathered resident input {other:?}"
                )))
            }
        };
        return gpu_helpers::restore_class_preserving_value(&handle, output, NAME);
    }

    call.check()?;
    let result = super::super::numerical_execution::completed(
        call.kind,
        provider.diff_dim(&handle, order, working_dim - 1),
    );
    if let Some(device_result) = call.completed_handle(result)? {
        let rank = handle.shape().len().max(working_dim);
        let valid = device_result.shape().len() == rank
            && device_result
                .shape()
                .iter()
                .enumerate()
                .all(|(axis, &extent)| {
                    let original = handle.shape().get(axis).copied().unwrap_or(1);
                    extent
                        == if axis == working_dim - 1 {
                            original.saturating_sub(order)
                        } else {
                            original
                        }
                })
            && device_result.device_id() == handle.device_id()
            && runmat_accelerate_api::handle_storage(&device_result)
                == runmat_accelerate_api::GpuTensorStorage::Real
            && runmat_accelerate_api::handle_precision(&device_result)
                == runmat_accelerate_api::handle_precision(&handle)
            && runmat_accelerate_api::handle_integer_type(&device_result).is_none()
            && !runmat_accelerate_api::handle_is_logical(&device_result)
            && !device_result.shares_allocation_with(&handle)
            && gpu_helpers::exact_provider_for_handle(&device_result).is_some_and(|owner| {
                runmat_accelerate_api::same_provider_instance(owner, provider)
            })
            && gpu_helpers::gpu_class_metadata_matches(
                &device_result,
                runmat_accelerate_api::handle_precision(&handle),
                None,
                false,
            );
        if !valid {
            return Err(gpu_helpers::reject_gpu_output(
                device_result,
                diff_internal_error(
                    "provider returned malformed, aliased, or wrong-class differences",
                ),
            ));
        }
        let device_result =
            gpu_helpers::propagate_owned_output_provenance(device_result, [&handle])?;
        return gpu_helpers::resident_gpu_value(device_result);
    }
    call.fallback([&Value::GpuTensor(handle.clone())])?;
    let gathered =
        gpu_helpers::download_value_preserving_residency_async(provider, &handle).await?;
    call.check()?;
    let Value::Tensor(tensor) = gathered else {
        unreachable!("real floating gather remains a tensor")
    };
    let output = diff_tensor_host(tensor, order, Some(working_dim))?;
    gpu_helpers::restore_class_preserving_value(&handle, tensor::tensor_into_value(output), NAME)
}
