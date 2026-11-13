use super::super::numerical_execution::NumericalCall;
use super::*;
use runmat_accelerate_api::{GpuTensorStorage, ProviderPrecision};
#[path = "readback.rs"]
mod readback;
fn failure(detail: impl AsRef<str>) -> RuntimeError {
    super::super::numerical_execution::error(
        runmat_builtins::NumericalReductionKind::CountNonzero,
        detail.as_ref(),
    )
}
/// A floating count is exact through its accumulator's consecutive-integer range.
pub(super) fn exact_slice(
    precision: ProviderPrecision,
    shape: &[usize],
    dim: Option<usize>,
) -> bool {
    let length = match dim {
        None => shape
            .iter()
            .try_fold(1usize, |n, &extent| n.checked_mul(extent)),
        Some(dim) => Some(shape.get(dim.saturating_sub(1)).copied().unwrap_or(1)),
    };
    length.is_some_and(|n| precision == ProviderPrecision::F64 || n <= 16_777_216)
}
pub(super) async fn execute(
    input: &GpuTensorHandle,
    dim: Option<usize>,
    call: &NumericalCall<'_>,
) -> BuiltinResult<Value> {
    // Readback is the declared nnz output boundary, not input-host fallback.
    let owner = gpu_helpers::require_resident_compute_owner(input, || {
        failure("resident count input has no exact owning provider")
    })?;
    input.allocation_lease().validate_live().map_err(|source| {
        crate::build_runtime_error("nnz input allocation is unavailable")
            .with_source(source)
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build()
    })?;
    let eligible = runmat_accelerate_api::handle_integer_type(input).is_none()
        && !runmat_accelerate_api::handle_is_logical(input)
        && runmat_accelerate_api::handle_storage(input) == GpuTensorStorage::Real
        && runmat_accelerate_api::handle_precision(input) == Some(owner.precision())
        && exact_slice(owner.precision(), input.shape(), dim);
    if eligible {
        let output = match dim {
            None => call.attempt_handle(owner.reduce_nnz(input)).await?,
            Some(axis) if axis <= input.shape().len() => {
                call.attempt_handle(owner.reduce_nnz_dim(input, axis - 1))
                    .await?
            }
            Some(_) => None,
        };
        if let Some(output) = output {
            let mut expected;
            let expected_shape = if let Some(axis) = dim {
                expected = normalize_scalar_shape(input.shape());
                expected[axis - 1] = 1;
                &expected[..]
            } else {
                &[1, 1][..]
            };
            let output = validate_provider_result(owner, input, output, expected_shape)?;
            let downloaded = readback::download(owner, &output, call).await;
            let host = match downloaded {
                Ok(value) => value,
                Err(error) => {
                    return Err(gpu_helpers::reject_gpu_output(
                        output,
                        super::super::numerical_execution::terminal(error),
                    ))
                }
            };
            output.release().map_err(|source| {
                crate::build_runtime_error("nnz result retirement failed")
                    .with_source(source)
                    .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                    .build()
            })?;
            call.check()?;
            let tensor = match host {
                Value::Tensor(tensor) => tensor,
                Value::Num(value) if dim.is_none() => {
                    return checked_scalar(value, input.shape(), dim)
                }
                _ => {
                    return Err(failure(
                        "provider count readback has invalid class or shape",
                    ))
                }
            };
            if dim.is_none() {
                return checked_scalar(tensor::tensor_value_f64(&tensor, 0), input.shape(), dim);
            }
            let bound: usize = match dim {
                None => input.shape().iter().copied().product(),
                Some(axis) => input.shape()[axis - 1],
            };
            if (0..tensor.len()).any(|i| {
                let n = tensor::tensor_value_f64(&tensor, i);
                !n.is_finite() || n < 0. || n.fract() != 0. || n > bound as f64
            }) {
                return Err(failure(
                    "provider count is not an integer within the slice bounds",
                ));
            }
            let tensor = tensor::coerce_tensor_dtype(tensor, runmat_value::NumericDType::F64);
            return Ok(tensor::tensor_into_value(tensor));
        }
    }
    call.fallback([&Value::GpuTensor(input.clone())])?;
    let gathered = gpu_helpers::download_value_preserving_residency_async(owner, input).await?;
    call.check()?;
    nnz_host_value(gathered, dim)
}
fn checked_scalar(value: f64, shape: &[usize], dim: Option<usize>) -> BuiltinResult<Value> {
    let bound: usize = match dim {
        None => shape.iter().copied().product(),
        Some(axis) => shape[axis - 1],
    };
    if value.is_finite() && value >= 0. && value.fract() == 0. && value <= bound as f64 {
        Ok(Value::Num(value))
    } else {
        Err(failure("provider returned an invalid scalar count"))
    }
}
pub(super) fn validate_provider_result(
    provider: &dyn runmat_accelerate_api::AccelProvider,
    source: &GpuTensorHandle,
    result: GpuTensorHandle,
    expected_shape: &[usize],
) -> BuiltinResult<GpuTensorHandle> {
    let valid = !source.shares_allocation_with(&result)
        && result.shape() == expected_shape
        && result.device_id() == provider.device_id()
        && gpu_helpers::exact_provider_for_handle(&result)
            .is_some_and(|owner| runmat_accelerate_api::same_provider_instance(owner, provider))
        && runmat_accelerate_api::handle_storage(&result)
            == runmat_accelerate_api::GpuTensorStorage::Real
        && runmat_accelerate_api::handle_integer_type(&result).is_none()
        && !runmat_accelerate_api::handle_is_logical(&result)
        && runmat_accelerate_api::handle_precision(&result) == Some(provider.precision())
        && gpu_helpers::gpu_class_metadata_matches(
            &result,
            Some(provider.precision()),
            None,
            false,
        );
    if valid {
        return Ok(result);
    }
    Err(gpu_helpers::reject_gpu_output(
        result,
        failure("provider returned a malformed, aliased, or foreign reduction result"),
    ))
}
