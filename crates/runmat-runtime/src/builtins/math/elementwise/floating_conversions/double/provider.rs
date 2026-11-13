use super::double_error_with_detail;
use super::host::double_from_gathered;
use crate::builtins::common::gpu_helpers;
use crate::BuiltinResult;
use runmat_accelerate_api::{
    GpuTensorHandle, GpuTensorStorage, NumericElementType, ProviderPrecision,
};
use runmat_builtins::DOUBLE_ERROR_INTERNAL;
use runmat_value::Value;

#[cfg(test)]
use double_from_gpu as cast_from_gpu;
#[cfg(test)]
const CAST_PRECISION: ProviderPrecision = ProviderPrecision::F64;

pub(super) async fn double_from_gpu(
    handle: GpuTensorHandle,
    borrowed: &[&GpuTensorHandle],
) -> BuiltinResult<Value> {
    let mut protected = borrowed.to_vec();
    protected.push(&handle);
    let provider = gpu_helpers::require_resident_compute_owner(&handle, || {
        double_error_with_detail(&DOUBLE_ERROR_INTERNAL, "GPU provider unavailable for input")
    })?;
    let input_element =
        gpu_helpers::expected_handle_numeric_element_type(&handle).map_err(|_| {
            double_error_with_detail(
                &DOUBLE_ERROR_INTERNAL,
                "GPU input class metadata contradicts its physical storage",
            )
        })?;

    if handle.descriptor().storage.is_none() {
        return Err(double_error_with_detail(
            &runmat_builtins::DOUBLE_ERROR_INVALID_INPUT,
            "GPU input requires explicit storage",
        ));
    }
    let _metadata_guard = gpu_helpers::ResidencyRestoreGuard::new(&handle);
    let input_provenance = runmat_accelerate_api::handle_provenance(&handle)
        .unwrap_or(runmat_accelerate_api::GpuHandleProvenance::Automatic);
    let direct_eligible = provider.precision() == ProviderPrecision::F64
        && input_element == NumericElementType::F64
        && !runmat_accelerate_api::handle_is_logical(&handle)
        && runmat_accelerate_api::handle_storage(&handle) == GpuTensorStorage::Real;
    if direct_eligible {
        let direct = provider.unary_double(&handle).await;
        _metadata_guard.restore();
        match direct {
            Ok(mut output)
                if valid_output(&output, &handle, provider)
                    && !borrowed.iter().any(|input| {
                        !input.shares_allocation_with(&handle)
                            && output.shares_allocation_with(input)
                    }) =>
            {
                output = output.with_provenance(input_provenance).map_err(|error| {
                    crate::build_runtime_error(format!(
                        "double: cannot preserve input intent: {error}"
                    ))
                    .with_builtin("double")
                    .with_source(error)
                    .build()
                })?;
                return gpu_helpers::resident_gpu_value(output);
            }
            Ok(output) => {
                return Err(gpu_helpers::reject_gpu_output(
                    output,
                    double_error_with_detail(
                        &DOUBLE_ERROR_INTERNAL,
                        "provider unary_double returned malformed output",
                    ),
                ));
            }
            Err(error) if error.is_unsupported() => {}
            Err(error) => {
                return Err(crate::runtime_error::with_provider_execution_source(
                    double_error_with_detail(
                        &DOUBLE_ERROR_INTERNAL,
                        format!("provider unary_double failed: {error}"),
                    ),
                    error,
                ));
            }
        }
    }

    let gathered_result =
        gpu_helpers::download_value_preserving_residency_async(provider, &handle).await;
    _metadata_guard.restore();
    let gathered = gathered_result.map_err(contextual_upload_error)?;
    let converted = double_from_gathered(gathered)?;
    if provider.precision() != ProviderPrecision::F64 {
        return Ok(converted);
    }

    let mut output = match &converted {
        Value::Tensor(tensor) => {
            gpu_helpers::upload_tensor_protecting(provider, tensor, &protected)
                .map_err(contextual_upload_error)?
        }
        Value::ComplexTensor(tensor) => {
            gpu_helpers::upload_complex_tensor_protecting(provider, tensor, &protected)?
        }
        other => {
            return Err(double_error_with_detail(
                &DOUBLE_ERROR_INTERNAL,
                format!("double fallback produced unsupported value {other:?}"),
            ));
        }
    };
    if !valid_output(&output, &handle, provider) {
        return Err(gpu_helpers::reject_gpu_output(
            output,
            double_error_with_detail(
                &DOUBLE_ERROR_INTERNAL,
                "provider upload returned malformed double fallback output",
            ),
        ));
    }
    output = output.with_provenance(input_provenance).map_err(|error| {
        crate::build_runtime_error(format!("double: cannot preserve input intent: {error}"))
            .with_builtin("double")
            .with_source(error)
            .build()
    })?;
    if runmat_accelerate_api::handle_storage(&output) == GpuTensorStorage::ComplexInterleaved {
        gpu_helpers::complex_gpu_value(output)
    } else {
        gpu_helpers::resident_gpu_value(output)
    }
}

pub(super) fn valid_double_gpu_output(
    output: &GpuTensorHandle,
    input: &GpuTensorHandle,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    complex: bool,
) -> bool {
    let expected_storage = if complex {
        GpuTensorStorage::ComplexInterleaved
    } else {
        GpuTensorStorage::Real
    };
    gpu_helpers::unary_gpu_output_matches(
        output,
        input,
        provider,
        gpu_helpers::UnaryGpuOutputContract {
            storage: expected_storage,
            precision: Some(ProviderPrecision::F64),
            integer: None,
            logical: false,
            // Immutable same-type casts may return the exact input view. Uploads
            // independently require a distinct physical allocation at their boundary.
            alias: gpu_helpers::GpuOutputAliasPolicy::AllowInput,
        },
    )
}

fn valid_output(
    output: &GpuTensorHandle,
    input: &GpuTensorHandle,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
) -> bool {
    valid_double_gpu_output(
        output,
        input,
        provider,
        runmat_accelerate_api::handle_storage(input) == GpuTensorStorage::ComplexInterleaved,
    )
}

fn contextual_upload_error(error: runmat_async::RuntimeError) -> runmat_async::RuntimeError {
    let mut contextual = double_error_with_detail(&DOUBLE_ERROR_INTERNAL, error.message());
    contextual.source = Some(Box::new(error));
    contextual
}

#[cfg(test)]
mod owner_tests {
    use super::*;

    #[test]
    fn immutable_cast_adopts_exact_input_alias_without_retiring_caller() {
        super::super::super::cast_owner_tests::immutable_cast_adopts_exact_input_alias_without_retiring_caller(
            CAST_PRECISION,
            |handle, borrowed| futures::executor::block_on(cast_from_gpu(handle, borrowed)),
        );
    }

    #[test]
    fn immutable_cast_rejects_unrelated_borrowed_prototype_alias() {
        super::super::super::cast_owner_tests::immutable_cast_rejects_unrelated_borrowed_prototype_alias(
            CAST_PRECISION,
            |handle, borrowed| futures::executor::block_on(cast_from_gpu(handle, borrowed)),
        );
    }
}
