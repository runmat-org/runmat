use runmat_accelerate_api::{
    handle_integer_type, handle_is_logical, handle_precision, handle_storage, AccelProvider,
    GpuTensorHandle, GpuTensorStorage,
};
use runmat_value::Value;

use crate::builtins::common::gpu_helpers::{self, GpuOutputAliasPolicy, UnaryGpuOutputContract};
use crate::BuiltinResult;

use super::{builtin_error_with_detail, host, BUILTIN_NAME, CONJ_ERROR_INTERNAL};

pub(super) async fn execute(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    let storage = handle_storage(&handle);
    if gpu_helpers::expected_handle_numeric_element_type(&handle).is_err() {
        return Err(builtin_error_with_detail(
            &CONJ_ERROR_INTERNAL,
            "GPU input class metadata contradicts its physical storage",
        ));
    }
    let provider = gpu_helpers::require_resident_compute_owner(&handle, || {
        builtin_error_with_detail(&CONJ_ERROR_INTERNAL, "GPU provider unavailable for input")
    })?;
    if storage == GpuTensorStorage::Real {
        if handle_integer_type(&handle).is_some() {
            return gpu_helpers::resident_gpu_value(handle);
        }
        if handle_is_logical(&handle) {
            return gpu_helpers::logical_gpu_value(handle);
        }
    }

    let kernel_compatible = handle_integer_type(&handle).is_none()
        && !handle_is_logical(&handle)
        && handle_precision(&handle) == Some(provider.precision());
    if kernel_compatible {
        if let Some(output) = try_provider(&handle, provider).await? {
            return Ok(output);
        }
    }
    gather_conjugate_restore(handle, provider).await
}

async fn try_provider(
    input: &GpuTensorHandle,
    provider: &'static dyn AccelProvider,
) -> BuiltinResult<Option<Value>> {
    let storage = handle_storage(input);
    let metadata = gpu_helpers::ResidencyRestoreGuard::new(input);
    let result = provider.unary_conj(input).await;
    drop(metadata);
    match result {
        Ok(Some(output)) if valid_output(&output, input, provider) => {
            let output = gpu_helpers::propagate_owned_output_provenance(output, [input])?;
            let value = if storage == GpuTensorStorage::ComplexInterleaved {
                gpu_helpers::complex_gpu_value(output)
            } else {
                gpu_helpers::resident_gpu_value(output)
            };
            value.map(Some)
        }
        Ok(Some(output)) => Err(gpu_helpers::reject_gpu_output(
            output,
            builtin_error_with_detail(
                &CONJ_ERROR_INTERNAL,
                "provider unary_conj returned malformed output",
            ),
        )),
        Ok(None) => Ok(None),
        Err(source) => {
            let mut error = builtin_error_with_detail(
                &CONJ_ERROR_INTERNAL,
                format!("provider unary_conj failed: {source}"),
            );
            error = crate::runtime_error::with_provider_execution_source(error, source);
            Err(error)
        }
    }
}

fn valid_output(
    output: &GpuTensorHandle,
    input: &GpuTensorHandle,
    provider: &'static dyn AccelProvider,
) -> bool {
    let storage = handle_storage(input);
    let alias = if storage == GpuTensorStorage::Real {
        GpuOutputAliasPolicy::AllowInput
    } else {
        GpuOutputAliasPolicy::RequireDistinctAllocation
    };
    gpu_helpers::unary_gpu_output_matches(
        output,
        input,
        provider,
        UnaryGpuOutputContract {
            storage,
            precision: handle_precision(input),
            integer: None,
            logical: false,
            alias,
        },
    )
}

async fn gather_conjugate_restore(
    handle: GpuTensorHandle,
    provider: &'static dyn AccelProvider,
) -> BuiltinResult<Value> {
    let metadata = gpu_helpers::ResidencyRestoreGuard::new(&handle);
    let gathered = gpu_helpers::download_value_preserving_residency_async(provider, &handle).await;
    drop(metadata);
    let host_value = gathered.map_err(|source| {
        let mut error = builtin_error_with_detail(&CONJ_ERROR_INTERNAL, source.to_string());
        error.source = Some(Box::new(source));
        error
    })?;
    let conjugated = host::execute(host_value)?;
    gpu_helpers::restore_class_preserving_value(&handle, conjugated, BUILTIN_NAME).map_err(
        |source| {
            let mut error = builtin_error_with_detail(&CONJ_ERROR_INTERNAL, source.message());
            error.source = Some(Box::new(source));
            error
        },
    )
}

#[cfg(test)]
mod owned_tests {
    use super::*;

    #[test]
    fn complex_conjugation_requires_fresh_physical_output_but_real_identity_may_alias() {
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let real = runmat_value::Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
            let real = gpu_helpers::upload_tensor(provider, &real).unwrap();
            assert!(valid_output(&real, &real, provider));
            let complex =
                runmat_value::ComplexTensor::new(vec![(1.0, 2.0), (3.0, -4.0)], vec![1, 2])
                    .unwrap();
            let source = gpu_helpers::upload_complex_tensor(provider, &complex).unwrap();
            let alias = source.with_view(source.view().clone()).unwrap();
            assert_ne!(alias.view_identity(), source.view_identity());
            assert!(!valid_output(&alias, &source, provider));
            let output = gpu_helpers::upload_complex_tensor(provider, &complex).unwrap();
            assert!(valid_output(&output, &source, provider));
            drop(alias);
            source.allocation_lease().validate_live().unwrap();
        });
    }
}
