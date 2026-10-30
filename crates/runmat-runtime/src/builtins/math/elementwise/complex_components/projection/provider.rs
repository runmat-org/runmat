use runmat_accelerate_api::{
    handle_integer_type, handle_is_logical, handle_precision, handle_storage, AccelProvider,
    GpuTensorHandle, GpuTensorStorage,
};
use runmat_value::Value;

use crate::builtins::common::gpu_helpers::{self, GpuOutputAliasPolicy, UnaryGpuOutputContract};
use crate::BuiltinResult;

use super::{host, ProjectionKind};

pub(super) async fn execute(kind: ProjectionKind, handle: GpuTensorHandle) -> BuiltinResult<Value> {
    let provider = gpu_helpers::require_resident_compute_owner(&handle, || {
        kind.internal("GPU provider unavailable for input")
    })?;
    if gpu_helpers::expected_handle_numeric_element_type(&handle).is_err() {
        return Err(kind.internal("GPU input class metadata contradicts its physical storage"));
    }

    let exact_host_path = handle_integer_type(&handle).is_some() || handle_is_logical(&handle);
    let kernel_compatible = handle_precision(&handle) == Some(provider.precision());
    if !exact_host_path && kernel_compatible {
        if let Some(output) = try_provider(kind, &handle, provider).await? {
            return Ok(output);
        }
    }

    gather_project_restore(kind, handle, provider).await
}

async fn try_provider(
    kind: ProjectionKind,
    input: &GpuTensorHandle,
    provider: &'static dyn AccelProvider,
) -> BuiltinResult<Option<Value>> {
    let metadata = gpu_helpers::ResidencyRestoreGuard::new(input);
    let result = match kind {
        ProjectionKind::Real => provider.unary_real(input).await,
        ProjectionKind::Imaginary => provider.unary_imag(input).await,
    };
    drop(metadata);

    match result {
        Ok(output) if valid_output(kind, &output, input, provider) => {
            let output = gpu_helpers::propagate_owned_output_provenance(output, [input])?;
            gpu_helpers::resident_gpu_value(output).map(Some)
        }
        Ok(output) => Err(gpu_helpers::reject_gpu_output(
            output,
            kind.internal(format!(
                "provider {} returned malformed output",
                kind.provider_hook()
            )),
        )),
        Err(error) if gpu_helpers::provider_hook_allows_host_fallback(&error) => Ok(None),
        Err(source) => {
            let mut error = kind.internal(format!(
                "provider {} failed: {source}",
                kind.provider_hook()
            ));
            error = crate::runtime_error::with_provider_execution_source(error, source.into());
            Err(error)
        }
    }
}

fn valid_output(
    kind: ProjectionKind,
    output: &GpuTensorHandle,
    input: &GpuTensorHandle,
    provider: &'static dyn AccelProvider,
) -> bool {
    let alias = if kind == ProjectionKind::Real && handle_storage(input) == GpuTensorStorage::Real {
        GpuOutputAliasPolicy::AllowInput
    } else {
        GpuOutputAliasPolicy::RequireDistinctAllocation
    };
    gpu_helpers::unary_gpu_output_matches(
        output,
        input,
        provider,
        UnaryGpuOutputContract {
            storage: GpuTensorStorage::Real,
            precision: handle_precision(input),
            integer: None,
            logical: false,
            alias,
        },
    )
}

async fn gather_project_restore(
    kind: ProjectionKind,
    handle: GpuTensorHandle,
    provider: &'static dyn AccelProvider,
) -> BuiltinResult<Value> {
    let metadata = gpu_helpers::ResidencyRestoreGuard::new(&handle);
    let gathered = gpu_helpers::download_value_preserving_residency_async(provider, &handle).await;
    drop(metadata);
    let host_value = gathered.map_err(|source| {
        let mut error = kind.internal(source.to_string());
        error.source = Some(Box::new(source));
        error
    })?;
    let projected = host::execute(kind, host_value)?;
    gpu_helpers::restore_class_preserving_value(&handle, projected, kind.name()).map_err(|source| {
        let mut error = kind.internal(source.to_string());
        error.source = Some(Box::new(source));
        error
    })
}

#[cfg(test)]
mod owned_tests {
    use super::*;

    #[test]
    fn real_identity_may_alias_but_imaginary_zero_result_requires_fresh_allocation() {
        crate::builtins::common::test_support::with_test_provider(|provider| {
            let tensor = runmat_value::Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
            let input = gpu_helpers::upload_tensor(provider, &tensor).unwrap();
            let alias = input.with_view(input.view().clone()).unwrap();
            assert!(valid_output(ProjectionKind::Real, &alias, &input, provider));
            assert!(!valid_output(
                ProjectionKind::Imaginary,
                &alias,
                &input,
                provider
            ));
            let output = gpu_helpers::upload_tensor(provider, &tensor).unwrap();
            assert!(valid_output(
                ProjectionKind::Imaginary,
                &output,
                &input,
                provider
            ));
            drop(alias);
            input.allocation_lease().validate_live().unwrap();
        });
    }
}
