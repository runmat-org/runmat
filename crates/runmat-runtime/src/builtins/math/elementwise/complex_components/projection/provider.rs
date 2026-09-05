use runmat_accelerate_api::{
    handle_integer_type, handle_is_logical, handle_precision, handle_provenance, handle_storage,
    set_handle_provenance, AccelProvider, GpuHandleProvenance, GpuTensorHandle, GpuTensorStorage,
};
use runmat_value::Value;

use crate::builtins::common::gpu_helpers::{self, GpuOutputAliasPolicy, UnaryGpuOutputContract};
use crate::BuiltinResult;

use super::{host, ProjectionKind};

pub(super) async fn execute(kind: ProjectionKind, handle: GpuTensorHandle) -> BuiltinResult<Value> {
    let provider = gpu_helpers::exact_provider_for_handle(&handle)
        .ok_or_else(|| kind.internal("GPU provider unavailable for input"))?;
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
    let metadata = gpu_helpers::snapshot_handle_metadata(input);
    let provenance = handle_provenance(input).unwrap_or(GpuHandleProvenance::Automatic);
    let result = match kind {
        ProjectionKind::Real => provider.unary_real(input).await,
        ProjectionKind::Imaginary => provider.unary_imag(input).await,
    };
    gpu_helpers::restore_handle_metadata(input, &metadata);

    match result {
        Ok(mut output) if valid_output(kind, &output, input, provider) => {
            set_handle_provenance(&mut output, provenance);
            Ok(Some(gpu_helpers::resident_gpu_value(output)))
        }
        Ok(output) => {
            gpu_helpers::free_unprotected_exact_owner(&output, &[input]);
            Err(kind.internal(format!(
                "provider {} returned malformed output",
                kind.provider_hook()
            )))
        }
        Err(error) if gpu_helpers::provider_hook_is_unsupported(&error) => Ok(None),
        Err(error) => {
            Err(kind.internal(format!("provider {} failed: {error}", kind.provider_hook())))
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
        GpuOutputAliasPolicy::RequireDistinct
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
    let metadata = gpu_helpers::snapshot_handle_metadata(&handle);
    let gathered = gpu_helpers::download_value_preserving_residency_async(provider, &handle).await;
    gpu_helpers::restore_handle_metadata(&handle, &metadata);
    let host_value = gathered.map_err(|error| kind.internal(error.to_string()))?;
    let projected = host::execute(kind, host_value)?;
    gpu_helpers::restore_class_preserving_value(&handle, projected, kind.name())
        .map_err(|error| kind.internal(error.to_string()))
}
