//! Resident spacing uses an authenticated real-elementwise recipe. Complex
//! input first obtains a provider-owned real magnitude; it is never gathered.

use runmat_accelerate_api::{
    GpuTensorHandle, GpuTensorStorage, ProviderElementType, ProviderLayout,
    ProviderRealElementwiseOperation, ProviderRealElementwiseRecipe,
    ProviderRealElementwiseRequest, ProviderRepresentation, ProviderResidency, ProviderStorage,
};
use runmat_value::Value;

use crate::builtins::common::gpu_helpers;
use crate::BuiltinResult;

pub(super) async fn value(input: &GpuTensorHandle) -> BuiltinResult<Value> {
    let provider = gpu_helpers::require_resident_compute_owner(input, || {
        super::super::errors::class("eps", "GPU input has no owning provider")
    })?;
    let precision = runmat_accelerate_api::handle_precision(input)
        .ok_or_else(|| super::super::errors::invalid_floating_prototype("eps"))?;
    let storage = input
        .descriptor()
        .storage
        .ok_or_else(|| super::super::errors::invalid_floating_prototype("eps"))?;
    if !matches!(
        storage,
        GpuTensorStorage::Real | GpuTensorStorage::ComplexInterleaved
    ) || runmat_accelerate_api::handle_integer_type(input).is_some()
        || runmat_accelerate_api::handle_is_logical(input)
        || !gpu_helpers::gpu_class_metadata_matches(input, Some(precision), None, false)
    {
        return Err(super::super::errors::invalid_floating_prototype("eps"));
    }
    let mut shape = Vec::new();
    shape
        .try_reserve_exact(input.shape().len())
        .map_err(|error| super::super::errors::internal("eps", error.to_string()))?;
    for dimension in input.shape() {
        shape.push(
            u64::try_from(*dimension)
                .map_err(|error| super::super::errors::internal("eps", error.to_string()))?,
        );
    }
    let element_type = match precision {
        runmat_accelerate_api::ProviderPrecision::F32 => ProviderElementType::F32,
        runmat_accelerate_api::ProviderPrecision::F64 => ProviderElementType::F64,
    };
    let representation = ProviderRepresentation {
        element_type,
        storage: ProviderStorage::DenseReal,
        layout: ProviderLayout::ColumnMajorContiguous,
        residency: ProviderResidency::Device,
        shape,
    };
    let logical_bytes = representation
        .checked_byte_len()
        .ok_or_else(|| super::super::errors::internal("eps", "GPU dimensions overflow"))?;
    if representation.checked_element_count() == Some(0) {
        // Empty arrays have no data-dependent computation. Construct the same
        // typed empty result through the owning provider, without downloading.
        let empty = match precision {
            runmat_accelerate_api::ProviderPrecision::F32 => {
                runmat_value::Tensor::from_f32(Vec::new(), input.shape().to_vec())
            }
            runmat_accelerate_api::ProviderPrecision::F64 => {
                runmat_value::Tensor::new(Vec::new(), input.shape().to_vec())
            }
        }
        .map_err(|error| super::super::errors::internal("eps", error))?;
        let output = gpu_helpers::upload_tensor_protecting(provider, &empty, &[input])?;
        if !gpu_helpers::unary_gpu_output_matches(
            &output,
            input,
            provider,
            gpu_helpers::UnaryGpuOutputContract {
                storage: GpuTensorStorage::Real,
                precision: Some(precision),
                integer: None,
                logical: false,
                alias: gpu_helpers::GpuOutputAliasPolicy::RequireDistinctAllocation,
            },
        ) {
            return Err(gpu_helpers::reject_gpu_output(
                output,
                terminal_after_effect("GPU empty spacing returned an invalid result"),
            ));
        }
        return publish_after_effect(output, input);
    }
    let mut request = ProviderRealElementwiseRequest {
        operation: ProviderRealElementwiseOperation::Spacing,
        source: representation,
        source_buffer_bytes: if storage == GpuTensorStorage::Real {
            input.physical().requested_bytes()
        } else {
            logical_bytes
        },
    };
    // This preflight precedes the complex magnitude effect. It also guarantees
    // that unsupported precision/device combinations never trigger a gather.
    let recipe = ProviderRealElementwiseRecipe::plan(provider, request.clone())
        .map_err(terminal)?
        .ok_or_else(|| unsupported("device does not support resident spacing"))?;
    let magnitude = if storage == GpuTensorStorage::ComplexInterleaved {
        let _guard = gpu_helpers::ResidencyRestoreGuard::new(input);
        let output = provider.unary_abs(input).await.map_err(|error| {
            terminal(runmat_accelerate_api::ProviderExecutionError::from(error).after_effect())
        })?;
        if !gpu_helpers::unary_gpu_output_matches(
            &output,
            input,
            provider,
            gpu_helpers::UnaryGpuOutputContract {
                storage: GpuTensorStorage::Real,
                precision: Some(precision),
                integer: None,
                logical: false,
                alias: gpu_helpers::GpuOutputAliasPolicy::RequireDistinctAllocation,
            },
        ) {
            return Err(gpu_helpers::reject_gpu_output(
                output,
                terminal_after_effect("provider magnitude returned an invalid result"),
            ));
        }
        Some(output)
    } else {
        None
    };
    let source = magnitude.as_ref().unwrap_or(input);
    let live_recipe = if magnitude.is_some() {
        // The pre-effect quote used the checked logical real footprint. The
        // provider's actual magnitude allocation is now known and must be
        // bound exactly by the retained recipe's live-view validation.
        request.source_buffer_bytes = source.physical().requested_bytes();
        ProviderRealElementwiseRecipe::plan(provider, request)
            .map_err(|error| terminal(error.after_effect()))?
            .ok_or_else(|| {
                terminal_after_effect("resident spacing recipe changed after magnitude")
            })?
    } else {
        recipe
    };
    let _guard = gpu_helpers::ResidencyRestoreGuard::new(input);
    let completed = live_recipe
        .execute_protected(source, &[input])
        .await
        .map_err(|error| {
            terminal(if magnitude.is_some() {
                error.after_effect()
            } else {
                error
            })
        })?;
    publish_after_effect(completed.output, input)
}

fn publish_after_effect(
    mut output: GpuTensorHandle,
    input: &GpuTensorHandle,
) -> BuiltinResult<Value> {
    if let Err(error) = gpu_helpers::propagate_output_provenance(&mut output, [input]) {
        return Err(gpu_helpers::reject_gpu_output(
            output,
            terminal_completed(error),
        ));
    }
    gpu_helpers::resident_gpu_value(output).map_err(terminal_completed)
}

fn terminal_completed(error: crate::RuntimeError) -> crate::RuntimeError {
    terminal(runmat_accelerate_api::ProviderExecutionError::from_source(error).after_effect())
}

fn unsupported(message: &'static str) -> crate::RuntimeError {
    let mut error = super::super::errors::class("eps", message);
    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
    error
}

fn terminal(error: runmat_accelerate_api::ProviderExecutionError) -> crate::RuntimeError {
    let fallback = unsupported("GPU spacing failed");
    crate::runtime_error::with_provider_execution_source(fallback, error)
}

fn terminal_after_effect(message: &'static str) -> crate::RuntimeError {
    let source =
        runmat_accelerate_api::ProviderExecutionError::from_source(std::io::Error::other(message))
            .after_effect();
    crate::runtime_error::with_provider_execution_source(unsupported(message), source)
}
