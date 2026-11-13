//! Narrow ordinary plus integration; unsupported mixed recipes decline before
//! effects. No failed partial composition is gathered and replayed.
use super::*;
use runmat_accelerate_api::{
    scalar_program::ScalarLiteral, AccelProvider, ProviderElementType, ProviderLayout,
    ProviderMixedRealAddRecipe, ProviderRealElementwiseOperation, ProviderRealElementwiseRequest,
    ProviderRepresentation, ProviderResidency, ProviderStorage,
};

pub(super) async fn try_add(
    provider: &'static dyn AccelProvider,
    resident: &GpuTensorHandle,
    host: &Value,
    control: ScalarLiteral,
) -> BuiltinResult<Option<Value>> {
    let host_shape = match host {
        Value::Num(_) => &[1, 1][..],
        Value::Tensor(tensor) => tensor.shape(),
        _ => return Ok(None),
    };
    if crate::builtins::common::broadcast::broadcast_shapes("plus", resident.shape(), host_shape)
        .ok()
        .as_deref()
        != Some(resident.shape())
    {
        return Ok(None);
    }
    if resident.view().layout() != runmat_accelerate_api::GpuTensorLayout::Contiguous {
        return Ok(None);
    }
    let Some(shape) = resident
        .shape()
        .iter()
        .map(|dimension| u64::try_from(*dimension).ok())
        .collect::<Option<Vec<_>>>()
    else {
        return Ok(None);
    };
    let request = ProviderRealElementwiseRequest {
        operation: ProviderRealElementwiseOperation::ScalarAdd {
            control,
            mode: runmat_types::RealFloatingAddMode::MixedDoubleThenSingle,
        },
        source: ProviderRepresentation {
            element_type: match (resident.physical().element_type(), control) {
                (runmat_accelerate_api::NumericElementType::F32, ScalarLiteral::F64Bits(_)) => {
                    ProviderElementType::F32
                }
                (runmat_accelerate_api::NumericElementType::F64, ScalarLiteral::F32Bits(_)) => {
                    ProviderElementType::F64
                }
                _ => return Ok(None),
            },
            storage: ProviderStorage::DenseReal,
            layout: ProviderLayout::ColumnMajorContiguous,
            residency: ProviderResidency::Device,
            shape,
        },
        source_buffer_bytes: resident.physical().requested_bytes(),
    };
    let recipe = match ProviderMixedRealAddRecipe::plan(provider, request) {
        Ok(Some(recipe)) => recipe,
        Ok(None) => return Ok(None),
        Err(error) if error.is_unsupported() => return Ok(None),
        Err(error) => return Err(terminal(error)),
    };
    match recipe.execute(resident).await {
        Ok(result) => publish_completed(result.output, resident).map(Some),
        Err(error) if error.is_unsupported() => Ok(None),
        Err(error) => Err(terminal(error)),
    }
}

fn publish_completed(output: GpuTensorHandle, resident: &GpuTensorHandle) -> BuiltinResult<Value> {
    super::super::super::resident_output_from_sources(output, [resident]).map_err(|error| {
        terminal(runmat_accelerate_api::ProviderExecutionError::from_source(error).after_effect())
    })
}

fn terminal(error: runmat_accelerate_api::ProviderExecutionError) -> crate::RuntimeError {
    let fallback = crate::build_runtime_error(format!("plus: {error}"))
        .with_builtin("plus")
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build();
    crate::runtime_error::with_provider_execution_source(fallback, error)
}

#[cfg(test)]
#[path = "mixed_add_publication_tests.rs"]
mod publication_tests;
