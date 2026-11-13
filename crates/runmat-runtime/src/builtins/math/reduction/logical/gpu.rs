use crate::builtins::common::{
    gpu_helpers::{self, HostTruthTensorOwned},
    shape::{is_scalar_shape, normalize_scalar_shape},
    spec::ReductionNaN,
    tensor,
};
use crate::BuiltinResult;
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle};
use runmat_builtins::LogicalReductionKind;
use runmat_value::{LogicalArray, Value};

use super::{arguments::ReductionSpec, host, route, shape, LogicalReductionConfig};

mod output;
#[cfg(test)]
mod tests;

pub(super) async fn reduce(
    config: &LogicalReductionConfig,
    handle: GpuTensorHandle,
    spec: ReductionSpec,
    nan_mode: ReductionNaN,
) -> BuiltinResult<Value> {
    gpu_helpers::require_resident_compute_owner(&handle, || {
        config.error(
            config.internal,
            "resident input has no exact owning provider",
        )
    })?;
    if route::select(config.kind, nan_mode) == route::LogicalReductionRoute::HostExact {
        return fallback(config, handle, spec, nan_mode).await;
    }
    let Some(provider) = gpu_helpers::exact_provider_for_handle(&handle) else {
        return fallback(config, handle, spec, nan_mode).await;
    };
    match try_provider(config, provider, &handle, &spec, nan_mode).await? {
        Some(output) => logical_from_host(config, output),
        None => fallback(config, handle, spec, nan_mode).await,
    }
}

async fn fallback(
    config: &LogicalReductionConfig,
    handle: GpuTensorHandle,
    spec: ReductionSpec,
    nan_mode: ReductionNaN,
) -> BuiltinResult<Value> {
    let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
    host::reduce(config, Value::Tensor(tensor), spec, nan_mode).await
}

async fn try_provider(
    config: &LogicalReductionConfig,
    provider: &'static dyn AccelProvider,
    handle: &GpuTensorHandle,
    spec: &ReductionSpec,
    nan_mode: ReductionNaN,
) -> BuiltinResult<Option<HostTruthTensorOwned>> {
    let omit_nan = nan_mode == ReductionNaN::Omit;
    if matches!(spec, ReductionSpec::All) {
        let reduced = match config.kind {
            LogicalReductionKind::All => provider.reduce_all(handle, omit_nan).await,
            LogicalReductionKind::Any => provider.reduce_any(handle, omit_nan).await,
        };
        if let Ok(reduced) = reduced {
            let mut outputs = output::ReductionOutputs::new(handle);
            // Full reductions may choose a scalar rank, but never a non-scalar.
            let expected = if is_scalar_shape(reduced.shape()) {
                reduced.shape().to_vec()
            } else {
                vec![1, 1]
            };
            outputs.adopt(config, provider, reduced, expected)?;
            let downloaded =
                gpu_helpers::download_truth_values_async(provider, outputs.current()).await;
            return downloaded
                .map(Some)
                .map_err(|error| download_error(config, error));
        }
    }
    reduce_dimensions(config, provider, handle, spec, omit_nan).await
}

async fn reduce_dimensions(
    config: &LogicalReductionConfig,
    provider: &'static dyn AccelProvider,
    handle: &GpuTensorHandle,
    spec: &ReductionSpec,
    omit_nan: bool,
) -> BuiltinResult<Option<HostTruthTensorOwned>> {
    let dimensions = shape::dimensions(spec, handle.shape());
    if dimensions.is_empty() {
        return Ok(None);
    }
    let mut outputs = output::ReductionOutputs::new(handle);

    for dimension in dimensions {
        let Some(axis) = dimension.checked_sub(1) else {
            return Ok(None);
        };
        if axis >= outputs.current().shape().len() {
            return Ok(None);
        }
        let mut expected = outputs.current().shape().to_vec();
        expected[axis] = 1;
        let next = match config.kind {
            LogicalReductionKind::All => {
                provider
                    .reduce_all_dim(outputs.current(), axis, omit_nan)
                    .await
            }
            LogicalReductionKind::Any => {
                provider
                    .reduce_any_dim(outputs.current(), axis, omit_nan)
                    .await
            }
        };
        let Ok(next) = next else {
            return Ok(None);
        };
        outputs.adopt(config, provider, next, expected)?;
    }

    let downloaded = gpu_helpers::download_truth_values_async(provider, outputs.current()).await;
    downloaded
        .map(Some)
        .map_err(|error| download_error(config, error))
}

fn download_error(
    config: &LogicalReductionConfig,
    error: crate::RuntimeError,
) -> crate::RuntimeError {
    let mut contextual = config.error(config.internal, error.message());
    contextual.source = Some(Box::new(error));
    contextual
}

fn logical_from_host(
    config: &LogicalReductionConfig,
    host: HostTruthTensorOwned,
) -> BuiltinResult<Value> {
    if host.data.len() == 1 {
        return Ok(Value::Bool(host.data[0] != 0));
    }
    let shape = if tensor::element_count(&host.shape) == host.data.len() {
        normalize_scalar_shape(&host.shape)
    } else if is_scalar_shape(&host.shape) {
        if host.data.is_empty() {
            Vec::new()
        } else {
            vec![host.data.len()]
        }
    } else {
        host.shape
    };
    let data = host
        .data
        .into_iter()
        .map(|value| u8::from(value != 0))
        .collect();
    LogicalArray::new(data, shape)
        .map(Value::LogicalArray)
        .map_err(|error| config.error(config.internal, error))
}
