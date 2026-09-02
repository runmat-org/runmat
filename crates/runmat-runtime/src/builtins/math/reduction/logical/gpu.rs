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

use super::{arguments::ReductionSpec, host, shape, LogicalReductionConfig};

pub(super) async fn reduce(
    config: &LogicalReductionConfig,
    handle: GpuTensorHandle,
    spec: ReductionSpec,
    nan_mode: ReductionNaN,
) -> BuiltinResult<Value> {
    if config.kind == LogicalReductionKind::Any && nan_mode == ReductionNaN::Omit {
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
            let downloaded = gpu_helpers::download_truth_values_async(provider, &reduced).await;
            let _ = provider.free(&reduced);
            return downloaded
                .map(Some)
                .map_err(|error| config.error(config.internal, error));
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
    let dimensions = shape::dimensions(spec, &handle.shape);
    if dimensions.is_empty() {
        return Ok(None);
    }
    let mut current = handle.clone();
    let mut current_owned = false;
    let mut intermediates = Vec::new();

    for dimension in dimensions {
        let Some(axis) = dimension.checked_sub(1) else {
            free_owned(provider, current_owned.then_some(&current), &intermediates);
            return Ok(None);
        };
        if axis >= current.shape.len() {
            free_owned(provider, current_owned.then_some(&current), &intermediates);
            return Ok(None);
        }
        let next = match config.kind {
            LogicalReductionKind::All => provider.reduce_all_dim(&current, axis, omit_nan).await,
            LogicalReductionKind::Any => provider.reduce_any_dim(&current, axis, omit_nan).await,
        };
        let Ok(next) = next else {
            free_owned(provider, current_owned.then_some(&current), &intermediates);
            return Ok(None);
        };
        if current_owned {
            intermediates.push(current);
        }
        current = next;
        current_owned = true;
    }
    if !current_owned {
        return Ok(None);
    }

    let downloaded = gpu_helpers::download_truth_values_async(provider, &current).await;
    let _ = provider.free(&current);
    free_owned(provider, None, &intermediates);
    downloaded
        .map(Some)
        .map_err(|error| config.error(config.internal, error))
}

fn free_owned(
    provider: &dyn AccelProvider,
    current: Option<&GpuTensorHandle>,
    intermediates: &[GpuTensorHandle],
) {
    if let Some(current) = current {
        let _ = provider.free(current);
    }
    for intermediate in intermediates {
        let _ = provider.free(intermediate);
    }
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
