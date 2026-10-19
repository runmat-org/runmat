//! Resident NaN-omitting sum stages and temporary ownership.
use super::*;

pub(super) async fn sum_gpu_with_omitnan(
    handle: GpuTensorHandle,
    parsed: &ParsedArguments,
) -> BuiltinResult<Value> {
    if runmat_accelerate_api::handle_storage(&handle)
        == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
    {
        // Complex omission is a pairwise numerical policy; the real-only
        // provider NaN map cannot implement it by transforming lanes alone.
        return sum_gpu_fallback(&handle, parsed).await;
    }
    if runmat_accelerate_api::handle_is_logical(&handle)
        || runmat_accelerate_api::handle_integer_type(&handle).is_some()
    {
        // Logical and integer values cannot contain NaN. Both ordinary
        // floating and native integer sums use their usual resident route.
        let include = ParsedArguments {
            invocation: parsed.invocation,
            selection: parsed.selection.clone(),
            nan_mode: ReductionNaN::Include,
            output: parsed.output,
        };
        return sum_gpu_include(handle, &include).await;
    }
    let Some(provider) = gpu_helpers::exact_provider_for_handle(&handle) else {
        return sum_gpu_fallback(&handle, parsed).await;
    };
    let resolved = resolve_dims(handle.shape(), &parsed.selection)?;
    if resolved.dims_in_bounds.is_empty() {
        return Ok(Value::GpuTensor(handle));
    }

    // Single observations widen exactly before the real-F64 NaN map. Keep the
    // widening on the same owner and retire it after the map has its own output.
    let widened =
        if runmat_accelerate_api::handle_precision(&handle) == Some(ProviderPrecision::F32) {
            match provider.unary_double(&handle).await {
                Ok(output) => Some(provider_output::admit_input_double(
                    provider, &handle, output,
                )?),
                Err(error) if error.is_unsupported() => {
                    return sum_gpu_fallback(&handle, parsed).await;
                }
                Err(error) => return Err(sum_provider_error(error)),
            }
        } else {
            None
        };
    let map_input = widened.as_ref().unwrap_or(&handle);
    let cleaned = match provider.map_nan_to_zero(map_input) {
        Ok(cleaned) => {
            match provider_output::admit_nan_map(provider, &handle, map_input, cleaned) {
                Ok(cleaned) => cleaned,
                Err(error) => {
                    return Err(if let Some(widened) = widened {
                        provider_output::fail_stage(widened, error)
                    } else {
                        error
                    });
                }
            }
        }
        Err(error) if error.is_unsupported() && widened.is_none() => {
            return sum_gpu_fallback(&handle, parsed).await;
        }
        Err(error) => {
            let error = sum_provider_error(if widened.is_some() {
                error.after_effect()
            } else {
                error
            });
            return Err(if let Some(widened) = widened {
                provider_output::fail_stage(widened, error)
            } else {
                error
            });
        }
    };
    if let Some(widened) = widened {
        if let Err(error) = provider_output::retire_widened_input(widened) {
            return Err(provider_output::fail_stage(cleaned, error));
        }
    }

    let mut current = cleaned;
    for &dim in &resolved.dims_in_bounds {
        match provider.reduce_sum_dim(&current, dim).await {
            Ok(next) => {
                current = provider_output::advance_reduction(
                    provider,
                    &handle,
                    current,
                    next,
                    dim,
                    parsed.output,
                )?;
            }
            Err(error) => {
                return Err(provider_output::fail_stage(
                    current,
                    sum_provider_error(error.after_effect()),
                ))
            }
        }
    }
    publish_submitted_sum(provider, current, &handle)
}
