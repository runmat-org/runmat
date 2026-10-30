//! Request-local Black–Scholes preparation and owning output admission.
use super::*;
use runmat_accelerate_api::{
    GpuTensorHandle, GpuTensorStorage, ProviderBlackScholesPriceInput,
    ProviderBlackScholesPriceRequest, ProviderExecutionError, ProviderPrecision,
};

#[cfg(test)]
mod tests;

pub(super) struct BlsPriceGpuEval {
    call: GpuTensorHandle,
    put: GpuTensorHandle,
}

impl BlsPriceGpuEval {
    pub(super) fn output(self, requested: Option<usize>) -> BuiltinResult<Value> {
        match requested {
            Some(0) => Ok(Value::output_list(Vec::new())),
            Some(1) => Ok(Value::output_list(vec![gpu_helpers::resident_gpu_value(
                self.call,
            )?])),
            Some(2) => Ok(Value::output_list(vec![
                gpu_helpers::resident_gpu_value(self.call)?,
                gpu_helpers::resident_gpu_value(self.put)?,
            ])),
            Some(_) => Err(error(
                BLSPRICE,
                "blsprice: too many output arguments",
                &ERROR_INVALID_ARGUMENT,
            )),
            None => gpu_helpers::resident_gpu_value(self.call),
        }
    }
}

struct PreparedInput {
    handle: GpuTensorHandle,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

fn provider_error(source: ProviderExecutionError) -> RuntimeError {
    crate::runtime_error::with_provider_execution_source(
        error(BLSPRICE, format!("blsprice: {source}"), &ERROR_INTERNAL),
        source,
    )
}

pub(super) async fn try_blsprice_gpu(
    values: &[Value],
    result_class: precision::PricingClass,
) -> BuiltinResult<Option<BlsPriceGpuEval>> {
    let borrowed: Vec<_> = values
        .iter()
        .filter_map(|value| match value {
            Value::GpuTensor(handle) => Some(handle),
            _ => None,
        })
        .collect();
    gpu_helpers::require_resident_compute_owners(&borrowed, false, || {
        error(
            BLSPRICE,
            "blsprice: resident input has no exact owning provider",
            &ERROR_INTERNAL,
        )
    })?;
    let Some(anchor) = borrowed.first() else {
        return Ok(None);
    };
    let Some(provider) = gpu_helpers::exact_provider_for_handle(anchor) else {
        return Ok(None);
    };
    if provider.precision() != result_class.provider_precision() {
        return Ok(None);
    }
    let _residency: Vec<_> = borrowed
        .iter()
        .map(|handle| gpu_helpers::ResidencyRestoreGuard::new(handle))
        .collect();
    // Admit every operand and broadcast shape before allocating any upload.
    let mut shapes = Vec::with_capacity(values.len());
    let mut hosts = Vec::with_capacity(values.len());
    let mut output_shape = vec![1, 1];
    for value in values {
        let (shape, host) = match value {
            Value::GpuTensor(handle) => {
                if !gpu_helpers::exact_provider_for_handle(handle).is_some_and(|owner| {
                    runmat_accelerate_api::same_provider_instance(owner, provider)
                }) {
                    return Ok(None);
                }
                if runmat_accelerate_api::handle_storage(handle) != GpuTensorStorage::Real {
                    return Err(error(
                        BLSPRICE,
                        "blsprice: complex gpuArray inputs are not supported",
                        &ERROR_INVALID_INPUT,
                    ));
                }
                if runmat_accelerate_api::handle_integer_type(handle).is_some()
                    || runmat_accelerate_api::handle_is_logical(handle)
                    || runmat_accelerate_api::handle_precision(handle) != Some(provider.precision())
                {
                    return Ok(None);
                }
                (canonical_shape(handle.shape()), None)
            }
            other => {
                let host = numeric_input(BLSPRICE, other.clone())?;
                (host.shape.clone(), Some(host.values))
            }
        };
        shape_len_checked(BLSPRICE, &shape)?;
        output_shape = broadcast::broadcast_shapes(BLSPRICE, &output_shape, &shape)
            .map_err(|cause| error(BLSPRICE, format!("blsprice: {cause}"), &ERROR_INVALID_INPUT))?;
        shapes.push(shape);
        hosts.push(host);
    }
    let len = shape_len_checked(BLSPRICE, &output_shape)?;
    let mut prepared: Vec<PreparedInput> = Vec::with_capacity(values.len());
    for ((value, shape), host) in values.iter().zip(shapes).zip(hosts) {
        let handle = match value {
            Value::GpuTensor(handle) => handle.clone(),
            _ => {
                let data = host.expect("host operand admitted above");
                let tensor = match provider.precision() {
                    ProviderPrecision::F32 => Tensor::from_f32(
                        data.into_iter()
                            .map(precision::single_input)
                            .collect::<BuiltinResult<Vec<_>>>()?
                            .into_iter()
                            .map(|value| value as f32)
                            .collect(),
                        shape.clone(),
                    ),
                    ProviderPrecision::F64 => Tensor::new(data, shape.clone()),
                }
                .map_err(|cause| error(BLSPRICE, cause, &ERROR_INTERNAL))?;
                let protected: Vec<_> = borrowed
                    .iter()
                    .copied()
                    .chain(prepared.iter().map(|input| &input.handle))
                    .collect();
                match gpu_helpers::upload_tensor_protecting_typed(provider, &tensor, &protected) {
                    Ok(handle) => handle,
                    Err(runmat_accelerate_api::NumericUploadError::Provider(cause))
                        if cause.is_unsupported() =>
                    {
                        return Ok(None)
                    }
                    Err(cause) => {
                        return Err(provider_error(ProviderExecutionError::from_source(cause)))
                    }
                }
            }
        };
        let aligned = broadcast::align_shape(&shape, output_shape.len());
        let mut physical_shape = aligned.clone();
        if runmat_accelerate_api::handle_is_transposed(&handle) {
            physical_shape.swap(0, 1);
        }
        let mut strides = broadcast::compute_strides(&physical_shape);
        if runmat_accelerate_api::handle_is_transposed(&handle) {
            strides.swap(0, 1);
        }
        prepared.push(PreparedInput {
            handle,
            shape: aligned,
            strides,
        });
    }
    let inputs: Vec<_> = prepared
        .iter()
        .map(|input| ProviderBlackScholesPriceInput {
            handle: &input.handle,
            shape: &input.shape,
            strides: &input.strides,
        })
        .collect();
    let result = match provider.black_scholes_price(&ProviderBlackScholesPriceRequest {
        inputs: &inputs,
        output_shape: &output_shape,
        len,
    }) {
        Ok(result) => result,
        Err(cause) if gpu_helpers::provider_hook_allows_host_fallback(&cause) => return Ok(None),
        Err(cause) => return Err(provider_error(cause.into())),
    };
    let valid = |handle: &GpuTensorHandle| {
        handle.shape() == output_shape
            && gpu_helpers::exact_provider_for_handle(handle)
                .is_some_and(|owner| runmat_accelerate_api::same_provider_instance(owner, provider))
            && runmat_accelerate_api::handle_storage(handle) == GpuTensorStorage::Real
            && !runmat_accelerate_api::handle_is_transposed(handle)
            && gpu_helpers::gpu_class_metadata_matches(
                handle,
                Some(provider.precision()),
                None,
                false,
            )
            && prepared
                .iter()
                .all(|input| !handle.shares_allocation_with(&input.handle))
            && borrowed
                .iter()
                .all(|input| !handle.shares_allocation_with(input))
    };
    if !valid(&result.call)
        || !valid(&result.put)
        || result.call.shares_allocation_with(&result.put)
    {
        let failure = error(
            BLSPRICE,
            "blsprice: provider returned invalid or aliased call/put outputs",
            &ERROR_INTERNAL,
        );
        let failure = gpu_helpers::reject_gpu_output(result.call, failure);
        return Err(gpu_helpers::reject_gpu_output(result.put, failure));
    }
    let mut call = result.call;
    let mut put = result.put;
    if let Err(error) =
        gpu_helpers::propagate_output_provenance(&mut call, borrowed.iter().copied())
    {
        return Err(gpu_helpers::reject_gpu_outputs([call, put], error));
    }
    if let Err(error) = gpu_helpers::propagate_output_provenance(&mut put, borrowed.iter().copied())
    {
        return Err(gpu_helpers::reject_gpu_outputs([call, put], error));
    }
    Ok(Some(BlsPriceGpuEval { call, put }))
}
