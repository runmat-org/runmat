//! Native-integer mean publication preserves the admitted class and owner.
use super::*;

pub(super) async fn evaluate(
    handle: GpuTensorHandle,
    parsed: &ParsedArguments,
) -> BuiltinResult<Value> {
    if let Some(weights) = parsed.weights.as_ref() {
        let provider = gpu_helpers::exact_provider_for_handle(&handle).ok_or_else(|| {
            mean_internal_error("mean: native integer gpuArray requires an acceleration provider")
        })?;
        let gathered = gpu_helpers::gather_tensor_async(&handle).await?;
        let storage = gathered
            .integer_storage()
            .cloned()
            .ok_or_else(|| mean_internal_error("mean: expected gathered integer storage"))?;
        let dim = match &parsed.axes {
            MeanAxes::Default => default_dimension_from_shape(gathered.shape()),
            MeanAxes::Dim(dim) => *dim,
            MeanAxes::Vec(_) | MeanAxes::All => {
                return Err(mean_invalid_argument(
                    "mean: Weights cannot be combined with vecdim or 'all'",
                ));
            }
        };
        let result = weighted_integer_mean(&storage, gathered.shape(), dim, weights)?;
        let uploaded = gpu_helpers::upload_tensor_protecting(provider, &result, &[&handle])
            .map_err(|source| {
                let mut error = mean_internal_error(format!(
                    "mean: failed to upload weighted native result: {source}"
                ));
                error.source = Some(Box::new(source));
                error
            })?;
        let uploaded = gpu_helpers::propagate_owned_output_provenance(uploaded, [&handle])?;
        return gpu_helpers::resident_gpu_value(uploaded);
    }
    let dims = resolve_native_dims(handle.shape(), &parsed.axes)?;
    if dims.is_empty() {
        return Ok(Value::GpuTensor(handle));
    }
    let provider = gpu_helpers::exact_provider_for_handle(&handle).ok_or_else(|| {
        mean_internal_error("mean: native integer gpuArray requires an acceleration provider")
    })?;
    let result = if dims.len() == handle.shape().len() {
        provider.reduce_integer_mean_native(&handle).await
    } else if dims.len() == 1 {
        provider
            .reduce_integer_mean_native_dim(&handle, dims[0])
            .await
    } else {
        provider
            .reduce_integer_mean_native_dims(&handle, &dims)
            .await
    }
    .map_err(|source| {
        let mut error = mean_internal_error(format!("mean: {source}"));
        error = crate::runtime_error::with_provider_execution_source(error, source.into());
        error.gpu_gather_retry = crate::GpuGatherRetry::Never;
        error
    })?;
    if !valid_result(provider, &handle, &result, &dims) {
        let mut error =
            mean_internal_error("mean: provider returned malformed native integer result");
        error.gpu_gather_retry = crate::GpuGatherRetry::Never;
        return Err(gpu_helpers::reject_gpu_output(result, error));
    }
    let result = gpu_helpers::propagate_owned_output_provenance(result, [&handle])?;
    gpu_helpers::resident_gpu_value(result)
}

fn valid_result(
    provider: &'static dyn AccelProvider,
    input: &GpuTensorHandle,
    output: &GpuTensorHandle,
    dims: &[usize],
) -> bool {
    let mut shape = input.shape().to_vec();
    for &dim in dims {
        shape[dim] = 1;
    }
    let Some(integer) = runmat_accelerate_api::handle_integer_type(input) else {
        return false;
    };
    let alias = if dims.iter().all(|&dim| input.shape()[dim] == 1) {
        gpu_helpers::GpuOutputAliasPolicy::AllowInput
    } else {
        gpu_helpers::GpuOutputAliasPolicy::RequireDistinctAllocation
    };
    gpu_helpers::binary_gpu_output_matches(
        output,
        input,
        input,
        provider,
        &gpu_helpers::BinaryGpuOutputContract {
            shape,
            storage: runmat_accelerate_api::GpuTensorStorage::Real,
            precision: None,
            integer: Some(integer),
            logical: false,
            alias,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_accelerate_api::{
        test_support::TestAllocationOwner, AccelDownloadFuture, GpuTensorStorage, HostTensorView,
        NumericElementType, ThreadProviderGuard,
    };

    struct Provider(TestAllocationOwner);
    impl AccelProvider for Provider {
        fn device_id(&self) -> u32 {
            self.0.device_id()
        }
        fn device_info(&self) -> String {
            "native mean output contract".into()
        }
        fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
            anyhow::bail!("unused upload")
        }
        fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
            Box::pin(async { anyhow::bail!("unused download") })
        }
    }

    #[test]
    fn native_mean_requires_exact_integer_class_shape_and_physical_result() {
        let provider = Box::leak(Box::new(Provider(TestAllocationOwner::new())));
        let _selected = ThreadProviderGuard::set(Some(provider));
        let input = provider
            .0
            .allocate_zeroed(vec![2, 2], NumericElementType::I64, GpuTensorStorage::Real)
            .unwrap();
        let valid = provider
            .0
            .allocate_zeroed(vec![1, 2], NumericElementType::I64, GpuTensorStorage::Real)
            .unwrap();
        let wrong_class = provider
            .0
            .allocate_zeroed(vec![1, 2], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let wrong_shape = provider
            .0
            .allocate_zeroed(vec![2, 1], NumericElementType::I64, GpuTensorStorage::Real)
            .unwrap();
        assert!(valid_result(provider, &input.handle, &valid.handle, &[0]));
        assert!(!valid_result(
            provider,
            &input.handle,
            &wrong_class.handle,
            &[0]
        ));
        assert!(!valid_result(
            provider,
            &input.handle,
            &wrong_shape.handle,
            &[0]
        ));
        assert!(valid_result(provider, &valid.handle, &valid.handle, &[0]));
        assert_eq!(input.probe.retirements(), 0);
    }
}
