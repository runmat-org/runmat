//! Class-preserving inputs for the real explained-variance matrix pattern.
use runmat_accelerate_api::{
    AccelProvider, GpuHandleProvenance, GpuTensorHandle, GpuTensorStorage, NumericElementType,
    ProviderExecutionError, ProviderExecutionResult,
};
use runmat_runtime::builtins::common::gpu_helpers::{
    exact_provider_for_handle, upload_tensor_protecting_typed,
};
use runmat_value::{NumericDType, Value};

pub(super) fn prepare(
    provider: &'static dyn AccelProvider,
    value: &Value,
    protected: &[&GpuTensorHandle],
) -> ProviderExecutionResult<GpuTensorHandle> {
    let handle = match value {
        Value::GpuTensor(handle) => handle.clone(),
        Value::Tensor(tensor) => {
            if !matches!(
                tensor.numeric_dtype(),
                NumericDType::F32 | NumericDType::F64
            ) {
                return Err(runmat_accelerate_api::unsupported_provider_operation(
                    "explained variance requires real floating matrix inputs",
                )
                .into());
            }
            upload_tensor_protecting_typed(provider, tensor, protected)
                .map_err(ProviderExecutionError::from_source)?
                .with_provenance(GpuHandleProvenance::Automatic)
                .map_err(ProviderExecutionError::from_source)?
        }
        _ => {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "explained variance requires matrix inputs",
            )
            .into())
        }
    };
    if handle.shape().len() != 2
        || handle.physical().storage() != GpuTensorStorage::Real
        || runmat_accelerate_api::handle_is_logical(&handle)
        || !matches!(
            handle.physical().element_type(),
            NumericElementType::F32 | NumericElementType::F64
        )
    {
        return Err(runmat_accelerate_api::unsupported_provider_operation(
            "explained variance requires real floating two-dimensional matrices",
        )
        .into());
    }
    if handle.device_id() != provider.device_id()
        || !exact_provider_for_handle(&handle)
            .is_some_and(|owner| runmat_accelerate_api::same_provider_instance(owner, provider))
    {
        return Err(
            anyhow::anyhow!("explained variance input belongs to a different provider").into(),
        );
    }
    handle
        .allocation_lease()
        .validate_live()
        .map_err(ProviderExecutionError::from_source)?;
    Ok(handle)
}
