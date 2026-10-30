use super::*;

pub(super) async fn provider_input_contains_nan(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    source: &GpuTensorHandle,
) -> BuiltinResult<Option<bool>> {
    if source.shape().contains(&0) || runmat_accelerate_api::handle_integer_type(source).is_some() {
        return Ok(Some(false));
    }
    let mask = match provider.logical_isnan(source) {
        Ok(mask) => mask,
        Err(error) if gpu_helpers::provider_hook_allows_host_fallback(&error) => return Ok(None),
        Err(error) => return Err(logical_provider_error(error.into())),
    };
    if !valid_logical_gpu_output(&mask, source, provider) {
        return Err(gpu_helpers::reject_gpu_output(
            mask,
            logical_error_with_message(
                "logical: provider returned a malformed NaN mask",
                &LOGICAL_ERROR_INTERNAL,
            ),
        ));
    }
    let reduction = match provider.reduce_any(&mask, false).await {
        Ok(reduction) => reduction,
        Err(error) => {
            return Err(logical_provider_error(
                runmat_accelerate_api::ProviderExecutionError::from(error).after_effect(),
            ));
        }
    };
    if reduction.shares_allocation_with(source)
        || reduction.shares_allocation_with(&mask)
        || reduction.shape() != [1, 1]
        || !runmat_accelerate_api::handle_is_logical(&reduction)
        || runmat_accelerate_api::handle_storage(&reduction)
            != runmat_accelerate_api::GpuTensorStorage::Real
        || !gpu_helpers::expected_handle_numeric_element_type(&reduction).is_ok_and(|element| {
            matches!(
                element,
                runmat_accelerate_api::NumericElementType::F32
                    | runmat_accelerate_api::NumericElementType::F64
                    | runmat_accelerate_api::NumericElementType::U8
            ) && gpu_helpers::numeric_descriptor_matches(
                &reduction,
                element,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
        })
        || reduction.device_id() != source.device_id()
        || !gpu_helpers::exact_provider_for_handle(&reduction)
            .is_some_and(|owner| runmat_accelerate_api::same_provider_instance(owner, provider))
    {
        return Err(gpu_helpers::reject_gpu_output(
            reduction,
            logical_error_with_message(
                "logical: provider returned a malformed NaN reduction",
                &LOGICAL_ERROR_INTERNAL,
            ),
        ));
    }
    let downloaded = provider
        .download_numeric(&reduction)
        .await
        .map_err(|error| logical_provider_error(error.into()))?;
    let expected_element = gpu_helpers::expected_handle_numeric_element_type(&reduction)
        .expect("validated live logical reduction encoding");
    if downloaded.validate().is_err()
        || downloaded.shape != [1, 1]
        || downloaded.storage != runmat_accelerate_api::GpuTensorStorage::Real
        || downloaded.data.element_type() != expected_element
    {
        return Err(logical_error_with_message(
            "logical: invalid NaN reduction payload",
            &LOGICAL_ERROR_INTERNAL,
        ));
    }
    let value = match &downloaded.data {
        runmat_accelerate_api::HostNumericDataOwned::F64(values) if values.len() == 1 => values[0],
        runmat_accelerate_api::HostNumericDataOwned::F32(values) if values.len() == 1 => {
            f64::from(values[0])
        }
        runmat_accelerate_api::HostNumericDataOwned::U8(values) if values.len() == 1 => {
            f64::from(values[0])
        }
        _ => {
            return Err(logical_error_with_message(
                "logical: invalid NaN reduction payload",
                &LOGICAL_ERROR_INTERNAL,
            ));
        }
    };
    if value != 0.0 && value != 1.0 {
        return Err(logical_error_with_message(
            "logical: invalid NaN reduction payload",
            &LOGICAL_ERROR_INTERNAL,
        ));
    }
    Ok(Some(value == 1.0))
}
