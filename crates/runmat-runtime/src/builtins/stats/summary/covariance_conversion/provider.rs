use super::*;

fn provider_is_unsupported(err: &anyhow::Error) -> bool {
    runmat_accelerate_api::is_unsupported_provider_operation(err)
}

#[cfg(test)]
mod tests {
    use super::provider_is_unsupported;

    #[test]
    fn provider_fallback_requires_typed_unsupported_not_error_wording() {
        assert!(!provider_is_unsupported(&anyhow::anyhow!(
            "device failed while processing an unsupported request"
        )));
        assert!(provider_is_unsupported(
            &runmat_accelerate_api::unsupported_provider_operation("fixture unavailable")
        ));
    }
}

pub(super) fn try_covariance_gpu(
    name: &'static str,
    value: &Value,
) -> BuiltinResult<Option<CovarianceGpuEval>> {
    let Value::GpuTensor(handle) = value else {
        return Ok(None);
    };
    if runmat_accelerate_api::handle_storage(handle) == GpuTensorStorage::ComplexInterleaved {
        return Err(conversion_error(
            name,
            format!("{name}: complex covariance matrices are not supported"),
        ));
    }
    if runmat_accelerate_api::handle_integer_type(handle).is_some()
        || runmat_accelerate_api::handle_is_logical(handle)
    {
        return Ok(None);
    }
    let Some(provider) = gpu_helpers::exact_provider_for_handle(handle) else {
        return Ok(None);
    };
    match provider.covariance_to_correlation(handle) {
        Ok(result) => CovarianceGpuEval::from_provider(name, provider, handle, result).map(Some),
        Err(err) if provider_is_unsupported(&err) => Ok(None),
        Err(err) => Err(crate::runtime_error::with_provider_execution_source(
            conversion_error(name, format!("{name}: {err}")),
            runmat_accelerate_api::ProviderExecutionError::from(err),
        )),
    }
}
