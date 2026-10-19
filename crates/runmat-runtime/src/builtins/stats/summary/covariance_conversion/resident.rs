//! Owned covariance outputs and output-count publication.
use super::*;

pub(super) struct CovarianceGpuEval {
    pub(super) provider: &'static dyn AccelProvider,
    pub(super) correlation: GpuTensorHandle,
    pub(super) sigma: GpuTensorHandle,
}

impl CovarianceGpuEval {
    pub(super) fn from_provider(
        name: &'static str,
        provider: &'static dyn AccelProvider,
        input: &GpuTensorHandle,
        result: ProviderCovarianceToCorrelationResult,
    ) -> BuiltinResult<Self> {
        let dimension = match input.shape() {
            [rows, columns] if rows == columns => *rows,
            _ => {
                let error =
                    conversion_error(name, format!("{name}: invalid covariance matrix shape"));
                let error = gpu_helpers::reject_gpu_output(result.correlation, error);
                return Err(gpu_helpers::reject_gpu_output(result.sigma, error));
            }
        };
        let valid = |output: &GpuTensorHandle, shape: &[usize]| {
            output.shape() == shape
                && output.physical().element_type() == input.physical().element_type()
                && output.physical().storage() == GpuTensorStorage::Real
                && !runmat_accelerate_api::handle_is_logical(output)
                && output.allocation_lease().validate_live().is_ok()
                && !output.shares_allocation_with(input)
                && gpu_helpers::exact_provider_for_handle(output).is_some_and(|owner| {
                    runmat_accelerate_api::same_provider_instance(owner, provider)
                })
        };
        if !valid(&result.correlation, &[dimension, dimension])
            || !valid(&result.sigma, &[dimension, 1])
            || result.correlation.shares_allocation_with(&result.sigma)
        {
            let error =
                internal_error(name, format!("{name}: invalid provider covariance outputs"));
            let error = gpu_helpers::reject_gpu_output(result.correlation, error);
            return Err(gpu_helpers::reject_gpu_output(result.sigma, error));
        }
        let provenance = input
            .view()
            .provenance()
            .unwrap_or(runmat_accelerate_api::GpuHandleProvenance::Automatic);
        let correlation = result
            .correlation
            .with_provenance(provenance)
            .map_err(|error| {
                crate::runtime_error::with_provider_execution_source(
                    internal_error(name, format!("{name}: {error}")),
                    runmat_accelerate_api::ProviderExecutionError::from_source(error),
                )
            })?;
        let sigma = result.sigma.with_provenance(provenance).map_err(|error| {
            crate::runtime_error::with_provider_execution_source(
                internal_error(name, format!("{name}: {error}")),
                runmat_accelerate_api::ProviderExecutionError::from_source(error),
            )
        })?;
        Ok(Self {
            provider,
            correlation,
            sigma,
        })
    }

    pub(super) fn output(self, kind: ConversionKind, name: &'static str) -> BuiltinResult<Value> {
        let (first, second) = match kind {
            ConversionKind::Corrcov => (self.correlation, self.sigma),
            ConversionKind::Cov2corr => {
                let len = if self.sigma.shape().contains(&0) {
                    0
                } else {
                    self.sigma
                        .shape()
                        .iter()
                        .try_fold(1usize, |count, extent| count.checked_mul(*extent))
                        .ok_or_else(|| {
                            conversion_error(name, format!("{name}: sigma shape overflow"))
                        })?
                };
                let sigma = self
                    .provider
                    .reshape(&self.sigma, &[1, len])
                    .map_err(|error| {
                        let message = format!("{name}: {error}");
                        crate::runtime_error::with_provider_execution_source(
                            conversion_error(name, message),
                            runmat_accelerate_api::ProviderExecutionError::from(error),
                        )
                    })?;
                if sigma.shape() != [1, len]
                    || sigma.physical().element_type() != self.sigma.physical().element_type()
                    || sigma.physical().storage() != self.sigma.physical().storage()
                    || runmat_accelerate_api::handle_is_logical(&sigma)
                    || sigma.shares_allocation_with(&self.correlation)
                    || !gpu_helpers::exact_provider_for_handle(&sigma).is_some_and(|owner| {
                        runmat_accelerate_api::same_provider_instance(owner, self.provider)
                    })
                {
                    return Err(gpu_helpers::reject_gpu_output(
                        sigma,
                        conversion_error(name, format!("{name}: invalid sigma reshape result")),
                    ));
                }
                (sigma, self.correlation)
            }
        };
        match crate::output_count::current_output_count() {
            Some(0) => Ok(Value::output_list(Vec::new())),
            Some(1) => Ok(Value::output_list(vec![gpu_helpers::resident_gpu_value(
                first,
            )?])),
            Some(2) => Ok(Value::output_list(vec![
                gpu_helpers::resident_gpu_value(first)?,
                gpu_helpers::resident_gpu_value(second)?,
            ])),
            Some(_) => Err(conversion_error(
                name,
                format!("{name}: too many output arguments; maximum is 2"),
            )),
            None => gpu_helpers::resident_gpu_value(first),
        }
    }
}
