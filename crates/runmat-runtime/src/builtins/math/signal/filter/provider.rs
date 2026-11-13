//! Causal resident submission and its one checked coefficient preparation.

use super::preparation::{
    prepared_real_coefficients, provider_resource_error, to_real_vec, PREPARATION_CHUNK,
};
use super::*;
use runmat_accelerate_api::{
    upload_numeric_protecting, GpuTensorStorage, HostNumericDataView, HostNumericTensorView,
    NumericUploadError, ProviderIirCoefficientSource, ProviderPrecision,
    ProviderPreparedIirCoefficients,
};

#[derive(Clone, Copy)]
enum UploadRole {
    Numerator,
    Denominator,
    InitialState,
}

impl UploadRole {
    fn label(self) -> &'static str {
        match self {
            Self::Numerator => "numerator",
            Self::Denominator => "denominator",
            Self::InitialState => "initial state",
        }
    }
}

fn upload_error(role: UploadRole, source: NumericUploadError) -> RuntimeError {
    let mut error = filter_error_with_detail(
        &FILTER_ERROR_PROVIDER_UPLOAD,
        format!("{} upload failed", role.label()),
    );
    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
    crate::runtime_error::with_provider_execution_source(
        error,
        runmat_accelerate_api::ProviderExecutionError::from_source(source),
    )
}

enum Coefficients {
    Prepared(ProviderPreparedIirCoefficients),
    Resident {
        numerator: GpuTensorHandle,
        denominator: GpuTensorHandle,
    },
}

impl Coefficients {
    fn numerator(&self) -> &GpuTensorHandle {
        match self {
            Self::Prepared(prepared) => prepared.numerator(),
            Self::Resident { numerator, .. } => numerator,
        }
    }

    fn denominator(&self) -> &GpuTensorHandle {
        match self {
            Self::Prepared(prepared) => prepared.denominator(),
            Self::Resident { denominator, .. } => denominator,
        }
    }

    fn source(&self) -> ProviderIirCoefficientSource<'_> {
        match self {
            Self::Prepared(prepared) => ProviderIirCoefficientSource::Prepared(prepared),
            Self::Resident {
                numerator,
                denominator,
            } => ProviderIirCoefficientSource::Resident {
                numerator,
                denominator,
            },
        }
    }
}

pub(super) async fn try_filter_gpu(
    args: &FilterArgs,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<Option<FilterEvaluation>> {
    #[cfg(all(test, feature = "wgpu"))]
    {
        if runmat_accelerate_api::provider().is_none() {
            let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
            );
        }
    }
    if args.is_complex {
        return Ok(None);
    }
    let signal = match args.signal.resident() {
        Some(handle)
            if runmat_accelerate_api::handle_integer_type(handle).is_none()
                && !runmat_accelerate_api::handle_is_logical(handle) =>
        {
            handle
        }
        None | Some(_) => return Ok(None),
    };
    if let Some(initial) = args.initial.gpu_handle.as_ref() {
        gpu_helpers::require_resident_compute_owners(&[signal, initial], false, || {
            filter_error(&FILTER_ERROR_INTERNAL)
        })?;
    } else {
        gpu_helpers::require_resident_compute_owner(signal, || {
            filter_error(&FILTER_ERROR_INTERNAL)
        })?;
    }
    let Some(provider) = gpu_helpers::exact_provider_for_handle(signal) else {
        return Ok(None);
    };
    if args.initial.provided {
        if let Some(handle) = &args.initial.gpu_handle {
            if !gpu_helpers::exact_provider_for_handle(handle)
                .is_some_and(|owner| runmat_accelerate_api::same_provider_instance(owner, provider))
            {
                return Ok(None);
            }
        }
    }
    let query = runmat_accelerate_api::provider_iir_causal_query(signal, args.order, args.dim_idx)
        .map_err(|source| {
            crate::runtime_error::with_provider_execution_source(
                filter_error(&FILTER_ERROR_INTERNAL),
                source.into(),
            )
        })?;
    if !provider.query_feasibility(&query).is_supported() {
        return Ok(None);
    }
    let signal_precision = runmat_accelerate_api::handle_precision(signal);
    let coefficients = if signal_precision == Some(provider.precision()) {
        let Some((numerator, denominator)) = prepared_real_coefficients(
            &args.coeffs_b.data,
            &args.coeffs_a.data,
            args.order,
            provider.precision(),
            cooperation,
        )
        .await?
        else {
            return Ok(None);
        };
        let mut protected = vec![signal];
        if let Some(initial) = args.initial.gpu_handle.as_ref() {
            protected.push(initial);
        }
        if let Some(original) = args.coeffs_b.original_handle.as_ref() {
            protected.push(original);
        }
        if let Some(original) = args.coeffs_a.original_handle.as_ref() {
            protected.push(original);
        }
        let prepared = ProviderPreparedIirCoefficients::upload(
            provider,
            &numerator,
            &denominator,
            provider.precision(),
            &protected,
        )
        .map_err(|source| {
            let mut error = filter_error(&FILTER_ERROR_PROVIDER_UPLOAD);
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            crate::runtime_error::with_provider_execution_source(
                error,
                runmat_accelerate_api::ProviderExecutionError::from_source(source),
            )
        })?;
        Coefficients::Prepared(prepared)
    } else {
        // InProcess also accepts F32 signal buffers while its preferred upload
        // precision is F64. Preserve that supported mixed-precision hook form.
        let Some(b_real) = to_real_vec(&args.coeffs_b.data, cooperation).await? else {
            return Ok(None);
        };
        let Some(a_real) = to_real_vec(&args.coeffs_a.data, cooperation).await? else {
            return Ok(None);
        };
        let b_shape = [args.coeffs_b.len, 1];
        let a_shape = [args.coeffs_a.len, 1];
        let mut protected = vec![signal];
        if let Some(initial) = args.initial.gpu_handle.as_ref() {
            protected.push(initial);
        }
        if let Some(original) = args.coeffs_b.original_handle.as_ref() {
            protected.push(original);
        }
        if let Some(original) = args.coeffs_a.original_handle.as_ref() {
            protected.push(original);
        }
        let numerator = upload_numeric_protecting(
            provider,
            &HostNumericTensorView {
                data: HostNumericDataView::F64(&b_real),
                shape: &b_shape,
                storage: GpuTensorStorage::Real,
            },
            &protected,
        )
        .map_err(|source| upload_error(UploadRole::Numerator, source))?;
        protected.push(&numerator);
        let denominator = upload_numeric_protecting(
            provider,
            &HostNumericTensorView {
                data: HostNumericDataView::F64(&a_real),
                shape: &a_shape,
                storage: GpuTensorStorage::Real,
            },
            &protected,
        )
        .map_err(|source| upload_error(UploadRole::Denominator, source))?;
        Coefficients::Resident {
            numerator,
            denominator,
        }
    };
    let (initial, _temporary_initial) = if args.state_len == 0 || !args.initial.provided {
        (None, None)
    } else if let Some(handle) = &args.initial.gpu_handle {
        (Some(handle.clone()), None)
    } else {
        let values = to_real_vec(&args.initial.column_major, cooperation)
            .await?
            .ok_or_else(|| filter_error(&FILTER_ERROR_INVALID_INITIAL_STATE))?;
        let mut protected = vec![signal, coefficients.numerator(), coefficients.denominator()];
        if let Some(original) = args.coeffs_b.original_handle.as_ref() {
            protected.push(original);
        }
        if let Some(original) = args.coeffs_a.original_handle.as_ref() {
            protected.push(original);
        }
        let mut single_values = Vec::new();
        if provider.precision() == ProviderPrecision::F32 {
            single_values
                .try_reserve_exact(values.len())
                .map_err(provider_resource_error)?;
            for chunk in values.chunks(PREPARATION_CHUNK) {
                single_values.extend(chunk.iter().map(|value| *value as f32));
                cooperation.completed(chunk.len() as u64).await?;
            }
        }
        let data = if provider.precision() == ProviderPrecision::F32 {
            HostNumericDataView::F32(&single_values)
        } else {
            HostNumericDataView::F64(&values)
        };
        let handle = upload_numeric_protecting(
            provider,
            &HostNumericTensorView {
                data,
                shape: &args.initial.shape,
                storage: GpuTensorStorage::Real,
            },
            &protected,
        )
        .map_err(|source| upload_error(UploadRole::InitialState, source))?;
        (Some(handle.clone()), Some(handle))
    };
    let result = provider
        .iir_filter(
            coefficients.source(),
            signal,
            ProviderIirFilterOptions {
                dim: args.dim_idx,
                zi: initial.clone(),
            },
        )
        .await
        .map_err(|source| {
            let mut error = filter_error(&FILTER_ERROR_PROVIDER_EXECUTION);
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            crate::runtime_error::with_provider_execution_source(
                error,
                runmat_accelerate_api::ProviderExecutionError::from(source).after_effect(),
            )
        })?;
    let (output, final_state) = provider_output::ProviderOutputContext {
        provider,
        signal,
        numerator: coefficients.numerator(),
        denominator: coefficients.denominator(),
        original_numerator: args.coeffs_b.original_handle.as_ref(),
        original_denominator: args.coeffs_a.original_handle.as_ref(),
        initial_state: initial.as_ref(),
        state_shape: &args.state_shape,
        state_len: args.state_len,
    }
    .admit(result)?;
    let final_state = match final_state {
        Some(handle) => Value::GpuTensor(handle),
        None => {
            let values = vec![0.0; tensor::element_count(&args.state_shape)];
            let zeros = if args.signal.output_single {
                Tensor::from_f32(
                    values.into_iter().map(|value| value as f32).collect(),
                    args.state_shape.clone(),
                )
            } else {
                Tensor::new(values, args.state_shape.clone())
            }
            .map_err(|source| runtime_error_for(format!("filter: {source}")))?;
            tensor::tensor_into_value(zeros)
        }
    };
    Ok(Some(FilterEvaluation {
        output: Value::GpuTensor(output),
        final_state,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;

    #[test]
    fn prepared_coefficients_preserve_source_precision_rounding_and_padding() {
        let numerator = [Complex::new(16_777_217.0, 0.0), Complex::new(1.0, 0.0)];
        let denominator = [Complex::new(3.0, 0.0)];
        let mut cooperation = SignalCooperation::new(None, BUILTIN_NAME);
        let (single_b, single_a) = block_on(prepared_real_coefficients(
            &numerator,
            &denominator,
            2,
            ProviderPrecision::F32,
            &mut cooperation,
        ))
        .unwrap()
        .unwrap();
        assert_eq!(single_b, vec![16_777_216.0 / 3.0, 1.0 / 3.0]);
        assert_eq!(single_a, vec![1.0, 0.0]);
        let (double_b, double_a) = block_on(prepared_real_coefficients(
            &numerator,
            &denominator,
            2,
            ProviderPrecision::F64,
            &mut cooperation,
        ))
        .unwrap()
        .unwrap();
        assert_eq!(double_b, vec![16_777_217.0 / 3.0, 1.0 / 3.0]);
        assert_eq!(double_a, vec![1.0, 0.0]);
    }

    #[test]
    fn tiny_nonzero_imaginary_coefficient_never_enters_real_provider_route() {
        let mut cooperation = SignalCooperation::new(None, BUILTIN_NAME);
        let result = block_on(prepared_real_coefficients(
            &[Complex::new(1.0, 1.0e-300)],
            &[Complex::new(1.0, 0.0)],
            1,
            ProviderPrecision::F64,
            &mut cooperation,
        ))
        .unwrap();
        assert!(result.is_none());
    }
}
