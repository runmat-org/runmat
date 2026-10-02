//! Host-staged prepared provider passes and their complete route admission.

use super::*;

fn provider_pass_error(pass: FilterPass, source: RuntimeError) -> RuntimeError {
    let mut builder = build_runtime_error(format!(
        "{}: {} pass failed",
        FILTFILT_ERROR_PASS_FAILED.message,
        pass.label()
    ))
    .with_builtin(BUILTIN_NAME)
    .with_source(source);
    if let Some(identifier) = FILTFILT_ERROR_PASS_FAILED.identifier {
        builder = builder.with_identifier(identifier);
    }
    let mut error = builder.build();
    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
    error
}

#[derive(Clone, Copy)]
enum FilterPass {
    Forward,
    Reverse,
}

impl FilterPass {
    fn label(self) -> &'static str {
        match self {
            Self::Forward => "forward",
            Self::Reverse => "reverse",
        }
    }
}

fn prepared_state_shape(
    signal_shape: &[usize],
    axis: usize,
    order: usize,
) -> BuiltinResult<Vec<usize>> {
    runmat_accelerate_api::provider_iir_state_shape(signal_shape, axis, order).map_err(|source| {
        crate::runtime_error::with_provider_execution_source(
            filtfilt_error(&FILTFILT_ERROR_INTERNAL),
            source.into(),
        )
    })
}

pub(super) async fn try_filtfilt_gpu(
    coeffs: &NormalizedCoefficients,
    input: &ComplexVectorInput,
    output_dtype: NumericDType,
    borrowed: &[&GpuTensorHandle],
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<Option<Value>> {
    let Some(input_handle) = input.gpu_handle.as_ref() else {
        return Ok(None);
    };
    let provider = resident_output::owner(input_handle, || {
        filtfilt_error_with_detail(&FILTFILT_ERROR_INVALID_SIGNAL, "invalid resident signal")
    })?;
    let Some(input_precision) = runmat_accelerate_api::handle_precision(input_handle) else {
        return Ok(None);
    };
    if matches!(output_dtype, NumericDType::F32)
        != matches!(
            provider.precision(),
            runmat_accelerate_api::ProviderPrecision::F32
        )
        || input_precision != provider.precision()
    {
        return Ok(None);
    }
    if input.is_complex || !coeffs.is_real() || coeffs.state_len == 0 {
        return Ok(None);
    }
    let nfact = 3usize
        .checked_mul(coeffs.state_len)
        .ok_or_else(|| filtfilt_error(&FILTFILT_ERROR_INTERNAL))?;
    if input.data.len() <= nfact {
        return Ok(None);
    }

    let zi = lfilter_zi(coeffs, cooperation).await?;
    if zi.iter().any(|z| z.im != 0.0) {
        return Ok(None);
    }
    let extended = odd_reflect(&input.data, nfact)?;
    let b_real = coeffs.b.iter().map(|z| z.re).collect::<Vec<_>>();
    let a_real = coeffs.a.iter().map(|z| z.re).collect::<Vec<_>>();
    let coefficient_len =
        u64::try_from(b_real.len()).map_err(|_| filtfilt_error(&FILTFILT_ERROR_INTERNAL))?;
    let extended_len =
        u64::try_from(extended.len()).map_err(|_| filtfilt_error(&FILTFILT_ERROR_INTERNAL))?;
    let extended_shape = if vector_is_row(&input.shape) {
        vec![1, extended_len]
    } else {
        vec![extended_len, 1]
    };
    let state_len =
        u64::try_from(coeffs.state_len).map_err(|_| filtfilt_error(&FILTFILT_ERROR_INTERNAL))?;
    let state_shape = prepared_state_shape(
        &input.shape,
        usize::from(vector_is_row(&input.shape)),
        coeffs.state_len,
    )?
    .into_iter()
    .map(|extent| u64::try_from(extent).map_err(|_| filtfilt_error(&FILTFILT_ERROR_INTERNAL)))
    .collect::<BuiltinResult<Vec<_>>>()?;
    let element_type = runmat_accelerate_api::ProviderElementType::from(provider.precision());
    let representation = |shape| runmat_accelerate_api::ProviderRepresentation {
        element_type,
        storage: runmat_accelerate_api::ProviderStorage::DenseReal,
        layout: runmat_accelerate_api::ProviderLayout::ColumnMajorContiguous,
        shape,
        residency: runmat_accelerate_api::ProviderResidency::Device,
    };
    let coefficients_shape = vec![coefficient_len, 1];
    let query = runmat_accelerate_api::ProviderFeasibilityQuery {
        operation: runmat_accelerate_api::ProviderOperationIdentity::new(
            runmat_accelerate_api::PROVIDER_IIR_PREPARED_PASS_OPERATION,
        ),
        family: runmat_accelerate_api::ProviderOperationFamily::Library,
        inputs: vec![
            representation(coefficients_shape.clone()),
            representation(coefficients_shape),
            representation(extended_shape.clone()),
            representation(state_shape.clone()),
        ],
        outputs: vec![representation(extended_shape), representation(state_shape)],
        workload: runmat_accelerate_api::ProviderWorkload {
            iir: Some(runmat_accelerate_api::ProviderIirWorkload {
                route: runmat_accelerate_api::ProviderIirRoute::PreparedPass,
                dim: u32::from(vector_is_row(&input.shape)),
                state_len,
                reflection_len: 0,
            }),
            ..Default::default()
        },
    };
    if !provider.query_feasibility(&query).is_supported() {
        return Ok(None);
    }
    let pass_borrowed = std::iter::once(input_handle)
        .chain(borrowed.iter().copied())
        .collect::<Vec<_>>();
    let prepared = ProviderPreparedIirCoefficients::upload(
        provider,
        &b_real,
        &a_real,
        input_precision,
        &pass_borrowed,
    )
    .map_err(|source| {
        let mut error = filtfilt_error_with_detail(
            &FILTFILT_ERROR_PASS_FAILED,
            "coefficient preparation failed",
        );
        error.gpu_gather_retry = crate::GpuGatherRetry::Never;
        error.source = Some(Box::new(source));
        error
    })?;
    let first_pass = gpu_filter_pass(
        provider,
        &prepared,
        coeffs,
        &extended,
        &input.shape,
        &zi,
        &pass_borrowed,
    )
    .await
    .map_err(|source| provider_pass_error(FilterPass::Forward, source))?;
    let mut reversed = first_pass;
    reversed.reverse();
    let second_pass = gpu_filter_pass(
        provider,
        &prepared,
        coeffs,
        &reversed,
        &input.shape,
        &zi,
        &pass_borrowed,
    )
    .await
    .map_err(|source| provider_pass_error(FilterPass::Reverse, source))?;
    let mut restored = second_pass;
    restored.reverse();
    let trimmed = restored[nfact..nfact + input.data.len()]
        .iter()
        .map(|z| z.re)
        .collect::<Vec<_>>();
    let view = runmat_accelerate_api::HostTensorView {
        data: &trimmed,
        shape: &input.shape,
    };
    let handle = provider.upload(&view).map_err(|source| {
        crate::runtime_error::with_provider_execution_source(
            filtfilt_error_with_detail(&FILTFILT_ERROR_PASS_FAILED, "result upload failed"),
            source.into(),
        )
    })?;
    let context = OutputContext {
        input: input_handle,
        provider,
        caller_inputs: borrowed,
    };
    let contract = resident_output::floating_contract(
        input_handle,
        input.shape.clone(),
        runmat_accelerate_api::GpuTensorStorage::Real,
    );
    let output = context.admit(handle, &contract, &[], || {
        filtfilt_error(&FILTFILT_ERROR_PROVIDER_OUTPUT)
    })?;
    Ok(Some(Value::GpuTensor(output)))
}

pub(super) async fn gpu_filter_pass(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    prepared: &ProviderPreparedIirCoefficients,
    coeffs: &NormalizedCoefficients,
    signal: &[Complex<f64>],
    original_shape: &[usize],
    zi: &[Complex<f64>],
    borrowed: &[&GpuTensorHandle],
) -> BuiltinResult<Vec<Complex<f64>>> {
    let is_row = vector_is_row(original_shape);
    let shape = if is_row {
        vec![1, signal.len()]
    } else {
        vec![signal.len(), 1]
    };
    let dim = if is_row { 1 } else { 0 };
    let real_signal = signal.iter().map(|z| z.re).collect::<Vec<_>>();
    let signal_handle = provider
        .upload(&runmat_accelerate_api::HostTensorView {
            data: &real_signal,
            shape: &shape,
        })
        .map_err(|source| {
            crate::runtime_error::with_provider_execution_source(
                filtfilt_error_with_detail(&FILTFILT_ERROR_PASS_FAILED, "signal upload failed"),
                source.into(),
            )
        })?;
    let scaled_zi = scale_state(zi, signal[0])
        .into_iter()
        .map(|z| z.re)
        .collect::<Vec<_>>();
    let zi_shape = prepared_state_shape(&shape, dim, coeffs.state_len)?;
    let zi_handle = provider
        .upload(&runmat_accelerate_api::HostTensorView {
            data: &scaled_zi,
            shape: &zi_shape,
        })
        .map_err(|source| {
            let mut error = filtfilt_error_with_detail(
                &FILTFILT_ERROR_PASS_FAILED,
                "initial-state upload failed",
            );
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            crate::runtime_error::with_provider_execution_source(error, source.into())
        })?;
    let result = provider
        .iir_filter(
            ProviderIirCoefficientSource::Prepared(prepared),
            &signal_handle,
            ProviderIirFilterOptions {
                dim,
                zi: Some(zi_handle.clone()),
            },
        )
        .await
        .map_err(|source| {
            let mut error = filtfilt_error(&FILTFILT_ERROR_PASS_FAILED);
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            crate::runtime_error::with_provider_execution_source(error, source.into())
        })?;
    let runmat_accelerate_api::ProviderIirFilterResult {
        output,
        final_state,
    } = result;
    let protected = borrowed
        .iter()
        .copied()
        .chain([prepared.numerator(), prepared.denominator(), &zi_handle])
        .collect::<Vec<_>>();
    let context = OutputContext {
        input: &signal_handle,
        provider,
        caller_inputs: &protected,
    };
    let Some(final_state) = final_state else {
        return Err(gpu_helpers::reject_gpu_output(
            output,
            filtfilt_error(&FILTFILT_ERROR_PROVIDER_OUTPUT),
        ));
    };
    let output_contract = resident_output::floating_contract(
        &signal_handle,
        shape,
        runmat_accelerate_api::GpuTensorStorage::Real,
    );
    let output = match context.admit(output, &output_contract, &[&final_state], || {
        filtfilt_error(&FILTFILT_ERROR_PROVIDER_OUTPUT)
    }) {
        Ok(output) => output,
        Err(error) => return Err(gpu_helpers::reject_gpu_output(final_state, error)),
    };
    let state_contract = resident_output::floating_contract(
        &signal_handle,
        zi_shape,
        runmat_accelerate_api::GpuTensorStorage::Real,
    );
    if let Err(error) = context.admit(final_state, &state_contract, &[&output], || {
        filtfilt_error(&FILTFILT_ERROR_PROVIDER_OUTPUT)
    }) {
        return Err(gpu_helpers::reject_gpu_output(output, error));
    }
    let tensor = gpu_helpers::gather_tensor_async(&output)
        .await
        .map_err(|source| {
            filtfilt_error_with_source(
                &FILTFILT_ERROR_PASS_FAILED,
                "provider result gather failed",
                source,
                true,
            )
        })?;
    Ok(tensor::tensor_into_values_f64(tensor)
        .into_iter()
        .map(|re| Complex::new(re, 0.0))
        .collect())
}
