use super::numerical::combine_trace;
use super::*;

fn preparation_failure(
    cause: impl std::fmt::Display + Into<runmat_accelerate_api::ProviderExecutionError>,
) -> RuntimeError {
    let mut error = pwelch_error_with_detail(
        &PWELCH_ERROR_INTERNAL,
        format!("provider output preparation failed: {cause}"),
    );
    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
    crate::runtime_error::with_provider_execution_source(error, cause.into())
}

pub(super) async fn output_gpu(
    handle: &runmat_accelerate_api::GpuTensorHandle,
    rows: usize,
    cols: usize,
    options: &PwelchOptions,
    protected: &[&runmat_accelerate_api::GpuTensorHandle],
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<Option<Value>> {
    cooperation.completed(1).await?;
    let FrequencyGrid::Uniform { nfft } = options.grid else {
        return Ok(None);
    };
    let scaling =
        normalized_window(&options.window, options.scale, options.fs, cooperation).await?;
    let window_len = scaling.values.len();
    // ColumnSliding requires an in-bounds frame. The host reference route
    // retains the admitted zero-padded short-signal behavior.
    if rows < window_len || nfft < window_len {
        return Ok(None);
    }
    let step = window_len - options.noverlap;
    let starts = segment_starts(rows, window_len, step, cooperation).await?;
    let segment_count = starts.len();
    let input_len = rows
        .checked_mul(cols)
        .ok_or_else(|| pwelch_error(&PWELCH_ERROR_INTERNAL))?;
    let frame_count = segment_count
        .checked_mul(cols)
        .ok_or_else(|| pwelch_error(&PWELCH_ERROR_INTERNAL))?;
    let range = gpu_range(options.range);
    let output_rows = match range {
        ProviderSpectralRange::Onesided => nfft / 2 + 1,
        ProviderSpectralRange::Twosided | ProviderSpectralRange::Centered => nfft,
    };
    checked_output_elements(output_rows, frame_count)?;
    let provider =
        super::super::resident_output::owner(handle, || pwelch_error(&PWELCH_ERROR_INTERNAL))?;
    let request = ProviderSpectralRequest {
        input: handle,
        input_len,
        input_complex: runmat_accelerate_api::handle_storage(handle)
            == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved,
        window: &scaling.values,
        nfft,
        frame_count,
        frame_mode: ProviderSpectralFrameMode::ColumnSliding {
            hop: step,
            input_rows: rows,
            frames_per_column: segment_count,
        },
        range,
        denominator: scaling.denominator,
    };
    let estimate = match provider.uniform_spectral_estimate(&request).await {
        Ok(estimate) => estimate,
        Err(error) if error.is_unsupported() => return Ok(None),
        Err(error) => {
            return Err(crate::runtime_error::with_provider_execution_source(
                pwelch_error_with_detail(
                    &PWELCH_ERROR_INTERNAL,
                    format!("provider spectral estimate failed: {error}"),
                ),
                error,
            ))
        }
    };
    let output_error = || {
        pwelch_error_with_detail(
            &PWELCH_ERROR_INTERNAL,
            "provider returned malformed spectral output",
        )
    };
    let estimate = super::super::resident_output::spectral(
        estimate,
        &request,
        provider,
        protected,
        output_error,
    )?;
    // A suspended call must hold only admitted provider results. Raw handles
    // are validated as one output batch before this cancellation point.
    cooperation.completed(1).await?;
    let context = super::super::resident_output::OutputContext {
        input: handle,
        provider,
        caller_inputs: protected,
    };

    let ps_frames = match provider.reshape(&estimate.ps, &[estimate.rows, segment_count, cols]) {
        Ok(ps_frames) => ps_frames,
        Err(err) => return Err(preparation_failure(err)),
    };
    let mut contract = super::super::resident_output::floating_contract(
        handle,
        vec![estimate.rows, segment_count, cols],
        runmat_accelerate_api::GpuTensorStorage::Real,
    );
    let ps_frames = context.admit(ps_frames, &contract, &[], output_error)?;
    let ps_mean = match options.trace {
        WelchTrace::Mean => provider
            .reduce_mean_nd(&ps_frames, &[1])
            .await
            .map_err(|error| preparation_failure(error.after_effect()))?
            .ok_or_else(|| {
                preparation_failure(
                    runmat_accelerate_api::ProviderExecutionError::from(anyhow::anyhow!(
                        "pwelch: required resident mean unavailable"
                    ))
                    .after_effect(),
                )
            })?,
        WelchTrace::MaxHold | WelchTrace::MinHold => {
            let result = match options.trace {
                WelchTrace::MaxHold => provider.reduce_max_dim(&ps_frames, 1).await,
                WelchTrace::MinHold => provider.reduce_min_dim(&ps_frames, 1).await,
                WelchTrace::Mean => unreachable!("mean handled above"),
            };
            let result = match result {
                Ok(result) => result,
                Err(error) if error.is_unsupported() => {
                    return output_host_hold_from_frames(
                        provider,
                        handle,
                        &ps_frames,
                        estimate.rows,
                        segment_count,
                        cols,
                        options,
                        protected,
                        cooperation,
                    )
                    .await
                    .map(Some);
                }
                Err(error) => return Err(preparation_failure(error)),
            };
            // An extrema provider produces both values and indices. Validate the
            // unused index before discarding it, so malformed or borrowed output
            // never escapes as a successful spectral result.
            let expected = [estimate.rows, 1, cols];
            let index_precision = runmat_accelerate_api::handle_precision(&result.indices);
            let index_contract = crate::builtins::common::gpu_helpers::BinaryGpuOutputContract {
                shape: expected.to_vec(),
                storage: runmat_accelerate_api::GpuTensorStorage::Real,
                precision: index_precision,
                integer: None,
                logical: false,
                alias: crate::builtins::common::gpu_helpers::GpuOutputAliasPolicy::RequireDistinctAllocation,
            };
            if !matches!(
                index_precision,
                Some(
                    runmat_accelerate_api::ProviderPrecision::F32
                        | runmat_accelerate_api::ProviderPrecision::F64
                )
            ) || (index_precision == Some(runmat_accelerate_api::ProviderPrecision::F32)
                && segment_count > (1 << 24))
                || result.indices.shares_allocation_with(&result.values)
            {
                let error = output_error();
                return Err(crate::builtins::common::gpu_helpers::reject_gpu_output(
                    result.indices,
                    crate::builtins::common::gpu_helpers::reject_gpu_output(result.values, error),
                ));
            }
            let index = match context.admit(
                result.indices,
                &index_contract,
                &[&ps_frames, &estimate.s, &estimate.ps, &result.values],
                output_error,
            ) {
                Ok(index) => index,
                Err(error) => {
                    return Err(crate::builtins::common::gpu_helpers::reject_gpu_output(
                        result.values,
                        error,
                    ));
                }
            };
            let mut value_contract = contract.clone();
            value_contract.shape = expected.to_vec();
            let values = context.admit(
                result.values,
                &value_contract,
                &[&ps_frames, &estimate.s, &estimate.ps, &index],
                output_error,
            )?;
            drop(index);
            values
        }
    };
    contract.shape = vec![estimate.rows, 1, cols];
    let mut protected_mean = vec![&estimate.s];
    if segment_count > 1 {
        protected_mean.extend([&ps_frames, &estimate.ps]);
    }
    let ps_mean = context.admit(ps_mean, &contract, &protected_mean, output_error)?;
    cooperation.completed(1).await?;
    drop(protected_mean);
    let pxx = match provider.reshape(&ps_mean, &[estimate.rows, cols]) {
        Ok(pxx) => pxx,
        Err(err) => return Err(preparation_failure(err)),
    };
    contract.shape = vec![estimate.rows, cols];
    let mut protected_final = vec![&estimate.s];
    if segment_count > 1 {
        protected_final.extend([&ps_frames, &estimate.ps]);
    }
    let pxx = context.admit(pxx, &contract, &protected_final, output_error)?;
    drop(protected_final);

    drop(ps_frames);
    drop(ps_mean);
    drop(estimate.s);
    drop(estimate.ps);
    let pxx = crate::builtins::common::gpu_helpers::resident_gpu_value(pxx)?;
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            return Ok(Some(Value::output_list(Vec::new())));
        }
        if out_count == 1 {
            return Ok(Some(Value::output_list(vec![pxx])));
        }
        let f_values = frequency_vector(nfft, options.fs, options.range, cooperation).await?;
        let f_shape = [estimate.rows, 1usize];
        let f = SignalPrecision::of_handle(handle)
            .real(f_values, f_shape.to_vec())
            .map_err(|cause| pwelch_error_with_detail(&PWELCH_ERROR_INTERNAL, cause))?;
        let mut protected_frequency = protected.to_vec();
        let Value::GpuTensor(pxx_handle) = &pxx else {
            unreachable!("adopted PSD")
        };
        protected_frequency.push(pxx_handle);
        let f = super::super::resident_output::restore(
            handle,
            Value::Tensor(f),
            BUILTIN_NAME,
            &protected_frequency,
        )?;
        return Ok(Some(crate::output_count::output_list_with_padding(
            out_count,
            vec![pxx, f],
        )));
    }
    Ok(Some(pxx))
}

/// A provider may implement the FFT/framing hook but not dimensional extrema.
/// Download its already-computed frames once; do not replay the spectral work.
async fn output_host_hold_from_frames(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    input: &runmat_accelerate_api::GpuTensorHandle,
    frames: &runmat_accelerate_api::GpuTensorHandle,
    rows: usize,
    segments: usize,
    columns: usize,
    options: &PwelchOptions,
    protected: &[&runmat_accelerate_api::GpuTensorHandle],
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<Value> {
    let downloaded =
        crate::builtins::common::gpu_helpers::download_value_preserving_residency_async(
            provider, frames,
        )
        .await
        .map_err(|source| {
            let span = source.span;
            let mut error = crate::build_runtime_error(format!(
                "pwelch: provider spectral-frame download failed: {}",
                source.message()
            ))
            .with_builtin(BUILTIN_NAME)
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .with_source(source);
            if let Some(identifier) = PWELCH_ERROR_INTERNAL.identifier {
                error = error.with_identifier(identifier);
            }
            if let Some(span) = span {
                error = error.with_span(span);
            }
            error.build()
        })?;
    cooperation.completed(0).await?;
    let Value::Tensor(tensor) = downloaded else {
        return Err(pwelch_error(&PWELCH_ERROR_INTERNAL));
    };
    if tensor.shape() != [rows, segments, columns] {
        return Err(pwelch_error(&PWELCH_ERROR_INTERNAL));
    }
    let values = tensor.materialize_f64();
    let mut pxx = Vec::new();
    pxx.try_reserve_exact(checked_output_elements(rows, columns)?)
        .map_err(|cause| window_allocation_error(BUILTIN_NAME, &PWELCH_ERROR_INTERNAL, cause))?;
    let mut work = CompletedWork::default();
    for column in 0..columns {
        for frequency in 0..rows {
            let mut selected = values[frequency + column * rows * segments];
            for segment in 1..segments {
                selected = combine_trace(
                    selected,
                    values[frequency + segment * rows + column * rows * segments],
                    segment,
                    options.trace,
                );
                if let Some(units) = work.record(1) {
                    cooperation.completed(units).await?;
                }
            }
            pxx.push(selected);
        }
    }
    work.finish(cooperation).await?;
    let FrequencyGrid::Uniform { nfft } = options.grid else {
        unreachable!("resident spectral frames require a uniform grid")
    };
    let f = frequency_vector(nfft, options.fs, options.range, cooperation).await?;
    let output = output_eval(
        PwelchEvaluation { pxx, f, columns },
        SignalPrecision::of_handle(input),
    )?;
    super::super::resident_output::restore(input, output, BUILTIN_NAME, protected)
}
