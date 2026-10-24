use super::*;

pub(super) async fn compute_pwelch(
    columns: &[Vec<Complex<f64>>],
    options: &PwelchOptions,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<PwelchEvaluation> {
    cooperation.completed(0).await?;
    let scaling =
        normalized_window(&options.window, options.scale, options.fs, cooperation).await?;
    let window_len = scaling.values.len();
    let step = window_len - options.noverlap;
    let density_scale = scaling.denominator;
    let mut planner = FftPlanner::<f64>::new();
    let mut fft_plan = match &options.grid {
        FrequencyGrid::Uniform { nfft } => {
            let mut buffer = Vec::new();
            buffer.try_reserve_exact(*nfft).map_err(|cause| {
                window_allocation_error(BUILTIN_NAME, &PWELCH_ERROR_INTERNAL, cause)
            })?;
            buffer.resize(*nfft, Complex::new(0.0, 0.0));
            Some((planner.plan_fft_forward(*nfft), buffer))
        }
        FrequencyGrid::Explicit { .. } => None,
    };
    let rows = match (&options.grid, options.range) {
        (FrequencyGrid::Uniform { nfft }, FrequencyRange::Onesided) => nfft / 2 + 1,
        (FrequencyGrid::Uniform { nfft }, _) => *nfft,
        (FrequencyGrid::Explicit { frequencies }, _) => frequencies.len(),
    };
    let cols = columns.len();
    let elements = checked_output_elements(rows, cols)?;
    let mut pxx = Vec::new();
    pxx.try_reserve_exact(elements)
        .map_err(|cause| window_allocation_error(BUILTIN_NAME, &PWELCH_ERROR_INTERNAL, cause))?;
    let mut selected_f = Vec::new();

    for column in columns {
        cooperation.completed(0).await?;
        let selected = match (&options.grid, &mut fft_plan) {
            (FrequencyGrid::Uniform { nfft }, Some((fft, buffer))) => {
                let accum = column_uniform_periodogram(
                    column,
                    &scaling.values,
                    step,
                    density_scale,
                    options.trace,
                    *nfft,
                    fft,
                    buffer,
                    cooperation,
                )
                .await?;
                select_range(accum, *nfft, options.fs, options.range, cooperation).await?
            }
            (FrequencyGrid::Explicit { frequencies }, None) => {
                column_explicit_frequency_periodogram(
                    column,
                    &scaling.values,
                    step,
                    density_scale,
                    options.trace,
                    frequencies,
                    options.fs,
                    cooperation,
                )
                .await?
            }
            _ => return Err(pwelch_error(&PWELCH_ERROR_INTERNAL)),
        };
        if selected_f.is_empty() {
            selected_f = selected.f;
        }
        cooperation.completed(selected.pxx.len() as u64).await?;
        pxx.extend(selected.pxx);
    }
    Ok(PwelchEvaluation {
        pxx,
        f: selected_f,
        columns: cols,
    })
}

pub(super) fn checked_output_elements(rows: usize, columns: usize) -> BuiltinResult<usize> {
    rows.checked_mul(columns).ok_or_else(|| {
        let mut error = pwelch_error_with_detail(
            &PWELCH_ERROR_INTERNAL,
            "spectral output dimensions overflow",
        );
        error.gpu_gather_retry = crate::GpuGatherRetry::Never;
        error
    })
}

pub(super) async fn column_uniform_periodogram(
    column: &[Complex<f64>],
    window: &[f64],
    step: usize,
    density_scale: f64,
    trace: WelchTrace,
    nfft: usize,
    fft: &std::sync::Arc<dyn rustfft::Fft<f64>>,
    buffer: &mut [Complex<f64>],
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<Vec<f64>> {
    let starts = segment_starts(column.len(), window.len(), step, cooperation).await?;
    let mut accum = Vec::new();
    accum
        .try_reserve_exact(nfft)
        .map_err(|cause| window_allocation_error(BUILTIN_NAME, &PWELCH_ERROR_INTERNAL, cause))?;
    accum.resize(nfft, 0.0);
    let mut work = CompletedWork::default();
    for (segment, &start) in starts.iter().enumerate() {
        cooperation.completed(0).await?;
        for chunk in buffer.chunks_mut(SIGNAL_WORK_CHUNK as usize) {
            chunk.fill(Complex::new(0.0, 0.0));
            cooperation.completed(chunk.len() as u64).await?;
        }
        for idx in 0..window.len() {
            let sample = column
                .get(start + idx)
                .copied()
                .unwrap_or_else(|| Complex::new(0.0, 0.0));
            // When the DFT is shorter than the segment, fold windowed samples
            // modulo nfft before transforming. For nfft >= window length each
            // destination is visited once, preserving the zero-padding path.
            buffer[idx % nfft] += sample * window[idx];
            if let Some(units) = work.record(1) {
                cooperation.completed(units).await?;
            }
        }
        // RustFFT is a synchronous leaf; cancellation is checked on both sides.
        work.finish(cooperation).await?;
        fft.process(buffer);
        cooperation.completed(buffer.len() as u64).await?;
        for (dst, value) in accum.iter_mut().zip(buffer.iter()) {
            let observed = value.norm_sqr() / density_scale;
            *dst = combine_trace(*dst, observed, segment, trace);
            if let Some(units) = work.record(1) {
                cooperation.completed(units).await?;
            }
        }
        work.finish(cooperation).await?;
    }
    work.finish(cooperation).await?;
    Ok(accum)
}

pub(super) async fn column_explicit_frequency_periodogram(
    column: &[Complex<f64>],
    window: &[f64],
    step: usize,
    density_scale: f64,
    trace: WelchTrace,
    frequencies: &[f64],
    fs: Option<f64>,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<PwelchEvaluation> {
    let starts = segment_starts(column.len(), window.len(), step, cooperation).await?;
    let mut pxx = Vec::new();
    pxx.try_reserve_exact(frequencies.len())
        .map_err(|cause| window_allocation_error(BUILTIN_NAME, &PWELCH_ERROR_INTERNAL, cause))?;
    pxx.resize(frequencies.len(), 0.0);
    let fs_scale = fs.map(|fs| 2.0 * std::f64::consts::PI / fs);
    let mut work = CompletedWork::default();
    for (segment, &start) in starts.iter().enumerate() {
        cooperation.completed(0).await?;
        for (freq_idx, &frequency) in frequencies.iter().enumerate() {
            let omega = fs_scale.map_or(frequency, |scale| frequency * scale);
            let mut sum = Complex::new(0.0, 0.0);
            for idx in 0..window.len() {
                let sample = column
                    .get(start + idx)
                    .copied()
                    .unwrap_or_else(|| Complex::new(0.0, 0.0));
                let phase = -omega * idx as f64;
                let twiddle = Complex::new(phase.cos(), phase.sin());
                sum += sample * window[idx] * twiddle;
                if let Some(units) = work.record(1) {
                    cooperation.completed(units).await?;
                }
            }
            pxx[freq_idx] = combine_trace(
                pxx[freq_idx],
                sum.norm_sqr() / density_scale,
                segment,
                trace,
            );
            if let Some(units) = work.record(1) {
                cooperation.completed(units).await?;
            }
        }
    }
    work.finish(cooperation).await?;
    let mut f = reserved_f64(frequencies.len())?;
    f.extend_from_slice(frequencies);
    Ok(PwelchEvaluation { pxx, f, columns: 1 })
}

/// Segment powers are nonnegative. An online mean avoids overflow in the sum
/// when the final mean remains representable.
pub(super) fn combine_trace(
    current: f64,
    observed: f64,
    completed: usize,
    trace: WelchTrace,
) -> f64 {
    if completed == 0 {
        return observed;
    }
    match trace {
        WelchTrace::Mean => current + (observed - current) / (completed as f64 + 1.0),
        WelchTrace::MaxHold if current.is_nan() || observed.is_nan() => f64::NAN,
        WelchTrace::MaxHold => current.max(observed),
        WelchTrace::MinHold if current.is_nan() || observed.is_nan() => f64::NAN,
        WelchTrace::MinHold => current.min(observed),
    }
}

fn reserved_f64(len: usize) -> BuiltinResult<Vec<f64>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|cause| window_allocation_error(BUILTIN_NAME, &PWELCH_ERROR_INTERNAL, cause))?;
    Ok(values)
}

pub(super) async fn select_range(
    mut power: Vec<f64>,
    nfft: usize,
    fs: Option<f64>,
    range: FrequencyRange,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<PwelchEvaluation> {
    let mut work = CompletedWork::default();
    match range {
        FrequencyRange::Twosided => {
            let f = frequency_vector(nfft, fs, range, cooperation).await?;
            Ok(PwelchEvaluation {
                pxx: power,
                f,
                columns: 1,
            })
        }
        FrequencyRange::Centered => {
            let shift = centered_shift(nfft);
            power.rotate_left(shift);
            cooperation.completed(nfft as u64).await?;
            let f = frequency_vector(nfft, fs, range, cooperation).await?;
            Ok(PwelchEvaluation {
                pxx: power,
                f,
                columns: 1,
            })
        }
        FrequencyRange::Onesided => {
            let len = nfft / 2 + 1;
            let even = nfft.is_multiple_of(2);
            let mut pxx = reserved_f64(len)?;
            pxx.extend_from_slice(&power[..len]);
            for (idx, value) in pxx.iter_mut().enumerate() {
                let is_dc = idx == 0;
                let is_nyquist = even && idx == len - 1;
                if !is_dc && !is_nyquist {
                    *value *= 2.0;
                }
                if let Some(units) = work.record(1) {
                    cooperation.completed(units).await?;
                }
            }
            work.finish(cooperation).await?;
            let f = frequency_vector(nfft, fs, range, cooperation).await?;
            Ok(PwelchEvaluation { pxx, f, columns: 1 })
        }
    }
}

pub(super) async fn frequency_vector(
    nfft: usize,
    fs: Option<f64>,
    range: FrequencyRange,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<Vec<f64>> {
    let len = match range {
        FrequencyRange::Onesided => nfft / 2 + 1,
        FrequencyRange::Twosided | FrequencyRange::Centered => nfft,
    };
    let mut f = reserved_f64(len)?;
    let freq_scale = fs.unwrap_or(2.0 * std::f64::consts::PI);
    let offset = if range == FrequencyRange::Centered {
        centered_frequency_offset(nfft) as f64
    } else {
        0.0
    };
    let mut work = CompletedWork::default();
    for idx in 0..len {
        f.push(freq_scale * ((idx as f64 - offset) / nfft as f64));
        if let Some(units) = work.record(1) {
            cooperation.completed(units).await?;
        }
    }
    work.finish(cooperation).await?;
    Ok(f)
}

pub(super) fn gpu_range(range: FrequencyRange) -> ProviderSpectralRange {
    match range {
        FrequencyRange::Onesided => ProviderSpectralRange::Onesided,
        FrequencyRange::Twosided => ProviderSpectralRange::Twosided,
        FrequencyRange::Centered => ProviderSpectralRange::Centered,
    }
}

pub(super) async fn segment_starts(
    signal_len: usize,
    window_len: usize,
    step: usize,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<Vec<usize>> {
    cooperation.completed(0).await?;
    if signal_len <= window_len {
        return Ok(vec![0]);
    }
    let count = (signal_len - window_len)
        .checked_div(step)
        .and_then(|count| count.checked_add(1))
        .ok_or_else(|| pwelch_error(&PWELCH_ERROR_INTERNAL))?;
    let mut starts = Vec::new();
    starts
        .try_reserve_exact(count)
        .map_err(|cause| window_allocation_error(BUILTIN_NAME, &PWELCH_ERROR_INTERNAL, cause))?;
    let mut start = 0usize;
    let mut work = CompletedWork::default();
    let last_start = signal_len - window_len;
    while start <= last_start {
        starts.push(start);
        if let Some(units) = work.record(1) {
            cooperation.completed(units).await?;
        }
        let Some(next) = start.checked_add(step) else {
            break;
        };
        start = next;
    }
    work.finish(cooperation).await?;
    if starts.is_empty() {
        starts.push(0);
    }
    Ok(starts)
}

pub(super) async fn default_window(
    input_len: usize,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<Vec<f64>> {
    let section_len = if input_len <= 1 {
        1
    } else {
        ((input_len / 9) * 2 + ((input_len % 9) * 2) / 9).clamp(1, DEFAULT_NFFT_MIN)
    };
    symmetric_window_values(section_len, hamming_coefficient, cooperation)
        .await?
        .map_err(pwelch_window_generation_error)
}
