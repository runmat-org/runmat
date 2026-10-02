use super::*;

pub(super) async fn compute_pwelch(
    columns: &[Vec<Complex<f64>>],
    options: &PwelchOptions,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<PwelchEvaluation> {
    cooperation.completed(0).await?;
    let window_len = options.window.len();
    let step = window_len - options.noverlap;
    let mut scale = 0.0;
    for value in &options.window {
        scale += value * value;
        cooperation.completed(1).await?;
    }
    if scale <= 0.0 || !scale.is_finite() {
        return Err(pwelch_error(&PWELCH_ERROR_INVALID_WINDOW));
    }
    let density_scale = match options.scale {
        SpectrumScale::Psd => options.fs.unwrap_or(2.0 * std::f64::consts::PI) * scale,
        SpectrumScale::Power => scale,
    };
    let mut planner = FftPlanner::<f64>::new();
    let mut fft_plan = match &options.grid {
        FrequencyGrid::Uniform { nfft } => Some((
            planner.plan_fft_forward(*nfft),
            vec![Complex::new(0.0, 0.0); *nfft],
        )),
        FrequencyGrid::Explicit { .. } => None,
    };
    let mut selected_columns = Vec::new();
    let mut selected_f = Vec::new();

    for column in columns {
        cooperation.completed(0).await?;
        let selected = match (&options.grid, &mut fft_plan) {
            (FrequencyGrid::Uniform { nfft }, Some((fft, buffer))) => {
                let accum = column_uniform_periodogram(
                    column,
                    &options.window,
                    step,
                    density_scale,
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
                    &options.window,
                    step,
                    density_scale,
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
        selected_columns.push(selected.pxx);
    }
    let rows = selected_f.len();
    let cols = selected_columns.len();
    let mut pxx = Vec::with_capacity(rows * cols);
    for column in selected_columns {
        cooperation.completed(column.len() as u64).await?;
        pxx.extend(column);
    }
    Ok(PwelchEvaluation {
        pxx,
        f: selected_f,
        columns: cols,
    })
}

pub(super) async fn column_uniform_periodogram(
    column: &[Complex<f64>],
    window: &[f64],
    step: usize,
    density_scale: f64,
    nfft: usize,
    fft: &std::sync::Arc<dyn rustfft::Fft<f64>>,
    buffer: &mut [Complex<f64>],
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<Vec<f64>> {
    let starts = segment_starts(column.len(), window.len(), step, cooperation).await?;
    let mut accum = vec![0.0; nfft];
    for &start in &starts {
        cooperation.completed(0).await?;
        buffer.fill(Complex::new(0.0, 0.0));
        cooperation.completed(buffer.len() as u64).await?;
        for idx in 0..window.len() {
            let sample = column
                .get(start + idx)
                .copied()
                .unwrap_or_else(|| Complex::new(0.0, 0.0));
            buffer[idx] = sample * window[idx];
            cooperation.completed(1).await?;
        }
        // RustFFT is a synchronous leaf; cancellation is checked on both sides.
        cooperation.completed(0).await?;
        fft.process(buffer);
        cooperation.completed(buffer.len() as u64).await?;
        for (dst, value) in accum.iter_mut().zip(buffer.iter()) {
            *dst += value.norm_sqr() / density_scale;
            cooperation.completed(1).await?;
        }
    }
    let segment_count = starts.len() as f64;
    for value in &mut accum {
        *value /= segment_count;
        cooperation.completed(1).await?;
    }
    Ok(accum)
}

pub(super) async fn column_explicit_frequency_periodogram(
    column: &[Complex<f64>],
    window: &[f64],
    step: usize,
    density_scale: f64,
    frequencies: &[f64],
    fs: Option<f64>,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<PwelchEvaluation> {
    let starts = segment_starts(column.len(), window.len(), step, cooperation).await?;
    let mut pxx = vec![0.0; frequencies.len()];
    let fs_scale = fs.map(|fs| 2.0 * std::f64::consts::PI / fs);
    for &start in &starts {
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
                cooperation.completed(1).await?;
            }
            pxx[freq_idx] += sum.norm_sqr() / density_scale;
            cooperation.completed(1).await?;
        }
    }
    let segment_count = starts.len() as f64;
    for value in &mut pxx {
        *value /= segment_count;
        cooperation.completed(1).await?;
    }
    Ok(PwelchEvaluation {
        pxx,
        f: frequencies.to_vec(),
        columns: 1,
    })
}

pub(super) async fn select_range(
    mut power: Vec<f64>,
    nfft: usize,
    fs: Option<f64>,
    range: FrequencyRange,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<PwelchEvaluation> {
    let freq_scale = fs.unwrap_or(2.0 * std::f64::consts::PI);
    match range {
        FrequencyRange::Twosided => {
            let mut f = Vec::with_capacity(nfft);
            for idx in 0..nfft {
                f.push(freq_scale * idx as f64 / nfft as f64);
                cooperation.completed(1).await?;
            }
            Ok(PwelchEvaluation {
                pxx: power,
                f,
                columns: 1,
            })
        }
        FrequencyRange::Centered => {
            let shift = nfft.div_ceil(2);
            power.rotate_left(shift);
            cooperation.completed(nfft as u64).await?;
            let offset = nfft / 2;
            let mut f = Vec::with_capacity(nfft);
            for idx in 0..nfft {
                f.push(freq_scale * (idx as f64 - offset as f64) / nfft as f64);
                cooperation.completed(1).await?;
            }
            Ok(PwelchEvaluation {
                pxx: power,
                f,
                columns: 1,
            })
        }
        FrequencyRange::Onesided => {
            let len = nfft / 2 + 1;
            let even = nfft.is_multiple_of(2);
            let mut pxx = power[..len].to_vec();
            for (idx, value) in pxx.iter_mut().enumerate() {
                let is_dc = idx == 0;
                let is_nyquist = even && idx == len - 1;
                if !is_dc && !is_nyquist {
                    *value *= 2.0;
                }
                cooperation.completed(1).await?;
            }
            let mut f = Vec::with_capacity(len);
            for idx in 0..len {
                f.push(freq_scale * idx as f64 / nfft as f64);
                cooperation.completed(1).await?;
            }
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
    Ok(select_range(vec![0.0; nfft], nfft, fs, range, cooperation)
        .await?
        .f)
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
    let mut starts = Vec::new();
    let mut start = 0usize;
    while start + window_len <= signal_len {
        starts.push(start);
        start += step;
        cooperation.completed(1).await?;
    }
    if starts.is_empty() {
        starts.push(0);
    }
    Ok(starts)
}

pub(super) fn default_window(input_len: usize) -> Vec<f64> {
    let section_len = if input_len <= 1 {
        1
    } else {
        ((2 * input_len) / 9).clamp(1, DEFAULT_NFFT_MIN)
    };
    hamming_window(section_len)
}

pub(super) fn hamming_window(len: usize) -> Vec<f64> {
    (0..len).map(|idx| hamming_coefficient(idx, len)).collect()
}

pub(super) async fn cooperative_hamming_window(
    len: usize,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<Vec<f64>> {
    let mut values = Vec::with_capacity(len);
    for idx in 0..len {
        values.push(hamming_coefficient(idx, len));
        cooperation.completed(1).await?;
    }
    Ok(values)
}

fn hamming_coefficient(idx: usize, len: usize) -> f64 {
    if len == 1 {
        return 1.0;
    }
    let phase = 2.0 * std::f64::consts::PI * idx as f64 / (len - 1) as f64;
    0.54 - 0.46 * phase.cos()
}
