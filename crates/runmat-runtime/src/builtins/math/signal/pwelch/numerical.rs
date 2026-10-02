use super::*;

pub(super) fn compute_pwelch(
    columns: &[Vec<Complex<f64>>],
    options: &PwelchOptions,
) -> BuiltinResult<PwelchEvaluation> {
    let window_len = options.window.len();
    let step = window_len - options.noverlap;
    let scale = options.window.iter().map(|v| v * v).sum::<f64>();
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
                );
                select_range(accum, *nfft, options.fs, options.range)
            }
            (FrequencyGrid::Explicit { frequencies }, None) => {
                column_explicit_frequency_periodogram(
                    column,
                    &options.window,
                    step,
                    density_scale,
                    frequencies,
                    options.fs,
                )
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
        pxx.extend(column);
    }
    Ok(PwelchEvaluation {
        pxx,
        f: selected_f,
        columns: cols,
    })
}

pub(super) fn column_uniform_periodogram(
    column: &[Complex<f64>],
    window: &[f64],
    step: usize,
    density_scale: f64,
    nfft: usize,
    fft: &std::sync::Arc<dyn rustfft::Fft<f64>>,
    buffer: &mut [Complex<f64>],
) -> Vec<f64> {
    let starts = segment_starts(column.len(), window.len(), step);
    let mut accum = vec![0.0; nfft];
    for &start in &starts {
        buffer.fill(Complex::new(0.0, 0.0));
        for idx in 0..window.len() {
            let sample = column
                .get(start + idx)
                .copied()
                .unwrap_or_else(|| Complex::new(0.0, 0.0));
            buffer[idx] = sample * window[idx];
        }
        fft.process(buffer);
        for (dst, value) in accum.iter_mut().zip(buffer.iter()) {
            *dst += value.norm_sqr() / density_scale;
        }
    }
    let segment_count = starts.len() as f64;
    for value in &mut accum {
        *value /= segment_count;
    }
    accum
}

pub(super) fn column_explicit_frequency_periodogram(
    column: &[Complex<f64>],
    window: &[f64],
    step: usize,
    density_scale: f64,
    frequencies: &[f64],
    fs: Option<f64>,
) -> PwelchEvaluation {
    let starts = segment_starts(column.len(), window.len(), step);
    let mut pxx = vec![0.0; frequencies.len()];
    let fs_scale = fs.map(|fs| 2.0 * std::f64::consts::PI / fs);
    for &start in &starts {
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
            }
            pxx[freq_idx] += sum.norm_sqr() / density_scale;
        }
    }
    let segment_count = starts.len() as f64;
    for value in &mut pxx {
        *value /= segment_count;
    }
    PwelchEvaluation {
        pxx,
        f: frequencies.to_vec(),
        columns: 1,
    }
}

pub(super) fn select_range(
    mut power: Vec<f64>,
    nfft: usize,
    fs: Option<f64>,
    range: FrequencyRange,
) -> PwelchEvaluation {
    let freq_scale = fs.unwrap_or(2.0 * std::f64::consts::PI);
    match range {
        FrequencyRange::Twosided => {
            let f = (0..nfft)
                .map(|idx| freq_scale * idx as f64 / nfft as f64)
                .collect();
            PwelchEvaluation {
                pxx: power,
                f,
                columns: 1,
            }
        }
        FrequencyRange::Centered => {
            let shift = nfft.div_ceil(2);
            power.rotate_left(shift);
            let offset = nfft / 2;
            let f = (0..nfft)
                .map(|idx| freq_scale * (idx as f64 - offset as f64) / nfft as f64)
                .collect();
            PwelchEvaluation {
                pxx: power,
                f,
                columns: 1,
            }
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
            }
            let f = (0..len)
                .map(|idx| freq_scale * idx as f64 / nfft as f64)
                .collect();
            PwelchEvaluation { pxx, f, columns: 1 }
        }
    }
}

pub(super) fn frequency_vector(nfft: usize, fs: Option<f64>, range: FrequencyRange) -> Vec<f64> {
    select_range(vec![0.0; nfft], nfft, fs, range).f
}

pub(super) fn gpu_range(range: FrequencyRange) -> ProviderSpectralRange {
    match range {
        FrequencyRange::Onesided => ProviderSpectralRange::Onesided,
        FrequencyRange::Twosided => ProviderSpectralRange::Twosided,
        FrequencyRange::Centered => ProviderSpectralRange::Centered,
    }
}

pub(super) fn segment_starts(signal_len: usize, window_len: usize, step: usize) -> Vec<usize> {
    if signal_len <= window_len {
        return vec![0];
    }
    let mut starts = Vec::new();
    let mut start = 0usize;
    while start + window_len <= signal_len {
        starts.push(start);
        start += step;
    }
    if starts.is_empty() {
        starts.push(0);
    }
    starts
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
    if len == 1 {
        return vec![1.0];
    }
    (0..len)
        .map(|idx| {
            let phase = 2.0 * std::f64::consts::PI * idx as f64 / (len - 1) as f64;
            0.54 - 0.46 * phase.cos()
        })
        .collect()
}
