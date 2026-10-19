use super::*;
use crate::builtins::math::interpolation::pp::{
    build_spline_pp, evaluate_pp, is_spline_execution_failure, Extrapolation, NumericSeries,
    PpOutputLayout, QueryPoints, SampleAxis, SplineCooperation,
};
use num_complex::Complex;
use rustfft::FftPlanner;

const EPS: f64 = 1.0e-12;

/// Charge explicit scalar work to the invocation's shared execution budget.
/// Opaque FFT/spline calls are checked immediately before and after; their
/// internal loops are owned by those libraries, not a second signal scheduler.
pub(super) struct NumericalWork<'a> {
    cooperation: &'a mut SignalCooperation,
    pending: u64,
}

impl<'a> NumericalWork<'a> {
    pub(super) fn new(cooperation: &'a mut SignalCooperation) -> Self {
        Self {
            cooperation,
            pending: 0,
        }
    }

    pub(super) async fn tick(&mut self) -> BuiltinResult<()> {
        self.pending += 1;
        if self.pending == super::super::common::SIGNAL_WORK_CHUNK {
            self.boundary().await?;
        }
        Ok(())
    }

    pub(super) async fn boundary(&mut self) -> BuiltinResult<()> {
        self.cooperation.completed(self.pending).await?;
        self.pending = 0;
        Ok(())
    }
}

impl SplineCooperation for NumericalWork<'_> {
    async fn checkpoint(&mut self, completed_work: u64) -> BuiltinResult<()> {
        self.boundary().await?;
        self.cooperation.completed(completed_work).await
    }
}

pub(super) async fn analytic_envelope(
    values: &[f64],
    filter_len: Option<usize>,
    work: &mut NumericalWork<'_>,
) -> BuiltinResult<(Vec<f64>, Vec<f64>)> {
    if values.len() == 1 {
        return Ok((vec![values[0]], vec![values[0]]));
    }
    let mut sum = 0.0;
    for value in values {
        sum += value;
        work.tick().await?;
    }
    let mean = sum / values.len() as f64;
    let mut centered = Vec::with_capacity(values.len());
    for value in values {
        centered.push(value - mean);
        work.tick().await?;
    }
    let magnitude = if let Some(filter_len) = filter_len {
        analytic_fir_magnitude(&centered, filter_len, work).await?
    } else {
        let analytic = analytic_signal(&centered, work).await?;
        let mut magnitude = Vec::with_capacity(analytic.len());
        for value in analytic {
            magnitude.push(value.norm());
            work.tick().await?;
        }
        magnitude
    };
    let mut upper = Vec::with_capacity(magnitude.len());
    let mut lower = Vec::with_capacity(magnitude.len());
    for value in magnitude {
        upper.push(mean + value);
        lower.push(mean - value);
        work.tick().await?;
    }
    Ok((upper, lower))
}

async fn analytic_fir_magnitude(
    values: &[f64],
    filter_len: usize,
    work: &mut NumericalWork<'_>,
) -> BuiltinResult<Vec<f64>> {
    if filter_len <= 1 {
        let mut magnitude = Vec::with_capacity(values.len());
        for value in values {
            magnitude.push(value.abs());
            work.tick().await?;
        }
        return Ok(magnitude);
    }
    let kernel = hilbert_fir_kernel(filter_len, work).await?;
    let quadrature = centered_convolution(values, &kernel, work).await?;
    let mut magnitude = Vec::with_capacity(values.len());
    for (&re, im) in values.iter().zip(quadrature) {
        magnitude.push(re.hypot(im));
        work.tick().await?;
    }
    Ok(magnitude)
}

async fn hilbert_fir_kernel(
    filter_len: usize,
    work: &mut NumericalWork<'_>,
) -> BuiltinResult<Vec<f64>> {
    let center = (filter_len as f64 - 1.0) / 2.0;
    let denominator = modified_bessel_i0(8.0);
    let mut kernel = Vec::with_capacity(filter_len);
    for idx in 0..filter_len {
        let k = idx as f64 - center;
        let ideal = if k.abs() <= EPS {
            0.0
        } else {
            let rounded = k.round();
            if (rounded - k).abs() <= EPS && (rounded as i64).rem_euclid(2) == 0 {
                0.0
            } else {
                2.0 / (std::f64::consts::PI * k)
            }
        };
        kernel.push(ideal * kaiser_window(idx, filter_len, 8.0, denominator));
        work.tick().await?;
    }
    Ok(kernel)
}

fn kaiser_window(idx: usize, len: usize, beta: f64, denominator: f64) -> f64 {
    if len <= 1 {
        return 1.0;
    }
    let ratio = 2.0 * idx as f64 / (len - 1) as f64 - 1.0;
    let argument = beta * (1.0 - ratio * ratio).max(0.0).sqrt();
    modified_bessel_i0(argument) / denominator
}

fn modified_bessel_i0(x: f64) -> f64 {
    let y = x * x / 4.0;
    let mut term = 1.0;
    let mut sum = 1.0;
    for k in 1..=32 {
        term *= y / ((k * k) as f64);
        sum += term;
        if term.abs() <= sum.abs() * 1.0e-15 {
            break;
        }
    }
    sum
}

async fn centered_convolution(
    values: &[f64],
    kernel: &[f64],
    work: &mut NumericalWork<'_>,
) -> BuiltinResult<Vec<f64>> {
    let center = kernel.len() / 2;
    let mut out = vec![0.0; values.len()];
    for (idx, slot) in out.iter_mut().enumerate() {
        let mut acc = 0.0;
        for (tap, &weight) in kernel.iter().enumerate() {
            let source = idx as isize + tap as isize - center as isize;
            if (0..values.len() as isize).contains(&source) {
                acc += values[source as usize] * weight;
            }
            work.tick().await?;
        }
        *slot = acc;
    }
    Ok(out)
}

async fn analytic_signal(
    values: &[f64],
    work: &mut NumericalWork<'_>,
) -> BuiltinResult<Vec<Complex<f64>>> {
    let n = values.len();
    work.boundary().await?;
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);
    let ifft = planner.plan_fft_inverse(n);
    work.boundary().await?;
    let mut buffer = Vec::with_capacity(n);
    for &value in values {
        buffer.push(Complex::new(value, 0.0));
        work.tick().await?;
    }
    work.boundary().await?;
    fft.process(&mut buffer);
    work.boundary().await?;
    for (freq, value) in buffer.iter_mut().enumerate() {
        *value *= analytic_multiplier(freq, n);
        work.tick().await?;
    }
    work.boundary().await?;
    ifft.process(&mut buffer);
    work.boundary().await?;
    let scale = 1.0 / n as f64;
    for value in &mut buffer {
        *value *= scale;
        work.tick().await?;
    }
    Ok(buffer)
}

fn analytic_multiplier(freq: usize, len: usize) -> f64 {
    if freq == 0 {
        return 1.0;
    }
    if len.is_multiple_of(2) {
        if freq < len / 2 {
            2.0
        } else if freq == len / 2 {
            1.0
        } else {
            0.0
        }
    } else if freq <= len / 2 {
        2.0
    } else {
        0.0
    }
}

pub(super) async fn rms_envelope(
    values: &[f64],
    window_len: usize,
    work: &mut NumericalWork<'_>,
) -> BuiltinResult<(Vec<f64>, Vec<f64>)> {
    let n = values.len();
    let half_before = (window_len - 1) / 2;
    let half_after = window_len / 2;
    let mut prefix = Vec::with_capacity(n + 1);
    prefix.push(0.0);
    for value in values {
        prefix.push(prefix.last().copied().unwrap_or(0.0) + value * value);
        work.tick().await?;
    }
    let mut upper = Vec::with_capacity(n);
    for idx in 0..n {
        let start = idx.saturating_sub(half_before);
        let end = (idx + half_after + 1).min(n);
        let count = (end - start).max(1) as f64;
        upper.push(((prefix[end] - prefix[start]) / count).sqrt());
        work.tick().await?;
    }
    let mut lower = Vec::with_capacity(upper.len());
    for value in &upper {
        lower.push(-*value);
        work.tick().await?;
    }
    Ok((upper, lower))
}

pub(super) async fn peak_envelope(
    values: &[f64],
    separation: usize,
    work: &mut NumericalWork<'_>,
) -> BuiltinResult<(Vec<f64>, Vec<f64>)> {
    if values.len() == 1 {
        return Ok((vec![values[0]], vec![values[0]]));
    }
    let upper_points = extrema_points(values, separation, true, work).await?;
    let lower_points = extrema_points(values, separation, false, work).await?;
    let mut query = Vec::with_capacity(values.len());
    for idx in 0..values.len() {
        query.push(idx as f64);
        work.tick().await?;
    }
    work.boundary().await?;
    let upper = spline_interpolate(&upper_points, &query, work).await?;
    work.boundary().await?;
    let lower = spline_interpolate(&lower_points, &query, work).await?;
    work.boundary().await?;
    Ok((upper, lower))
}

async fn extrema_points(
    values: &[f64],
    separation: usize,
    maxima: bool,
    work: &mut NumericalWork<'_>,
) -> BuiltinResult<Vec<(usize, f64)>> {
    let mut candidates = Vec::new();
    for idx in 1..values.len() - 1 {
        let prev = values[idx - 1];
        let curr = values[idx];
        let next = values[idx + 1];
        let is_extremum = if maxima {
            (curr >= prev && curr > next) || (curr > prev && curr >= next)
        } else {
            (curr <= prev && curr < next) || (curr < prev && curr <= next)
        };
        if is_extremum {
            candidates.push((idx, curr));
        }
        work.tick().await?;
    }
    work.boundary().await?;
    candidates.sort_by(|a, b| {
        let lhs = if maxima { a.1 } else { -a.1 };
        let rhs = if maxima { b.1 } else { -b.1 };
        rhs.partial_cmp(&lhs).unwrap_or(std::cmp::Ordering::Equal)
    });
    work.boundary().await?;

    let mut selected: Vec<(usize, f64)> = Vec::new();
    for candidate in candidates {
        let mut separated = true;
        for (idx, _) in &selected {
            if idx.abs_diff(candidate.0) < separation {
                separated = false;
                break;
            }
            work.tick().await?;
        }
        if separated {
            selected.push(candidate);
        }
        work.tick().await?;
    }
    selected.push((0, values[0]));
    selected.push((values.len() - 1, values[values.len() - 1]));
    selected.sort_by_key(|(idx, _)| *idx);
    selected.dedup_by_key(|(idx, _)| *idx);
    work.boundary().await?;
    if selected.len() == 1 {
        selected.push((values.len() - 1, values[values.len() - 1]));
    }
    Ok(selected)
}

async fn spline_interpolate(
    points: &[(usize, f64)],
    query: &[f64],
    work: &mut NumericalWork<'_>,
) -> BuiltinResult<Vec<f64>> {
    if points.len() < 2 {
        return Err(envelope_error(&ENVELOPE_ERROR_INTERNAL));
    }
    let series = NumericSeries {
        x: points
            .iter()
            .map(|(idx, _)| *idx as f64)
            .collect::<Vec<_>>(),
        y: points.iter().map(|(_, value)| *value).collect::<Vec<_>>(),
        series: 1,
        component_shape: Vec::new(),
        sample_axis: SampleAxis::First,
    };
    let query = QueryPoints {
        values: query.to_vec(),
        shape: vec![1, query.len()],
    };
    let pp = build_spline_pp(&series, BUILTIN_NAME, work)
        .await
        .map_err(|err| {
            if is_spline_execution_failure(&err) {
                crate::builtins::common::map_control_flow_with_builtin(err, BUILTIN_NAME)
            } else {
                envelope_error_with_detail(&ENVELOPE_ERROR_INTERNAL, err.message())
            }
        })?;
    let value = evaluate_pp(
        &pp,
        &query,
        &Extrapolation::Extrapolate,
        PpOutputLayout::QueryLeading,
        BUILTIN_NAME,
    )
    .map_err(|err| envelope_error_with_detail(&ENVELOPE_ERROR_INTERNAL, err.message()))?;
    match value {
        Value::Tensor(tensor) => Ok(tensor::tensor_into_values_f64(tensor)),
        Value::Num(value) => Ok(vec![value]),
        _ => Err(envelope_error(&ENVELOPE_ERROR_INTERNAL)),
    }
}
