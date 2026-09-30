//! MATLAB-compatible `histcounts` builtin with GPU-aware semantics for RunMat.

use std::cmp::Ordering;

use runmat_accelerate_api::{GpuTensorHandle, IntegerElementType};
use runmat_builtins::{
    BuiltinErrorDescriptor, HISTCOUNTS_ERROR_BINMETHOD_CONFLICT, HISTCOUNTS_ERROR_BINWIDTH_INVALID,
};
use runmat_macros::runtime_builtin;
use runmat_value::{Tensor, Value};

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::tensor;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "histcounts";
const DEFAULT_BIN_COUNT: usize = 10;
const RANGE_EPS: f64 = 1.0e-12;
fn builtin_error_with(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn descriptor_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    builtin_error_with(error, error.message)
}

fn builtin_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

#[runtime_builtin(
    name = "histcounts",
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::hist::histcounts"
)]
async fn histcounts_builtin(data: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let eval = evaluate(data, &rest).await?;
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            return Ok(Value::OutputList(Vec::new()));
        }
        if out_count == 1 {
            return Ok(Value::OutputList(vec![eval.into_counts_value()]));
        }
        let (counts, edges) = eval.into_pair();
        return Ok(crate::output_count::output_list_with_padding(
            out_count,
            vec![counts, edges],
        ));
    }
    Ok(eval.into_counts_value())
}

/// Evaluate `histcounts` once and surface both primary outputs.
pub async fn evaluate(data: Value, rest: &[Value]) -> BuiltinResult<HistcountsEvaluation> {
    let options = parse_options(rest)?;
    match data {
        Value::GpuTensor(handle) => histcounts_gpu(handle, &options).await,
        other => histcounts_host(other, &options),
    }
}

async fn histcounts_gpu(
    handle: GpuTensorHandle,
    options: &HistcountsOptions,
) -> BuiltinResult<HistcountsEvaluation> {
    if matches!(
        runmat_accelerate_api::handle_integer_type(&handle),
        Some(IntegerElementType::I64 | IntegerElementType::U64)
    ) {
        return Err(build_runtime_error(
            "histcounts: 64-bit integer gpuArray inputs are not supported",
        )
        .with_builtin(BUILTIN_NAME)
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build());
    }
    let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
    histcounts_from_tensor(tensor, options)
}

fn histcounts_host(
    value: Value,
    options: &HistcountsOptions,
) -> BuiltinResult<HistcountsEvaluation> {
    let tensor = tensor::value_into_tensor_for(BUILTIN_NAME, value).map_err(builtin_error)?;
    histcounts_from_tensor(tensor, options)
}

fn histcounts_from_tensor(
    tensor: Tensor,
    options: &HistcountsOptions,
) -> BuiltinResult<HistcountsEvaluation> {
    let mut values = Vec::new();
    let mut min_val: Option<f64> = None;
    let mut max_val: Option<f64> = None;
    // Numeric histogram edges and normalization are defined in a floating
    // computation domain. Keep this conversion explicit: automatic binning of
    // wide integer inputs intentionally has double-precision granularity.
    let samples = tensor::tensor_values_f64_cow(&tensor);

    for &sample in samples.iter() {
        if sample.is_nan() {
            continue;
        }
        values.push(sample);
        if sample.is_finite() {
            min_val = Some(match min_val {
                Some(current) if sample >= current => current,
                Some(_) => sample,
                None => sample,
            });
            max_val = Some(match max_val {
                Some(current) if sample <= current => current,
                Some(_) => sample,
                None => sample,
            });
        }
    }

    let original_range_zero = match (min_val, max_val) {
        (Some(minimum), Some(maximum)) => minimum == maximum,
        _ => false,
    };

    let edges = compute_edges(&values, min_val, max_val, original_range_zero, options)?;
    let mut counts = vec![0.0f64; edges.len() - 1];

    for value in values {
        if value < edges[0] || value > edges[edges.len() - 1] {
            continue;
        }
        if value == edges[edges.len() - 1] {
            if let Some(last) = counts.last_mut() {
                *last += 1.0;
            }
            continue;
        }

        match edges.binary_search_by(|edge| match edge.partial_cmp(&value) {
            Some(order) => order,
            None => Ordering::Less,
        }) {
            Ok(index) => {
                if index == 0 {
                    counts[0] += 1.0;
                } else if index < counts.len() {
                    counts[index] += 1.0;
                } else if let Some(last) = counts.last_mut() {
                    *last += 1.0;
                }
            }
            Err(index) => {
                if index == 0 || index > counts.len() {
                    continue;
                }
                counts[index - 1] += 1.0;
            }
        }
    }

    let normalised = apply_normalization(&counts, &edges, options.normalization);
    let counts_tensor = Tensor::new(normalised, vec![1, counts.len()])
        .map_err(|e| builtin_error(format!("histcounts: {e}")))?;
    let edges_tensor = Tensor::new(edges.clone(), vec![1, edges.len()])
        .map_err(|e| builtin_error(format!("histcounts: {e}")))?;

    Ok(HistcountsEvaluation::new(counts_tensor, edges_tensor))
}

fn compute_edges(
    values: &[f64],
    min_val: Option<f64>,
    max_val: Option<f64>,
    original_range_zero: bool,
    options: &HistcountsOptions,
) -> BuiltinResult<Vec<f64>> {
    if let Some(edges) = &options.explicit_edges {
        validate_edges(edges)?;
        return Ok(edges.clone());
    }

    if let Some(method) = options.bin_method {
        return compute_edges_with_method(
            values,
            min_val,
            max_val,
            original_range_zero,
            method,
            options,
        );
    }

    compute_edges_standard(min_val, max_val, original_range_zero, options)
}

fn compute_edges_standard(
    min_val: Option<f64>,
    max_val: Option<f64>,
    original_range_zero: bool,
    options: &HistcountsOptions,
) -> BuiltinResult<Vec<f64>> {
    let (mut lower, mut upper) = derive_initial_limits(min_val, max_val, options.bin_limits);

    if !lower.is_finite() || !upper.is_finite() {
        return Err(builtin_error(
            "histcounts: data range must be finite; specify BinLimits or BinEdges",
        ));
    }

    if upper < lower {
        return Err(builtin_error("histcounts: bin limits must be increasing"));
    }

    if options.bin_limits.is_some() && lower == upper {
        return Err(builtin_error(
            "histcounts: BinLimits must specify a non-zero width",
        ));
    }

    if let Some(width) = options.bin_width {
        if !width.is_finite() || width <= 0.0 {
            return Err(builtin_error(HISTCOUNTS_ERROR_BINWIDTH_INVALID.message));
        }

        if original_range_zero && options.bin_limits.is_none() {
            let centre = min_val.unwrap_or(0.0);
            lower = centre - width / 2.0;
            upper = centre + width / 2.0;
        } else if options.bin_limits.is_none() {
            if let (Some(minimum), Some(maximum)) = (min_val, max_val) {
                lower = minimum;
                upper = maximum;
            }
        }

        if lower == upper {
            upper = lower + width;
        }

        let bins = ((upper - lower) / width).ceil().max(1.0) as usize;
        let mut edges = Vec::with_capacity(bins + 1);
        for i in 0..=bins {
            edges.push(lower + width * i as f64);
        }
        if let Some(last) = edges.last_mut() {
            *last = upper;
        }
        return finalize_generated_edges(edges);
    }

    let mut num_bins = options.num_bins.unwrap_or(DEFAULT_BIN_COUNT);
    if num_bins == 0 {
        return Err(builtin_error(
            "histcounts: NumBins must be a positive integer",
        ));
    }

    if original_range_zero {
        let centre = min_val.unwrap_or(0.0);
        if options.num_bins.is_none() && options.bin_limits.is_none() && options.bin_width.is_none()
        {
            num_bins = 1;
        }
        if options.bin_limits.is_none() && options.bin_width.is_none() {
            lower = centre - 0.5;
            upper = centre + 0.5;
        }
    } else if options.bin_limits.is_none() {
        if let (Some(minimum), Some(maximum)) = (min_val, max_val) {
            lower = lower.min(minimum);
            upper = upper.max(maximum);
        }
    }

    if lower == upper {
        upper = lower + 1.0;
    }

    finalize_generated_edges(linspace(lower, upper, num_bins + 1))
}

fn compute_edges_with_method(
    values: &[f64],
    min_val: Option<f64>,
    max_val: Option<f64>,
    original_range_zero: bool,
    method: BinMethod,
    options: &HistcountsOptions,
) -> BuiltinResult<Vec<f64>> {
    if values.is_empty() {
        return compute_edges_standard(min_val, max_val, original_range_zero, options);
    }

    if matches!(method, BinMethod::Integers) {
        return finalize_generated_edges(compute_integer_edges(min_val, max_val, options)?);
    }

    let (lower, upper) = derive_initial_limits(min_val, max_val, options.bin_limits);
    if !lower.is_finite() || !upper.is_finite() {
        return Err(builtin_error(
            "histcounts: data range must be finite for BinMethod",
        ));
    }

    if lower == upper {
        if options.bin_limits.is_some() {
            return Err(builtin_error(
                "histcounts: BinLimits must specify a non-zero width",
            ));
        }
        return compute_edges_standard(min_val, max_val, true, options);
    }

    let finite_values: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if finite_values.is_empty() {
        return compute_edges_standard(min_val, max_val, original_range_zero, options);
    }

    let range = upper - lower;
    if range <= 0.0 {
        return compute_edges_standard(min_val, max_val, true, options);
    }

    let mut sorted = finite_values.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    let width = match method {
        BinMethod::Auto => {
            let fd = freedman_diaconis_width(&sorted);
            if fd > 0.0 {
                fd
            } else {
                let scott = scott_width(&finite_values);
                if scott > 0.0 {
                    scott
                } else {
                    range / sturges_bin_count(sorted.len()) as f64
                }
            }
        }
        BinMethod::Scott => scott_width(&finite_values),
        BinMethod::Fd => freedman_diaconis_width(&sorted),
        BinMethod::Sturges => range / sturges_bin_count(sorted.len()) as f64,
        BinMethod::Sqrt => range / sqrt_bin_count(sorted.len()) as f64,
        BinMethod::Integers => unreachable!(),
    };

    if !width.is_finite() || width <= 0.0 {
        return compute_edges_standard(min_val, max_val, original_range_zero, options);
    }

    let bins = ((range / width).ceil().max(1.0)) as usize;
    let mut edges = Vec::with_capacity(bins + 1);
    for i in 0..=bins {
        edges.push(lower + width * i as f64);
    }
    if let Some(last) = edges.last_mut() {
        *last = upper;
    }
    finalize_generated_edges(edges)
}

fn compute_integer_edges(
    min_val: Option<f64>,
    max_val: Option<f64>,
    options: &HistcountsOptions,
) -> BuiltinResult<Vec<f64>> {
    let (mut lower, mut upper) = derive_initial_limits(min_val, max_val, options.bin_limits);

    if !lower.is_finite() || !upper.is_finite() {
        return Err(builtin_error(
            "histcounts: BinLimits must be finite for 'integers' BinMethod",
        ));
    }

    if lower == upper {
        let centre = min_val.or(max_val).unwrap_or(lower);
        lower = centre - 0.5;
        upper = centre + 0.5;
    }

    if let Some((lo, hi)) = options.bin_limits {
        lower = lo;
        upper = hi;
    } else {
        lower = lower.floor();
        upper = upper.ceil();
    }

    if upper <= lower {
        upper = lower + 1.0;
    }

    let mut edges = Vec::new();
    edges.push(lower);

    if let Some((lo, hi)) = options.bin_limits {
        let mut k = lo.ceil() as i64;
        while (k as f64) < hi {
            let candidate = k as f64;
            if candidate > lo + RANGE_EPS {
                edges.push(candidate);
            }
            k += 1;
        }
    } else {
        let mut current = lower.floor() as i64 + 1;
        let end = upper.ceil() as i64;
        while current < end {
            edges.push(current as f64);
            current += 1;
        }
    }

    if *edges.last().unwrap() != upper {
        edges.push(upper);
    }

    if edges.len() < 2 {
        edges.push(upper + 1.0);
    }

    Ok(edges)
}

fn derive_initial_limits(
    min_val: Option<f64>,
    max_val: Option<f64>,
    bin_limits: Option<(f64, f64)>,
) -> (f64, f64) {
    if let Some((lo, hi)) = bin_limits {
        (lo, hi)
    } else if let (Some(minimum), Some(maximum)) = (min_val, max_val) {
        (minimum, maximum)
    } else {
        (0.0, 1.0)
    }
}

fn scott_width(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 2 {
        return 0.0;
    }
    let sigma = standard_deviation(values);
    if sigma <= 0.0 {
        return 0.0;
    }
    3.5 * sigma / (n as f64).powf(1.0 / 3.0)
}

fn freedman_diaconis_width(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n < 2 {
        return 0.0;
    }
    let q1 = percentile(sorted, 0.25);
    let q3 = percentile(sorted, 0.75);
    let iqr = q3 - q1;
    if iqr <= 0.0 {
        return 0.0;
    }
    2.0 * iqr / (n as f64).powf(1.0 / 3.0)
}

fn sturges_bin_count(n: usize) -> usize {
    if n <= 1 {
        1
    } else {
        ((n as f64).log2() + 1.0).ceil().max(1.0) as usize
    }
}

fn sqrt_bin_count(n: usize) -> usize {
    if n == 0 {
        1
    } else {
        (n as f64).sqrt().ceil().max(1.0) as usize
    }
}

fn standard_deviation(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 2 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / n as f64;
    let mut acc = 0.0;
    for &value in values {
        let diff = value - mean;
        acc += diff * diff;
    }
    (acc / (n as f64 - 1.0).max(1.0)).sqrt()
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return sorted[0];
    }
    let clamped = p.clamp(0.0, 1.0);
    let rank = clamped * (n as f64 - 1.0);
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        let weight = rank - lower as f64;
        sorted[lower] * (1.0 - weight) + sorted[upper] * weight
    }
}

fn apply_normalization(counts: &[f64], edges: &[f64], mode: HistogramNormalization) -> Vec<f64> {
    let widths: Vec<f64> = edges.windows(2).map(|pair| pair[1] - pair[0]).collect();
    let total: f64 = counts.iter().sum();

    match mode {
        HistogramNormalization::Count => counts.to_vec(),
        HistogramNormalization::Probability => {
            if total > 0.0 {
                counts.iter().map(|&c| c / total).collect()
            } else {
                vec![0.0; counts.len()]
            }
        }
        HistogramNormalization::CountDensity => counts
            .iter()
            .zip(widths.iter())
            .map(|(&c, &w)| if w > 0.0 { c / w } else { 0.0 })
            .collect(),
        HistogramNormalization::Pdf => {
            if total > 0.0 {
                counts
                    .iter()
                    .zip(widths.iter())
                    .map(|(&c, &w)| if w > 0.0 { c / (total * w) } else { 0.0 })
                    .collect()
            } else {
                vec![0.0; counts.len()]
            }
        }
        HistogramNormalization::CumCount => {
            let mut acc = 0.0;
            counts
                .iter()
                .map(|&c| {
                    acc += c;
                    acc
                })
                .collect()
        }
        HistogramNormalization::Cdf => {
            if total > 0.0 {
                let mut acc = 0.0;
                counts
                    .iter()
                    .map(|&c| {
                        acc += c;
                        acc / total
                    })
                    .collect()
            } else {
                vec![0.0; counts.len()]
            }
        }
    }
}

fn validate_edges(edges: &[f64]) -> BuiltinResult<()> {
    if edges.len() < 2 {
        return Err(builtin_error(
            "histcounts: bin edges must contain at least two elements",
        ));
    }
    for pair in edges.windows(2) {
        if pair[0].is_nan() || pair[1].is_nan() {
            return Err(builtin_error(
                "histcounts: bin edges must be finite numbers",
            ));
        }
        if pair[1] <= pair[0] {
            return Err(builtin_error(
                "histcounts: bin edges must be strictly increasing",
            ));
        }
    }
    Ok(())
}

fn linspace(start: f64, stop: f64, count: usize) -> Vec<f64> {
    if count <= 1 {
        return vec![start];
    }
    let step = (stop - start) / (count - 1) as f64;
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        out.push(start + step * i as f64);
    }
    if let Some(last) = out.last_mut() {
        *last = stop;
    }
    out
}

fn finalize_generated_edges(edges: Vec<f64>) -> BuiltinResult<Vec<f64>> {
    let mut compact = Vec::with_capacity(edges.len());
    for edge in edges {
        if !edge.is_finite() {
            return Err(builtin_error(
                "histcounts: generated bin edges must be finite numbers",
            ));
        }
        if match compact.last() {
            Some(previous) => edge > *previous,
            None => true,
        } {
            compact.push(edge);
        }
    }

    if compact.len() == 1 {
        let centre = compact[0];
        let lower = next_down(centre);
        if lower.is_finite() {
            compact.insert(0, lower);
        } else {
            let upper = next_up(centre);
            if upper.is_finite() {
                compact.push(upper);
            }
        }
    }

    validate_edges(&compact)?;
    Ok(compact)
}

fn next_up(value: f64) -> f64 {
    if value.is_nan() || value == f64::INFINITY {
        return value;
    }
    if value == 0.0 {
        return f64::from_bits(1);
    }
    let bits = value.to_bits();
    f64::from_bits(if value > 0.0 { bits + 1 } else { bits - 1 })
}

fn next_down(value: f64) -> f64 {
    if value.is_nan() || value == f64::NEG_INFINITY {
        return value;
    }
    if value == 0.0 {
        return -f64::from_bits(1);
    }
    let bits = value.to_bits();
    f64::from_bits(if value > 0.0 { bits - 1 } else { bits + 1 })
}

#[derive(Clone, Default)]
struct HistcountsOptions {
    explicit_edges: Option<Vec<f64>>,
    num_bins: Option<usize>,
    bin_width: Option<f64>,
    bin_limits: Option<(f64, f64)>,
    bin_method: Option<BinMethod>,
    normalization: HistogramNormalization,
}

impl HistcountsOptions {
    fn validate(&self) -> BuiltinResult<()> {
        if self.explicit_edges.is_some()
            && (self.num_bins.is_some() || self.bin_width.is_some() || self.bin_limits.is_some())
        {
            return Err(builtin_error(
                "histcounts: BinEdges cannot be combined with NumBins, BinWidth, or BinLimits",
            ));
        }
        if self.bin_method.is_some()
            && (self.explicit_edges.is_some()
                || self.bin_width.is_some()
                || self.num_bins.is_some())
        {
            return Err(descriptor_error(&HISTCOUNTS_ERROR_BINMETHOD_CONFLICT));
        }
        if self.num_bins.is_some() && self.bin_width.is_some() {
            return Err(builtin_error(
                "histcounts: specify only one of NumBins or BinWidth",
            ));
        }
        if let Some((lo, hi)) = self.bin_limits {
            if !lo.is_finite() || !hi.is_finite() {
                return Err(builtin_error("histcounts: BinLimits must be finite"));
            }
            if hi < lo {
                return Err(builtin_error("histcounts: BinLimits must be increasing"));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BinMethod {
    Auto,
    Scott,
    Fd,
    Sturges,
    Sqrt,
    Integers,
}

#[derive(Clone, Copy, Debug, Default)]
enum HistogramNormalization {
    #[default]
    Count,
    Probability,
    CountDensity,
    Pdf,
    CumCount,
    Cdf,
}

#[derive(Clone)]
pub struct HistcountsEvaluation {
    counts: Tensor,
    edges: Tensor,
}

impl HistcountsEvaluation {
    fn new(counts: Tensor, edges: Tensor) -> Self {
        Self { counts, edges }
    }

    pub fn into_counts_value(self) -> Value {
        Value::Tensor(self.counts)
    }

    pub fn into_pair(self) -> (Value, Value) {
        let counts = self.counts;
        let edges = self.edges;
        (Value::Tensor(counts), Value::Tensor(edges))
    }
}

fn parse_options(args: &[Value]) -> BuiltinResult<HistcountsOptions> {
    let mut options = HistcountsOptions::default();
    let mut index = 0;

    if index < args.len() && !is_option_key(&args[index]) {
        match classify_bin_argument(&args[index])? {
            BinArgument::NumBins(n) => {
                options.num_bins = Some(n);
            }
            BinArgument::Edges(edges) => {
                validate_edges(&edges)?;
                options.explicit_edges = Some(edges);
            }
        }
        index += 1;
    }

    while index < args.len() {
        let key = tensor::value_to_string(&args[index])
            .ok_or_else(|| builtin_error("histcounts: expected name/value pair arguments"))?;
        index += 1;
        if index >= args.len() {
            return Err(builtin_error(format!(
                "histcounts: missing value for option '{key}'"
            )));
        }
        let lowered = key.trim().to_ascii_lowercase();
        let value = &args[index];
        index += 1;

        match lowered.as_str() {
            "binedges" => {
                let edges = numeric_vector(value, "histcounts", "BinEdges")?;
                validate_edges(&edges)?;
                options.explicit_edges = Some(edges);
            }
            "numbins" => {
                let scalar = positive_usize(value, "histcounts", "NumBins")?;
                options.num_bins = Some(scalar);
            }
            "binwidth" => {
                let width = positive_scalar(value, "histcounts", "BinWidth")?;
                options.bin_width = Some(width);
            }
            "binlimits" => {
                let limits = numeric_vector(value, "histcounts", "BinLimits")?;
                if limits.len() != 2 {
                    return Err(builtin_error(
                        "histcounts: BinLimits must contain exactly two elements",
                    ));
                }
                let lo = limits[0];
                let hi = limits[1];
                if hi < lo {
                    return Err(builtin_error("histcounts: BinLimits must be increasing"));
                }
                if !lo.is_finite() || !hi.is_finite() {
                    return Err(builtin_error("histcounts: BinLimits must be finite"));
                }
                options.bin_limits = Some((lo, hi));
            }
            "normalization" => {
                let text = tensor::value_to_string(value)
                    .ok_or_else(|| builtin_error("histcounts: Normalization must be a string"))?;
                options.normalization = parse_normalization(&text)?;
            }
            "binmethod" => {
                let text = tensor::value_to_string(value)
                    .ok_or_else(|| builtin_error("histcounts: BinMethod must be a string"))?;
                options.bin_method = Some(parse_bin_method(&text)?);
            }
            other => {
                return Err(builtin_error(format!(
                    "histcounts: unrecognised option '{other}'"
                )));
            }
        }
    }

    options.validate()?;
    Ok(options)
}

#[derive(Debug)]
enum BinArgument {
    NumBins(usize),
    Edges(Vec<f64>),
}

fn classify_bin_argument(value: &Value) -> BuiltinResult<BinArgument> {
    match value {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => {
            let n = positive_usize(value, "histcounts", "NumBins")?;
            Ok(BinArgument::NumBins(n))
        }
        Value::Tensor(tensor) => {
            if tensor::is_scalar_tensor(tensor) {
                Ok(BinArgument::NumBins(positive_usize(
                    value,
                    "histcounts",
                    "NumBins",
                )?))
            } else {
                let edges = tensor::tensor_values_f64(tensor);
                Ok(BinArgument::Edges(edges))
            }
        }
        Value::LogicalArray(logical) => {
            if logical.data.len() == 1 {
                Ok(BinArgument::NumBins(positive_usize_from_f64(
                    f64::from(logical.data[0]),
                    "histcounts",
                    "NumBins",
                )?))
            } else {
                Ok(BinArgument::Edges(
                    logical.data.iter().copied().map(f64::from).collect(),
                ))
            }
        }
        Value::GpuTensor(_) => Err(build_runtime_error(
            "histcounts: bin specification cannot be a gpuArray",
        )
        .with_builtin(BUILTIN_NAME)
        .with_gpu_gather_retry(crate::GpuGatherRetry::Requested)
        .build()),
        other => Err(builtin_error(format!(
            "histcounts: unsupported bin specification {:?}",
            other
        ))),
    }
}

fn is_option_key(value: &Value) -> bool {
    if let Some(text) = tensor::value_to_string(value) {
        let lowered = text.trim().to_ascii_lowercase();
        matches!(
            lowered.as_str(),
            "binedges" | "numbins" | "binwidth" | "binlimits" | "normalization" | "binmethod"
        )
    } else {
        false
    }
}

fn numeric_vector(value: &Value, name: &str, option: &str) -> BuiltinResult<Vec<f64>> {
    let tensor = tensor::value_to_tensor(value)
        .map_err(|_| builtin_error(format!("{name}: {option} must be numeric")))?;
    Ok(tensor::tensor_into_values_f64(tensor))
}

fn positive_usize(value: &Value, name: &str, option: &str) -> BuiltinResult<usize> {
    if let Some(value) = tensor::scalar_integer_value(value) {
        return value
            .try_to_usize()
            .filter(|value| *value > 0 && *value < usize::MAX)
            .ok_or_else(|| {
                builtin_error(format!("{name}: {option} must be a positive finite scalar"))
            });
    }
    let scalar = scalar_value(value, name, option)?;
    positive_usize_from_f64(scalar, name, option)
}

fn positive_usize_from_f64(scalar: f64, name: &str, option: &str) -> BuiltinResult<usize> {
    if scalar <= 0.0 || !scalar.is_finite() {
        return Err(builtin_error(format!(
            "{name}: {option} must be a positive finite scalar"
        )));
    }
    let rounded = scalar.round();
    if (scalar - rounded).abs() > f64::EPSILON {
        return Err(builtin_error(format!(
            "{name}: {option} must be an integer"
        )));
    }
    if rounded > usize::MAX as f64 || (usize::BITS == 64 && rounded == usize::MAX as f64) {
        return Err(builtin_error(format!(
            "{name}: {option} is outside the supported range"
        )));
    }
    Ok(rounded as usize)
}

fn positive_scalar(value: &Value, name: &str, option: &str) -> BuiltinResult<f64> {
    let scalar = scalar_value(value, name, option)?;
    if !scalar.is_finite() || scalar <= 0.0 {
        if name == BUILTIN_NAME && option == "BinWidth" {
            return Err(descriptor_error(&HISTCOUNTS_ERROR_BINWIDTH_INVALID));
        }
        return Err(builtin_error(format!(
            "{name}: {option} must be a positive finite scalar"
        )));
    }
    Ok(scalar)
}

fn scalar_value(value: &Value, name: &str, option: &str) -> BuiltinResult<f64> {
    match value {
        Value::Num(n) => Ok(*n),
        Value::Int(i) => Ok(i.to_f64()),
        Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        Value::Tensor(tensor) => {
            if !tensor::is_scalar_tensor(tensor) {
                return Err(builtin_error(format!("{name}: {option} must be a scalar")));
            }
            Ok(tensor::tensor_value_f64(tensor, 0))
        }
        Value::LogicalArray(logical) => {
            if logical.data.len() != 1 {
                return Err(builtin_error(format!("{name}: {option} must be a scalar")));
            }
            Ok(if logical.data[0] != 0 { 1.0 } else { 0.0 })
        }
        other => Err(builtin_error(format!(
            "{name}: {option} must be numeric, got {:?}",
            other
        ))),
    }
}

fn parse_bin_method(text: &str) -> BuiltinResult<BinMethod> {
    match text.trim().to_ascii_lowercase().as_str() {
        "auto" => Ok(BinMethod::Auto),
        "scott" => Ok(BinMethod::Scott),
        "fd" | "freedmandiaconis" => Ok(BinMethod::Fd),
        "sturges" => Ok(BinMethod::Sturges),
        "sqrt" => Ok(BinMethod::Sqrt),
        "integers" => Ok(BinMethod::Integers),
        other => Err(builtin_error(format!(
            "histcounts: unrecognised BinMethod value '{other}'"
        ))),
    }
}

fn parse_normalization(text: &str) -> BuiltinResult<HistogramNormalization> {
    match text.trim().to_ascii_lowercase().as_str() {
        "count" => Ok(HistogramNormalization::Count),
        "probability" => Ok(HistogramNormalization::Probability),
        "countdensity" => Ok(HistogramNormalization::CountDensity),
        "pdf" | "probabilitydensity" => Ok(HistogramNormalization::Pdf),
        "cumcount" => Ok(HistogramNormalization::CumCount),
        "cdf" => Ok(HistogramNormalization::Cdf),
        other => Err(builtin_error(format!(
            "histcounts: unrecognised Normalization value '{other}'"
        ))),
    }
}

#[cfg(test)]
mod tests;
