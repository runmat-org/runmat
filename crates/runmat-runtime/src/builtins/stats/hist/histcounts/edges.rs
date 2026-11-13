use super::super::integer_bins::{
    distinct_generated_double_edges, reserve_double_edges, unit_integer_edges, GeneratedEdgeError,
    UnitEdgeError,
};
use super::options::{BinMethod, HistogramNormalization};
use super::*;
use runmat_builtins::{HISTCOUNTS_ERROR_BINWIDTH_INVALID, HISTCOUNTS_ERROR_EDGE_PRECISION};
pub(super) fn compute_edges(
    values: &[f64],
    min_val: Option<f64>,
    max_val: Option<f64>,
    original_range_zero: bool,
    options: &HistcountsOptions,
) -> BuiltinResult<Vec<f64>> {
    if let Some(edges) = &options.explicit_edges {
        super::validate_bin_edges(edges, options.exact_edges.as_deref())?;
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
        let capacity = bins.checked_add(1).ok_or_else(edge_resource_error)?;
        let mut edges = reserve_double_edges(capacity).ok_or_else(edge_resource_error)?;
        for i in 0..=bins {
            edges.push(lower + width * i as f64);
        }
        if let Some(last) = edges.last_mut() {
            *last = upper;
        }
        return finalize_generated_edges(edges, true);
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

    let capacity = num_bins.checked_add(1).ok_or_else(edge_resource_error)?;
    finalize_generated_edges(
        linspace(lower, upper, capacity)?,
        options.num_bins.is_some(),
    )
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
        return finalize_generated_edges(compute_integer_edges(min_val, max_val, options)?, true);
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
    let capacity = bins.checked_add(1).ok_or_else(edge_resource_error)?;
    let mut edges = reserve_double_edges(capacity).ok_or_else(edge_resource_error)?;
    for i in 0..=bins {
        edges.push(lower + width * i as f64);
    }
    if let Some(last) = edges.last_mut() {
        *last = upper;
    }
    finalize_generated_edges(edges, false)
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

    unit_integer_edges(lower, upper, options.bin_limits.is_some()).map_err(|error| match error {
        UnitEdgeError::UnrepresentableBoundary => {
            descriptor_error(&HISTCOUNTS_ERROR_EDGE_PRECISION)
        }
        UnitEdgeError::TooManyEdges => {
            builtin_error("histcounts: integer-aligned bin count exceeds available resources")
        }
    })
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

pub(super) fn apply_normalization(
    counts: &[f64],
    edges: &[f64],
    exact_edges: Option<&[IntValue]>,
    mode: HistogramNormalization,
) -> Vec<f64> {
    let widths = super::super::integer_bins::edge_widths(edges, exact_edges);
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

pub(super) fn validate_edges(edges: &[f64]) -> BuiltinResult<()> {
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

fn edge_resource_error() -> RuntimeError {
    builtin_error("histcounts: generated bin count exceeds available resources")
}

fn linspace(start: f64, stop: f64, count: usize) -> BuiltinResult<Vec<f64>> {
    if count <= 1 {
        return Ok(vec![start]);
    }
    let step = (stop - start) / (count - 1) as f64;
    let mut out = reserve_double_edges(count).ok_or_else(edge_resource_error)?;
    for i in 0..count {
        out.push(start + step * i as f64);
    }
    if let Some(last) = out.last_mut() {
        *last = stop;
    }
    Ok(out)
}

fn finalize_generated_edges(
    edges: Vec<f64>,
    exact_count_required: bool,
) -> BuiltinResult<Vec<f64>> {
    let original_count = edges.len();
    let compact = distinct_generated_double_edges(edges).map_err(|error| match error {
        GeneratedEdgeError::Nonfinite => {
            builtin_error("histcounts: generated bin edges must be finite numbers")
        }
        GeneratedEdgeError::Unrepresentable => descriptor_error(&HISTCOUNTS_ERROR_EDGE_PRECISION),
    })?;
    if exact_count_required && compact.len() != original_count {
        return Err(descriptor_error(&HISTCOUNTS_ERROR_EDGE_PRECISION));
    }
    validate_edges(&compact)?;
    Ok(compact)
}
