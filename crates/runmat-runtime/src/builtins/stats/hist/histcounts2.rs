//! MATLAB-compatible `histcounts2` builtin with GPU-aware semantics for RunMat.

use std::cmp::Ordering;

use runmat_builtins::{
    BuiltinErrorDescriptor, HISTCOUNTS2_ERROR_BINMETHOD_CONFLICT,
    HISTCOUNTS2_ERROR_NUMBINS_INVALID, HISTCOUNTS2_EXTRA_SYNTAX_EXTENSION,
    HISTCOUNTS2_GPU_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{IntValue, Tensor, Value};

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::tensor;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "histcounts2";
const DEFAULT_BIN_COUNT: usize = 10;
const RANGE_EPS: f64 = 1.0e-12;
fn builtin_error_with(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn descriptor_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    builtin_error_with(error, error.message)
}

fn builtin_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(NAME).build()
}

#[runtime_builtin(
    name = "histcounts2",
    binding_variant = "default",
    builtin_path = "crate::builtins::stats::hist::histcounts2"
)]
async fn histcounts2_builtin(x: Value, y: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    parse_options(&rest)?;
    ensure_histcounts2_public_syntax(&rest)?;
    let eval = evaluate(x, y, &rest).await?;
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            return Ok(Value::OutputList(Vec::new()));
        }
        if out_count == 1 {
            return Ok(Value::OutputList(vec![eval.into_counts_value()]));
        }
        if out_count == 2 {
            let (counts, x_edges) = eval.into_pair();
            return Ok(Value::OutputList(vec![counts, x_edges]));
        }
        if out_count > 5 {
            return Err(builtin_error("histcounts2: too many output arguments"));
        }
        let (counts, x_edges, y_edges, x_bins, y_bins) = eval.into_all();
        return Ok(crate::output_count::output_list_with_padding(
            out_count,
            vec![counts, x_edges, y_edges, x_bins, y_bins],
        ));
    }
    Ok(eval.into_counts_value())
}

pub(crate) fn ensure_histcounts2_public_syntax(args: &[Value]) -> BuiltinResult<()> {
    let mut extension = false;
    if args.len() >= 2 && !is_option_key(&args[0]) && !is_option_key(&args[1]) {
        let first = numeric_vector(&args[0], NAME, "positional X argument")?;
        let second = numeric_vector(&args[1], NAME, "positional Y argument")?;
        extension |= first.len() == 1 && second.len() == 1;
    }
    let mut index = if args.first().is_some_and(|value| !is_option_key(value)) {
        if args.get(1).is_some_and(|value| !is_option_key(value)) {
            2
        } else {
            1
        }
    } else {
        0
    };
    while index + 1 < args.len() {
        let key = tensor::value_to_string(&args[index])
            .unwrap_or_default()
            .to_ascii_lowercase();
        let value = &args[index + 1];
        extension |= matches!(
            key.as_str(),
            "xbinwidth" | "ybinwidth" | "binlimits" | "xbinmethod" | "ybinmethod"
        );
        if key == "binwidth" {
            extension |= numeric_vector(value, NAME, "BinWidth")?.len() == 1;
        }
        if key == "binmethod" {
            extension |= tensor::value_to_string(value).is_some_and(|method| {
                matches!(
                    method.trim().to_ascii_lowercase().as_str(),
                    "sturges" | "sqrt"
                )
            });
        }
        index += 2;
    }
    if extension {
        crate::compatibility::ensure_builtin_extension_enabled(
            &HISTCOUNTS2_EXTRA_SYNTAX_EXTENSION,
            NAME,
        )?;
    }
    Ok(())
}

/// Evaluate `histcounts2` once and surface all outputs.
pub async fn evaluate(x: Value, y: Value, rest: &[Value]) -> BuiltinResult<Histcounts2Evaluation> {
    let options = parse_options(rest)?;
    if matches!(&x, Value::GpuTensor(_)) || matches!(&y, Value::GpuTensor(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(&HISTCOUNTS2_GPU_EXTENSION, NAME)?;
    }
    let x_tensor = match x {
        Value::GpuTensor(handle) => gpu_helpers::gather_tensor_async(&handle).await?,
        other => tensor::value_into_tensor_for(NAME, other).map_err(builtin_error)?,
    };
    let y_tensor = match y {
        Value::GpuTensor(handle) => gpu_helpers::gather_tensor_async(&handle).await?,
        other => tensor::value_into_tensor_for(NAME, other).map_err(builtin_error)?,
    };
    histcounts2_from_tensors(x_tensor, y_tensor, &options)
}

fn histcounts2_from_tensors(
    x_tensor: Tensor,
    y_tensor: Tensor,
    options: &Histcounts2Options,
) -> BuiltinResult<Histcounts2Evaluation> {
    if x_tensor.len() != y_tensor.len() {
        return Err(builtin_error(format!(
            "{NAME}: X and Y must contain the same number of elements"
        )));
    }
    if x_tensor.shape != y_tensor.shape {
        return Err(builtin_error(format!(
            "{NAME}: X and Y must be the same size"
        )));
    }

    // Histogram statistics and returned edges are defined in the floating
    // computation domain even when observations arrive in integer storage.
    let x_shape = x_tensor.shape.clone();
    let y_shape = y_tensor.shape.clone();
    let x_exact = x_tensor
        .integer_storage()
        .map(|storage| storage.exact_values());
    let y_exact = y_tensor
        .integer_storage()
        .map(|storage| storage.exact_values());
    let x_values = tensor::tensor_into_values_f64(x_tensor);
    let y_values = tensor::tensor_into_values_f64(y_tensor);
    let x_axis = collect_axis_data(&x_values);
    let y_axis = collect_axis_data(&y_values);

    let x_edges = compute_edges_for_axis(&x_axis, &options.x, "X")?;
    let y_edges = compute_edges_for_axis(&y_axis, &options.y, "Y")?;

    let x_bin_indices = compute_axis_bin_indices(
        &x_values,
        x_exact.as_deref(),
        &x_edges,
        options.x.exact_edges.as_deref(),
    );
    let y_bin_indices = compute_axis_bin_indices(
        &y_values,
        y_exact.as_deref(),
        &y_edges,
        options.y.exact_edges.as_deref(),
    );
    let counts = compute_histogram_counts(
        &x_bin_indices,
        &y_bin_indices,
        x_edges.len() - 1,
        y_edges.len() - 1,
    );
    let normalised = apply_normalization_2d(
        &counts,
        &x_edges,
        &y_edges,
        options.normalization,
        x_values.len(),
    );

    let x_bins = x_edges.len() - 1;
    let y_bins = y_edges.len() - 1;
    let counts_tensor = Tensor::new(normalised, vec![x_bins, y_bins])
        .map_err(|e| builtin_error(format!("{NAME}: {e}")))?;
    let x_edges_tensor = Tensor::new(x_edges.clone(), vec![1, x_edges.len()])
        .map_err(|e| builtin_error(format!("{NAME}: {e}")))?;
    let y_edges_tensor = Tensor::new(y_edges.clone(), vec![1, y_edges.len()])
        .map_err(|e| builtin_error(format!("{NAME}: {e}")))?;
    let x_bins_tensor =
        Tensor::new(x_bin_indices, x_shape).map_err(|e| builtin_error(format!("{NAME}: {e}")))?;
    let y_bins_tensor =
        Tensor::new(y_bin_indices, y_shape).map_err(|e| builtin_error(format!("{NAME}: {e}")))?;

    Ok(Histcounts2Evaluation::new(
        counts_tensor,
        x_edges_tensor,
        y_edges_tensor,
        x_bins_tensor,
        y_bins_tensor,
    ))
}

fn compute_histogram_counts(
    x_indices: &[f64],
    y_indices: &[f64],
    x_bins: usize,
    y_bins: usize,
) -> Vec<f64> {
    let mut counts = vec![0.0f64; x_bins * y_bins];
    for (&x, &y) in x_indices.iter().zip(y_indices) {
        if x > 0.0 && y > 0.0 {
            let ix = x as usize - 1;
            let iy = y as usize - 1;
            let idx = ix + iy * x_bins;
            if idx < counts.len() {
                counts[idx] += 1.0;
            }
        }
    }
    counts
}

fn compute_axis_bin_indices(
    values: &[f64],
    exact_values: Option<&[IntValue]>,
    edges: &[f64],
    exact_edges: Option<&[IntValue]>,
) -> Vec<f64> {
    if let (Some(values), Some(edges_exact)) = (exact_values, exact_edges) {
        if values
            .first()
            .zip(edges_exact.first())
            .is_some_and(|(value, edge)| {
                std::mem::discriminant(value) == std::mem::discriminant(edge)
            })
        {
            return values
                .iter()
                .map(|value| {
                    exact_integer_bin_index(value, edges_exact).map_or(0.0, |i| (i + 1) as f64)
                })
                .collect();
        }
    }
    values
        .iter()
        .map(|&value| find_bin_index(value, edges).map_or(0.0, |i| (i + 1) as f64))
        .collect()
}

fn exact_integer_bin_index(value: &IntValue, edges: &[IntValue]) -> Option<usize> {
    if edges.len() < 2
        || exact_int_cmp(value, &edges[0])? == Ordering::Less
        || exact_int_cmp(value, edges.last()?)? == Ordering::Greater
    {
        return None;
    }
    if value == edges.last()? {
        return Some(edges.len() - 2);
    }
    (0..edges.len() - 1).find(|&index| {
        exact_int_cmp(value, &edges[index]).is_some_and(|order| order != Ordering::Less)
            && exact_int_cmp(value, &edges[index + 1]) == Some(Ordering::Less)
    })
}

fn exact_int_cmp(left: &IntValue, right: &IntValue) -> Option<Ordering> {
    macro_rules! same_variant_cmp {
        ($($variant:ident),+ $(,)?) => {
            match (left, right) {
                $((IntValue::$variant(a), IntValue::$variant(b)) => Some(a.cmp(b)),)+
                _ => None,
            }
        };
    }
    same_variant_cmp!(I8, I16, I32, I64, U8, U16, U32, U64)
}

#[derive(Clone)]
struct AxisData {
    values: Vec<f64>,
    min_val: Option<f64>,
    max_val: Option<f64>,
    original_range_zero: bool,
}

fn collect_axis_data(values: &[f64]) -> AxisData {
    let mut filtered = Vec::new();
    let mut min_val: Option<f64> = None;
    let mut max_val: Option<f64> = None;

    for &sample in values {
        if sample.is_nan() {
            continue;
        }
        filtered.push(sample);
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
        (Some(minimum), Some(maximum)) => approx_equal(minimum, maximum),
        _ => false,
    };

    AxisData {
        values: filtered,
        min_val,
        max_val,
        original_range_zero,
    }
}

fn compute_edges_for_axis(
    data: &AxisData,
    options: &AxisOptions,
    axis: &str,
) -> BuiltinResult<Vec<f64>> {
    if let Some(edges) = &options.explicit_edges {
        if options.exact_edges.is_none() {
            validate_edges(edges, axis)?;
        }
        return Ok(edges.clone());
    }

    if let Some(method) = options.bin_method {
        return compute_edges_with_method(
            &data.values,
            data.min_val,
            data.max_val,
            data.original_range_zero,
            method,
            options,
            axis,
        );
    }

    compute_edges_standard(
        data.min_val,
        data.max_val,
        data.original_range_zero,
        options,
        axis,
    )
}

fn compute_edges_standard(
    min_val: Option<f64>,
    max_val: Option<f64>,
    original_range_zero: bool,
    options: &AxisOptions,
    axis: &str,
) -> BuiltinResult<Vec<f64>> {
    let (mut lower, mut upper) = derive_initial_limits(min_val, max_val, options.bin_limits);

    if !lower.is_finite() || !upper.is_finite() {
        return Err(builtin_error(format!(
            "{NAME}: data range for {axis} must be finite; specify {axis}BinLimits or {axis}BinEdges"
        )));
    }

    if upper < lower {
        return Err(builtin_error(format!(
            "{NAME}: {axis} bin limits must be increasing"
        )));
    }

    if options.bin_limits.is_some() && approx_equal(lower, upper) {
        return Err(builtin_error(format!(
            "{NAME}: {axis}BinLimits must specify a non-zero width"
        )));
    }

    if let Some(width) = options.bin_width {
        if !width.is_finite() || width <= 0.0 {
            return Err(builtin_error(format!(
                "{NAME}: {axis}BinWidth must be a positive finite scalar"
            )));
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

        if approx_equal(lower, upper) {
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
        validate_edges(&edges, axis)?;
        return Ok(edges);
    }

    let mut num_bins = options.num_bins.unwrap_or(DEFAULT_BIN_COUNT);
    if num_bins == 0 {
        return Err(builtin_error(format!(
            "{NAME}: NumBins must be a positive integer"
        )));
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

    if approx_equal(lower, upper) {
        upper = lower + 1.0;
    }

    let edges = linspace(lower, upper, num_bins + 1);
    validate_edges(&edges, axis)?;
    Ok(edges)
}

fn compute_edges_with_method(
    values: &[f64],
    min_val: Option<f64>,
    max_val: Option<f64>,
    original_range_zero: bool,
    method: BinMethod,
    options: &AxisOptions,
    axis: &str,
) -> BuiltinResult<Vec<f64>> {
    if values.is_empty() {
        return compute_edges_standard(min_val, max_val, original_range_zero, options, axis);
    }

    if matches!(method, BinMethod::Integers) {
        let edges = compute_integer_edges(min_val, max_val, options, axis)?;
        validate_edges(&edges, axis)?;
        return Ok(edges);
    }

    let (lower, upper) = derive_initial_limits(min_val, max_val, options.bin_limits);
    if !lower.is_finite() || !upper.is_finite() {
        return Err(builtin_error(format!(
            "{NAME}: {axis} data range must be finite for BinMethod"
        )));
    }

    if approx_equal(lower, upper) {
        if options.bin_limits.is_some() {
            return Err(builtin_error(format!(
                "{NAME}: {axis}BinLimits must specify a non-zero width"
            )));
        }
        return compute_edges_standard(min_val, max_val, true, options, axis);
    }

    let finite_values: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if finite_values.is_empty() {
        return compute_edges_standard(min_val, max_val, original_range_zero, options, axis);
    }

    let range = upper - lower;
    if range <= 0.0 {
        return compute_edges_standard(min_val, max_val, true, options, axis);
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
        return compute_edges_standard(min_val, max_val, original_range_zero, options, axis);
    }

    let bins = ((range / width).ceil().max(1.0)) as usize;
    let mut edges = Vec::with_capacity(bins + 1);
    for i in 0..=bins {
        edges.push(lower + width * i as f64);
    }
    if let Some(last) = edges.last_mut() {
        *last = upper;
    }
    validate_edges(&edges, axis)?;
    Ok(edges)
}

fn compute_integer_edges(
    min_val: Option<f64>,
    max_val: Option<f64>,
    options: &AxisOptions,
    axis: &str,
) -> BuiltinResult<Vec<f64>> {
    let (mut lower, mut upper) = derive_initial_limits(min_val, max_val, options.bin_limits);

    if !lower.is_finite() || !upper.is_finite() {
        return Err(builtin_error(format!(
            "{NAME}: {axis}BinLimits must be finite for 'integers' BinMethod"
        )));
    }

    if approx_equal(lower, upper) {
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

    if !approx_equal(*edges.last().unwrap(), upper) {
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

fn find_bin_index(value: f64, edges: &[f64]) -> Option<usize> {
    if value < edges[0] || value > edges[edges.len() - 1] {
        return None;
    }
    if value == edges[edges.len() - 1] {
        return Some(edges.len() - 2);
    }
    match edges.binary_search_by(|edge| edge.partial_cmp(&value).unwrap_or(Ordering::Less)) {
        Ok(index) => {
            if index == 0 {
                Some(0)
            } else if index < edges.len() - 1 {
                Some(index)
            } else {
                Some(edges.len() - 2)
            }
        }
        Err(index) => {
            if index == 0 || index > edges.len() - 1 {
                None
            } else {
                Some(index - 1)
            }
        }
    }
}

fn apply_normalization_2d(
    counts: &[f64],
    x_edges: &[f64],
    y_edges: &[f64],
    mode: HistogramNormalization,
    observation_count: usize,
) -> Vec<f64> {
    let x_bins = x_edges.len() - 1;
    let y_bins = y_edges.len() - 1;
    let total = observation_count as f64;
    let x_widths: Vec<f64> = x_edges.windows(2).map(|pair| pair[1] - pair[0]).collect();
    let y_widths: Vec<f64> = y_edges.windows(2).map(|pair| pair[1] - pair[0]).collect();

    match mode {
        HistogramNormalization::Count => counts.to_vec(),
        HistogramNormalization::Probability => {
            if total > 0.0 {
                counts.iter().map(|&c| c / total).collect()
            } else {
                vec![0.0; counts.len()]
            }
        }
        HistogramNormalization::Percentage => {
            if total > 0.0 {
                counts.iter().map(|&c| 100.0 * c / total).collect()
            } else {
                vec![0.0; counts.len()]
            }
        }
        HistogramNormalization::CountDensity => {
            let mut out = vec![0.0; counts.len()];
            for (iy, y_width) in y_widths.iter().enumerate().take(y_bins) {
                for (ix, x_width) in x_widths.iter().enumerate().take(x_bins) {
                    let idx = ix + iy * x_bins;
                    let area = x_width * y_width;
                    out[idx] = if area > 0.0 { counts[idx] / area } else { 0.0 };
                }
            }
            out
        }
        HistogramNormalization::Pdf => {
            if total > 0.0 {
                let mut out = vec![0.0; counts.len()];
                for (iy, y_width) in y_widths.iter().enumerate().take(y_bins) {
                    for (ix, x_width) in x_widths.iter().enumerate().take(x_bins) {
                        let idx = ix + iy * x_bins;
                        let area = x_width * y_width;
                        out[idx] = if area > 0.0 {
                            counts[idx] / (total * area)
                        } else {
                            0.0
                        };
                    }
                }
                out
            } else {
                vec![0.0; counts.len()]
            }
        }
        HistogramNormalization::CumCount => cumulative_2d(counts, x_bins, y_bins, None),
        HistogramNormalization::Cdf => {
            if total > 0.0 {
                cumulative_2d(counts, x_bins, y_bins, Some(total))
            } else {
                vec![0.0; counts.len()]
            }
        }
    }
}

fn cumulative_2d(counts: &[f64], x_bins: usize, y_bins: usize, divisor: Option<f64>) -> Vec<f64> {
    let mut out = vec![0.0; counts.len()];
    for y in 0..y_bins {
        for x in 0..x_bins {
            let idx = x + y * x_bins;
            let left = if x > 0 { out[idx - 1] } else { 0.0 };
            let below = if y > 0 { out[idx - x_bins] } else { 0.0 };
            let diagonal = if x > 0 && y > 0 {
                out[idx - x_bins - 1]
            } else {
                0.0
            };
            out[idx] = counts[idx] + left + below - diagonal;
        }
    }
    if let Some(divisor) = divisor {
        for value in &mut out {
            *value /= divisor;
        }
    }
    out
}

fn validate_edges(edges: &[f64], axis: &str) -> BuiltinResult<()> {
    if edges.len() < 2 {
        return Err(builtin_error(format!(
            "{NAME}: {axis}BinEdges must contain at least two elements"
        )));
    }
    for pair in edges.windows(2) {
        if pair[0].is_nan() || pair[1].is_nan() {
            return Err(builtin_error(format!(
                "{NAME}: {axis}BinEdges must contain finite numbers"
            )));
        }
        if pair[1] <= pair[0] {
            return Err(builtin_error(format!(
                "{NAME}: {axis}BinEdges must be strictly increasing"
            )));
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

fn approx_equal(a: f64, b: f64) -> bool {
    (a - b).abs() <= RANGE_EPS * (a.abs().max(b.abs()).max(1.0))
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

#[derive(Clone, Default)]
struct Histcounts2Options {
    x: AxisOptions,
    y: AxisOptions,
    normalization: HistogramNormalization,
}

impl Histcounts2Options {
    fn validate(&self) -> BuiltinResult<()> {
        self.x.validate("X")?;
        self.y.validate("Y")?;
        Ok(())
    }
}

#[derive(Clone, Default)]
struct AxisOptions {
    explicit_edges: Option<Vec<f64>>,
    exact_edges: Option<Vec<IntValue>>,
    num_bins: Option<usize>,
    bin_width: Option<f64>,
    bin_limits: Option<(f64, f64)>,
    bin_method: Option<BinMethod>,
}

impl AxisOptions {
    fn validate(&self, axis: &str) -> BuiltinResult<()> {
        if self.explicit_edges.is_some()
            && (self.num_bins.is_some() || self.bin_width.is_some() || self.bin_limits.is_some())
        {
            return Err(builtin_error(format!(
                "{NAME}: {axis}BinEdges cannot be combined with NumBins, {axis}BinWidth, or {axis}BinLimits"
            )));
        }
        if self.bin_method.is_some()
            && (self.explicit_edges.is_some()
                || self.bin_width.is_some()
                || self.num_bins.is_some())
        {
            return Err(descriptor_error(&HISTCOUNTS2_ERROR_BINMETHOD_CONFLICT));
        }
        if self.num_bins.is_some() && self.bin_width.is_some() {
            return Err(builtin_error(format!(
                "{NAME}: specify only one of NumBins or {axis}BinWidth"
            )));
        }
        if let Some((lo, hi)) = self.bin_limits {
            if !lo.is_finite() || !hi.is_finite() {
                return Err(builtin_error(format!(
                    "{NAME}: {axis}BinLimits must be finite"
                )));
            }
            if hi < lo {
                return Err(builtin_error(format!(
                    "{NAME}: {axis}BinLimits must be increasing"
                )));
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
    Percentage,
    CountDensity,
    Pdf,
    CumCount,
    Cdf,
}

#[derive(Clone)]
pub struct Histcounts2Evaluation {
    counts: Tensor,
    x_edges: Tensor,
    y_edges: Tensor,
    x_bins: Tensor,
    y_bins: Tensor,
}

impl Histcounts2Evaluation {
    fn new(
        counts: Tensor,
        x_edges: Tensor,
        y_edges: Tensor,
        x_bins: Tensor,
        y_bins: Tensor,
    ) -> Self {
        Self {
            counts,
            x_edges,
            y_edges,
            x_bins,
            y_bins,
        }
    }

    pub fn into_counts_value(self) -> Value {
        Value::Tensor(self.counts)
    }

    pub fn into_pair(self) -> (Value, Value) {
        let counts = self.counts;
        let x_edges = self.x_edges;
        (Value::Tensor(counts), Value::Tensor(x_edges))
    }

    pub fn into_triple(self) -> (Value, Value, Value) {
        let counts = self.counts;
        let x_edges = self.x_edges;
        let y_edges = self.y_edges;
        (
            Value::Tensor(counts),
            Value::Tensor(x_edges),
            Value::Tensor(y_edges),
        )
    }

    pub fn into_all(self) -> (Value, Value, Value, Value, Value) {
        (
            Value::Tensor(self.counts),
            Value::Tensor(self.x_edges),
            Value::Tensor(self.y_edges),
            Value::Tensor(self.x_bins),
            Value::Tensor(self.y_bins),
        )
    }
}

fn parse_options(args: &[Value]) -> BuiltinResult<Histcounts2Options> {
    let mut options = Histcounts2Options::default();
    let mut index = 0;

    if index < args.len() && !is_option_key(&args[index]) {
        if index + 1 < args.len() && !is_option_key(&args[index + 1]) {
            let x_vec = numeric_vector(&args[index], NAME, "positional X argument")?;
            let y_vec = numeric_vector(&args[index + 1], NAME, "positional Y argument")?;
            if x_vec.len() >= 2 && y_vec.len() >= 2 {
                validate_edge_value(&args[index], &x_vec, "X")?;
                validate_edge_value(&args[index + 1], &y_vec, "Y")?;
                options.x.explicit_edges = Some(x_vec);
                options.y.explicit_edges = Some(y_vec);
                options.x.exact_edges = exact_integer_vector(&args[index]);
                options.y.exact_edges = exact_integer_vector(&args[index + 1]);
            } else if x_vec.len() == 1 && y_vec.len() == 1 {
                let nx = positive_usize(&args[index], NAME, "NumBins")?;
                let ny = positive_usize(&args[index + 1], NAME, "NumBins")?;
                options.x.num_bins = Some(nx);
                options.y.num_bins = Some(ny);
            } else {
                return Err(builtin_error(format!(
                    "{NAME}: positional bin arguments must either specify two edge vectors or two scalar bin counts"
                )));
            }
            index += 2;
        } else {
            let bins = num_bins_vector(&args[index])?;
            match bins.as_slice() {
                [n] => {
                    options.x.num_bins = Some(*n);
                    options.y.num_bins = Some(*n);
                }
                [nx, ny] => {
                    options.x.num_bins = Some(*nx);
                    options.y.num_bins = Some(*ny);
                }
                _ => {
                    return Err(builtin_error(format!(
                        "{NAME}: NumBins must be a scalar or two-element vector"
                    )))
                }
            }
            index += 1;
        }
    }

    while index < args.len() {
        let key = tensor::value_to_string(&args[index])
            .ok_or_else(|| builtin_error(format!("{NAME}: expected name/value pair arguments")))?;
        index += 1;
        if index >= args.len() {
            return Err(builtin_error(format!(
                "{NAME}: missing value for option '{key}'"
            )));
        }
        let value = &args[index];
        index += 1;
        let lowered = key.trim().to_ascii_lowercase();

        match lowered.as_str() {
            "numbins" => {
                let bins = num_bins_vector(value)?;
                match bins.as_slice() {
                    [n] => {
                        options.x.num_bins = Some(*n);
                        options.y.num_bins = Some(*n);
                    }
                    [nx, ny] => {
                        options.x.num_bins = Some(*nx);
                        options.y.num_bins = Some(*ny);
                    }
                    _ => {
                        return Err(builtin_error(format!(
                            "{NAME}: NumBins must be a scalar or two-element vector"
                        )))
                    }
                }
            }
            "xbinwidth" => {
                let width = positive_scalar(value, NAME, "XBinWidth")?;
                options.x.bin_width = Some(width);
            }
            "ybinwidth" => {
                let width = positive_scalar(value, NAME, "YBinWidth")?;
                options.y.bin_width = Some(width);
            }
            "binwidth" => {
                let widths = numeric_vector(value, NAME, "BinWidth")?;
                match widths.len() {
                    1 => {
                        let width = positive_scalar_from_f64(widths[0], "BinWidth")?;
                        options.x.bin_width = Some(width);
                        options.y.bin_width = Some(width);
                    }
                    2 => {
                        let wx = positive_scalar_from_f64(widths[0], "BinWidth")?;
                        let wy = positive_scalar_from_f64(widths[1], "BinWidth")?;
                        options.x.bin_width = Some(wx);
                        options.y.bin_width = Some(wy);
                    }
                    _ => {
                        return Err(builtin_error(format!(
                            "{NAME}: BinWidth must be a scalar or two-element vector"
                        )))
                    }
                }
            }
            "xbinlimits" => {
                let limits = numeric_vector(value, NAME, "XBinLimits")?;
                if limits.len() != 2 {
                    return Err(builtin_error(format!(
                        "{NAME}: XBinLimits must contain exactly two elements"
                    )));
                }
                let lo = limits[0];
                let hi = limits[1];
                if hi < lo {
                    return Err(builtin_error(format!(
                        "{NAME}: XBinLimits must be increasing"
                    )));
                }
                if !lo.is_finite() || !hi.is_finite() {
                    return Err(builtin_error(format!("{NAME}: XBinLimits must be finite")));
                }
                options.x.bin_limits = Some((lo, hi));
            }
            "ybinlimits" => {
                let limits = numeric_vector(value, NAME, "YBinLimits")?;
                if limits.len() != 2 {
                    return Err(builtin_error(format!(
                        "{NAME}: YBinLimits must contain exactly two elements"
                    )));
                }
                let lo = limits[0];
                let hi = limits[1];
                if hi < lo {
                    return Err(builtin_error(format!(
                        "{NAME}: YBinLimits must be increasing"
                    )));
                }
                if !lo.is_finite() || !hi.is_finite() {
                    return Err(builtin_error(format!("{NAME}: YBinLimits must be finite")));
                }
                options.y.bin_limits = Some((lo, hi));
            }
            "binlimits" => {
                let limits = numeric_vector(value, NAME, "BinLimits")?;
                match limits.len() {
                    2 => {
                        let lo = limits[0];
                        let hi = limits[1];
                        if hi < lo {
                            return Err(builtin_error(format!(
                                "{NAME}: BinLimits must be increasing"
                            )));
                        }
                        if !lo.is_finite() || !hi.is_finite() {
                            return Err(builtin_error(format!("{NAME}: BinLimits must be finite")));
                        }
                        options.x.bin_limits = Some((lo, hi));
                        options.y.bin_limits = Some((lo, hi));
                    }
                    4 => {
                        let x_lo = limits[0];
                        let x_hi = limits[1];
                        let y_lo = limits[2];
                        let y_hi = limits[3];
                        if x_hi < x_lo || y_hi < y_lo {
                            return Err(builtin_error(format!(
                                "{NAME}: BinLimits must be increasing"
                            )));
                        }
                        if !x_lo.is_finite()
                            || !x_hi.is_finite()
                            || !y_lo.is_finite()
                            || !y_hi.is_finite()
                        {
                            return Err(builtin_error(format!("{NAME}: BinLimits must be finite")));
                        }
                        options.x.bin_limits = Some((x_lo, x_hi));
                        options.y.bin_limits = Some((y_lo, y_hi));
                    }
                    _ => {
                        return Err(builtin_error(format!(
                            "{NAME}: BinLimits must contain two or four elements"
                        )))
                    }
                }
            }
            "xbinedges" => {
                let edges = numeric_vector(value, NAME, "XBinEdges")?;
                validate_edge_value(value, &edges, "X")?;
                options.x.explicit_edges = Some(edges);
                options.x.exact_edges = exact_integer_vector(value);
            }
            "ybinedges" => {
                let edges = numeric_vector(value, NAME, "YBinEdges")?;
                validate_edge_value(value, &edges, "Y")?;
                options.y.explicit_edges = Some(edges);
                options.y.exact_edges = exact_integer_vector(value);
            }
            "binmethod" => {
                let text = tensor::value_to_string(value)
                    .ok_or_else(|| builtin_error(format!("{NAME}: BinMethod must be a string")))?;
                let method = parse_bin_method(&text)?;
                options.x.bin_method = Some(method);
                options.y.bin_method = Some(method);
            }
            "xbinmethod" => {
                let text = tensor::value_to_string(value)
                    .ok_or_else(|| builtin_error(format!("{NAME}: XBinMethod must be a string")))?;
                options.x.bin_method = Some(parse_bin_method(&text)?);
            }
            "ybinmethod" => {
                let text = tensor::value_to_string(value)
                    .ok_or_else(|| builtin_error(format!("{NAME}: YBinMethod must be a string")))?;
                options.y.bin_method = Some(parse_bin_method(&text)?);
            }
            "normalization" => {
                let text = tensor::value_to_string(value).ok_or_else(|| {
                    builtin_error(format!("{NAME}: Normalization must be a string"))
                })?;
                options.normalization = parse_normalization(&text)?;
            }
            other => {
                return Err(builtin_error(format!(
                    "{NAME}: unrecognised option '{other}'"
                )));
            }
        }
    }

    options.validate()?;
    Ok(options)
}

fn is_option_key(value: &Value) -> bool {
    matches!(
        value,
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_)
    )
}

fn numeric_vector(value: &Value, name: &str, option: &str) -> BuiltinResult<Vec<f64>> {
    let tensor = tensor::value_to_tensor(value)
        .map_err(|_| builtin_error(format!("{name}: {option} must be numeric")))?;
    Ok(tensor::tensor_into_values_f64(tensor))
}

fn exact_integer_vector(value: &Value) -> Option<Vec<IntValue>> {
    match value {
        Value::Int(value) => Some(vec![value.clone()]),
        Value::Tensor(tensor) => tensor
            .integer_storage()
            .map(|storage| storage.exact_values()),
        _ => None,
    }
}

fn validate_edge_value(value: &Value, floating: &[f64], axis: &str) -> BuiltinResult<()> {
    if let Some(exact) = exact_integer_vector(value) {
        if exact.len() < 2 {
            return Err(builtin_error(format!(
                "{NAME}: {axis}BinEdges must contain at least two elements"
            )));
        }
        if exact
            .windows(2)
            .any(|pair| exact_int_cmp(&pair[0], &pair[1]) != Some(Ordering::Less))
        {
            return Err(builtin_error(format!(
                "{NAME}: {axis}BinEdges must be strictly increasing"
            )));
        }
        Ok(())
    } else {
        validate_edges(floating, axis)
    }
}

fn num_bins_vector(value: &Value) -> BuiltinResult<Vec<usize>> {
    if let Some(integer) = tensor::scalar_integer_value(value) {
        return Ok(vec![integer
            .try_to_usize()
            .filter(valid_num_bins)
            .ok_or_else(|| {
                descriptor_error(&HISTCOUNTS2_ERROR_NUMBINS_INVALID)
            })?]);
    }
    let tensor = tensor::value_to_tensor(value)
        .map_err(|_| builtin_error(format!("{NAME}: NumBins must be numeric")))?;
    if let Some(storage) = tensor.integer_storage() {
        return storage
            .exact_values()
            .into_iter()
            .map(|value| {
                value
                    .try_to_usize()
                    .filter(valid_num_bins)
                    .ok_or_else(|| descriptor_error(&HISTCOUNTS2_ERROR_NUMBINS_INVALID))
            })
            .collect();
    }
    tensor
        .materialize_f64()
        .iter()
        .map(|value| positive_usize_from_f64(*value, "NumBins"))
        .collect()
}

fn positive_usize(value: &Value, name: &str, option: &str) -> BuiltinResult<usize> {
    if let Some(value) = tensor::scalar_integer_value(value) {
        return value.try_to_usize().filter(valid_num_bins).ok_or_else(|| {
            if name == NAME && option == "NumBins" {
                descriptor_error(&HISTCOUNTS2_ERROR_NUMBINS_INVALID)
            } else {
                builtin_error(format!("{name}: {option} must be a positive finite scalar"))
            }
        });
    }
    let scalar = scalar_value(value, name, option)?;
    positive_usize_from_f64(scalar, option)
}

fn valid_num_bins(value: &usize) -> bool {
    *value > 0 && *value < usize::MAX
}

fn positive_scalar(value: &Value, name: &str, option: &str) -> BuiltinResult<f64> {
    let scalar = scalar_value(value, name, option)?;
    if !scalar.is_finite() || scalar <= 0.0 {
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

fn positive_usize_from_f64(value: f64, option: &str) -> BuiltinResult<usize> {
    if !value.is_finite() || value <= 0.0 {
        if option == "NumBins" {
            return Err(descriptor_error(&HISTCOUNTS2_ERROR_NUMBINS_INVALID));
        }
        return Err(builtin_error(format!(
            "{NAME}: {option} must be a positive finite scalar"
        )));
    }
    let rounded = value.round();
    if (value - rounded).abs() > f64::EPSILON {
        return Err(builtin_error(format!(
            "{NAME}: {option} must be an integer"
        )));
    }
    if rounded > usize::MAX as f64 || (usize::BITS == 64 && rounded == usize::MAX as f64) {
        return Err(builtin_error(format!(
            "{NAME}: {option} is outside the supported range"
        )));
    }
    Ok(rounded as usize)
}

fn positive_scalar_from_f64(value: f64, option: &str) -> BuiltinResult<f64> {
    if !value.is_finite() || value <= 0.0 {
        return Err(builtin_error(format!(
            "{NAME}: {option} must be a positive finite scalar"
        )));
    }
    Ok(value)
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
            "{NAME}: unrecognised BinMethod value '{other}'"
        ))),
    }
}

fn parse_normalization(text: &str) -> BuiltinResult<HistogramNormalization> {
    match text.trim().to_ascii_lowercase().as_str() {
        "count" => Ok(HistogramNormalization::Count),
        "probability" => Ok(HistogramNormalization::Probability),
        "percentage" => Ok(HistogramNormalization::Percentage),
        "countdensity" => Ok(HistogramNormalization::CountDensity),
        "pdf" | "probabilitydensity" => Ok(HistogramNormalization::Pdf),
        "cumcount" => Ok(HistogramNormalization::CumCount),
        "cdf" => Ok(HistogramNormalization::Cdf),
        other => Err(builtin_error(format!(
            "{NAME}: unrecognised Normalization value '{other}'"
        ))),
    }
}

#[cfg(test)]
mod tests;
