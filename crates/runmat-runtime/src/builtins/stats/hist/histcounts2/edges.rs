use super::options::{AxisOptions, BinMethod};
use super::*;
use crate::builtins::stats::hist::integer_bins::{
    distinct_generated_double_edges, reserve_double_edges, unit_integer_edges, GeneratedEdgeError,
    UnitEdgeError,
};
use runmat_builtins::HISTCOUNTS2_ERROR_EDGE_PRECISION;
use std::cmp::Ordering;
mod heuristics;
mod normalization;
use heuristics::{freedman_diaconis_width, scott_width, sqrt_bin_count, sturges_bin_count};
pub(super) use normalization::apply_normalization_2d;
#[derive(Clone)]
pub(super) struct AxisData {
    values: Vec<f64>,
    min_val: Option<f64>,
    max_val: Option<f64>,
    original_range_zero: bool,
}

pub(super) fn collect_axis_data(values: &[f64]) -> AxisData {
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
        (Some(minimum), Some(maximum)) => minimum == maximum,
        _ => false,
    };

    AxisData {
        values: filtered,
        min_val,
        max_val,
        original_range_zero,
    }
}

pub(super) fn compute_edges_for_axis(
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

    if options.bin_limits.is_some() && lower == upper {
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

        if lower == upper {
            upper = lower + width;
        }

        let bins = ((upper - lower) / width).ceil().max(1.0) as usize;
        let capacity = bins
            .checked_add(1)
            .ok_or_else(|| edge_resource_error(axis))?;
        let mut edges = reserve_double_edges(capacity).ok_or_else(|| edge_resource_error(axis))?;
        for i in 0..=bins {
            edges.push(lower + width * i as f64);
        }
        if let Some(last) = edges.last_mut() {
            *last = upper;
        }
        return finalize_generated_edges(edges, axis, true);
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

    if lower == upper {
        upper = lower + 1.0;
    }

    let capacity = num_bins
        .checked_add(1)
        .ok_or_else(|| edge_resource_error(axis))?;
    finalize_generated_edges(
        linspace(lower, upper, capacity, axis)?,
        axis,
        options.num_bins.is_some(),
    )
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

    if lower == upper {
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
    let capacity = bins
        .checked_add(1)
        .ok_or_else(|| edge_resource_error(axis))?;
    let mut edges = reserve_double_edges(capacity).ok_or_else(|| edge_resource_error(axis))?;
    for i in 0..=bins {
        edges.push(lower + width * i as f64);
    }
    if let Some(last) = edges.last_mut() {
        *last = upper;
    }
    finalize_generated_edges(edges, axis, false)
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
        UnitEdgeError::UnrepresentableBoundary => builtin_error_with(
            &HISTCOUNTS2_ERROR_EDGE_PRECISION,
            format!(
                "{NAME}: {axis} integer-aligned bin boundaries cannot be represented distinctly as double"
            ),
        ),
        UnitEdgeError::TooManyEdges => builtin_error(format!(
            "{NAME}: {axis} integer-aligned bin count exceeds available resources"
        )),
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

pub(super) fn find_bin_index(value: f64, edges: &[f64]) -> Option<usize> {
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

pub(super) fn validate_edges(edges: &[f64], axis: &str) -> BuiltinResult<()> {
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

fn edge_resource_error(axis: &str) -> RuntimeError {
    builtin_error(format!(
        "{NAME}: {axis} generated bin count exceeds available resources"
    ))
}

fn linspace(start: f64, stop: f64, count: usize, axis: &str) -> BuiltinResult<Vec<f64>> {
    if count <= 1 {
        return Ok(vec![start]);
    }
    let step = (stop - start) / (count - 1) as f64;
    let mut out = reserve_double_edges(count).ok_or_else(|| edge_resource_error(axis))?;
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
    axis: &str,
    exact_count_required: bool,
) -> BuiltinResult<Vec<f64>> {
    let original_count = edges.len();
    let compact = distinct_generated_double_edges(edges).map_err(|error| match error {
        GeneratedEdgeError::Nonfinite => builtin_error(format!(
            "{NAME}: {axis} generated bin edges must be finite numbers"
        )),
        GeneratedEdgeError::Unrepresentable => builtin_error_with(
            &HISTCOUNTS2_ERROR_EDGE_PRECISION,
            format!("{NAME}: {axis} bin boundaries cannot be represented distinctly as double"),
        ),
    })?;
    if exact_count_required && compact.len() != original_count {
        return Err(builtin_error_with(
            &HISTCOUNTS2_ERROR_EDGE_PRECISION,
            format!("{NAME}: {axis} bin boundaries cannot be represented distinctly as double"),
        ));
    }
    validate_edges(&compact, axis)?;
    Ok(compact)
}
