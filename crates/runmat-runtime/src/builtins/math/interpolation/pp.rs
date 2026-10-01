use runmat_value::{StructValue, Tensor, Value};

use crate::builtins::common::random_args::keyword_of;
use crate::builtins::common::tensor;
use crate::{build_runtime_error, dispatcher, BuiltinResult, RuntimeError};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InterpMethod {
    Linear,
    Nearest,
    Spline,
    Pchip,
}

#[derive(Clone, Debug)]
pub enum Extrapolation {
    Nan,
    Extrapolate,
    Value(f64),
}

impl Default for Extrapolation {
    fn default() -> Self {
        Self::Nan
    }
}

#[derive(Clone, Debug)]
pub struct NumericSeries {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub series: usize,
    /// Shape of the non-sample axes. They trail samples for interp1 and
    /// precede samples for pchip/spline's final-axis array form.
    pub trailing_shape: Vec<usize>,
    pub sample_axis: SampleAxis,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EndpointSlopes {
    pub first: f64,
    pub last: f64,
}

#[derive(Clone, Debug)]
pub struct SplineSamples {
    pub series: NumericSeries,
    pub endpoint_slopes: Option<Vec<EndpointSlopes>>,
}

/// Where samples lie in a nonscalar input. Interp1 consumes columns, while
/// pchip and spline consume the final axis of an array-valued function.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampleAxis {
    First,
    Last,
}

#[derive(Clone, Debug)]
pub struct QueryPoints {
    pub values: Vec<f64>,
    pub shape: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct PiecewisePolynomial {
    pub breaks: Vec<f64>,
    pub coefs: Vec<f64>,
    pub pieces: usize,
    pub order: usize,
    pub dim: usize,
    /// Component dimensions before query dimensions in public pp evaluation.
    /// The scalar `dim` remains the checked product for coefficient indexing.
    pub component_shape: Vec<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PpOutputLayout {
    /// Public ppval/pchip/spline place function-value axes before query axes.
    ComponentLeading,
    /// Interp1 retains its existing query-leading matrix/array presentation.
    QueryLeading,
}

pub fn interp_error(name: &'static str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(name).build()
}

async fn materialize_if_resident(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(_) => dispatcher::gather_if_needed_async(&value).await,
        host => Ok(host),
    }
}

#[path = "pp/series.rs"]
mod series;
pub use series::*;

pub fn evaluate_linear_or_nearest(
    series: &NumericSeries,
    query: &QueryPoints,
    method: InterpMethod,
    extrap: &Extrapolation,
    name: &'static str,
) -> BuiltinResult<Value> {
    let mut out = vec![0.0; query.values.len() * series.series];
    for s in 0..series.series {
        let y = &series.y[s * series.x.len()..(s + 1) * series.x.len()];
        for (q_index, &xq) in query.values.iter().enumerate() {
            out[q_index + s * query.values.len()] = match method {
                InterpMethod::Linear => eval_linear(&series.x, y, xq, extrap),
                InterpMethod::Nearest => eval_nearest(&series.x, y, xq, extrap),
                _ => unreachable!("only direct methods are accepted here"),
            };
        }
    }
    tensor_from_query_output(out, query, series, name)
}

pub fn build_spline_pp(
    series: &NumericSeries,
    name: &'static str,
) -> BuiltinResult<PiecewisePolynomial> {
    build_spline_pp_with_slopes(series, None, name)
}

pub fn build_spline_pp_with_slopes(
    series: &NumericSeries,
    endpoint_slopes: Option<&[EndpointSlopes]>,
    name: &'static str,
) -> BuiltinResult<PiecewisePolynomial> {
    let n = series.x.len();
    if endpoint_slopes.is_some_and(|slopes| slopes.len() != series.series) {
        return Err(interp_error(
            name,
            format!("{name}: endpoint slope shape mismatch"),
        ));
    }
    let pieces = n - 1;
    let order = 4;
    let rows = pieces * series.series;
    let mut coefs = vec![0.0; rows * order];
    for s in 0..series.series {
        let y = &series.y[s * n..(s + 1) * n];
        let local = spline_coefs_for_series(&series.x, y, endpoint_slopes.map(|slopes| slopes[s]))?;
        for piece in 0..pieces {
            let row = s * pieces + piece;
            for col in 0..order {
                coefs[row + col * rows] = local[piece][col];
            }
        }
    }
    validate_finite_coefs(&coefs, name)?;
    Ok(PiecewisePolynomial {
        breaks: series.x.clone(),
        coefs,
        pieces,
        order,
        dim: series.series,
        component_shape: series.trailing_shape.clone(),
    })
}

pub fn build_pchip_pp(
    series: &NumericSeries,
    name: &'static str,
) -> BuiltinResult<PiecewisePolynomial> {
    let n = series.x.len();
    let pieces = n - 1;
    let order = 4;
    let rows = pieces * series.series;
    let mut coefs = vec![0.0; rows * order];
    for s in 0..series.series {
        let y = &series.y[s * n..(s + 1) * n];
        let local = pchip_coefs_for_series(&series.x, y);
        for piece in 0..pieces {
            let row = s * pieces + piece;
            for col in 0..order {
                coefs[row + col * rows] = local[piece][col];
            }
        }
    }
    validate_finite_coefs(&coefs, name)?;
    Ok(PiecewisePolynomial {
        breaks: series.x.clone(),
        coefs,
        pieces,
        order,
        dim: series.series,
        component_shape: series.trailing_shape.clone(),
    })
}

#[path = "pp/format.rs"]
mod format;
pub use format::{pp_from_value, pp_to_value};

pub fn evaluate_pp(
    pp: &PiecewisePolynomial,
    query: &QueryPoints,
    extrap: &Extrapolation,
    layout: PpOutputLayout,
    name: &'static str,
) -> BuiltinResult<Value> {
    let output_count = query
        .values
        .len()
        .checked_mul(pp.dim)
        .ok_or_else(|| interp_error(name, format!("{name}: output size overflow")))?;
    let mut out = vec![0.0; output_count];
    for d in 0..pp.dim {
        for (q_index, &xq) in query.values.iter().enumerate() {
            let offset = match layout {
                PpOutputLayout::ComponentLeading => d + q_index * pp.dim,
                PpOutputLayout::QueryLeading => q_index + d * query.values.len(),
            };
            out[offset] = eval_pp_scalar(pp, d, xq, extrap);
        }
    }
    tensor_from_pp_output(out, query, pp, layout, name)
}

pub fn tensor_from_query_output(
    data: Vec<f64>,
    query: &QueryPoints,
    series: &NumericSeries,
    name: &'static str,
) -> BuiltinResult<Value> {
    let shape = output_shape(&query.shape, series.series, &series.trailing_shape);
    value_from_data(data, shape, name)
}

fn tensor_from_pp_output(
    data: Vec<f64>,
    query: &QueryPoints,
    pp: &PiecewisePolynomial,
    layout: PpOutputLayout,
    name: &'static str,
) -> BuiltinResult<Value> {
    let shape = match layout {
        PpOutputLayout::ComponentLeading if pp.component_shape.is_empty() => query.shape.clone(),
        PpOutputLayout::ComponentLeading => {
            let mut shape = pp.component_shape.clone();
            if is_vector_shape(&query.shape) {
                shape.push(query.values.len());
            } else {
                shape.extend(query.shape.iter().copied());
            }
            shape
        }
        PpOutputLayout::QueryLeading if pp.dim == 1 => query.shape.clone(),
        PpOutputLayout::QueryLeading => {
            let mut shape = query.shape.clone();
            shape.extend(pp.component_shape.iter().copied());
            shape
        }
    };
    value_from_data(data, shape, name)
}

fn value_from_data(data: Vec<f64>, shape: Vec<usize>, name: &'static str) -> BuiltinResult<Value> {
    if data.len() == 1 {
        return Ok(Value::Num(data[0]));
    }
    let tensor =
        Tensor::new(data, shape).map_err(|err| interp_error(name, format!("{name}: {err}")))?;
    Ok(Value::Tensor(tensor))
}

fn output_shape(query_shape: &[usize], series: usize, trailing_shape: &[usize]) -> Vec<usize> {
    if series == 1 {
        return query_shape.to_vec();
    }
    let mut shape = query_shape.to_vec();
    if trailing_shape.is_empty() {
        shape.push(series);
    } else {
        shape.extend(trailing_shape.iter().copied());
    }
    shape
}

fn canonical_shape(shape: &[usize], len: usize) -> Vec<usize> {
    tensor::default_shape_for(shape, len)
}

fn first_non_singleton_len(tensor: &Tensor) -> Option<usize> {
    canonical_shape(&tensor.shape, tensor_len(tensor))
        .into_iter()
        .find(|&dim| dim > 1)
}

fn tensor_len(tensor: &Tensor) -> usize {
    tensor::tensor_element_len(tensor)
}

pub(super) fn is_vector_shape(shape: &[usize]) -> bool {
    match shape {
        [] => true,
        [_] => true,
        [rows, cols] => *rows == 1 || *cols == 1,
        dims => dims.iter().filter(|&&dim| dim > 1).count() <= 1,
    }
}

fn eval_linear(x: &[f64], y: &[f64], xq: f64, extrap: &Extrapolation) -> f64 {
    if !xq.is_finite() {
        return f64::NAN;
    }
    let Some(piece) = interval_index(x, xq, matches!(extrap, Extrapolation::Extrapolate)) else {
        return out_of_range_value(extrap);
    };
    let h = x[piece + 1] - x[piece];
    let t = (xq - x[piece]) / h;
    y[piece] + t * (y[piece + 1] - y[piece])
}

fn eval_nearest(x: &[f64], y: &[f64], xq: f64, extrap: &Extrapolation) -> f64 {
    if !xq.is_finite() {
        return f64::NAN;
    }
    if xq < x[0] {
        return match extrap {
            Extrapolation::Extrapolate => y[0],
            _ => out_of_range_value(extrap),
        };
    }
    if xq > x[x.len() - 1] {
        return match extrap {
            Extrapolation::Extrapolate => y[y.len() - 1],
            _ => out_of_range_value(extrap),
        };
    }
    match x.binary_search_by(|probe| probe.partial_cmp(&xq).unwrap()) {
        Ok(index) => y[index],
        Err(index) => {
            let left = index.saturating_sub(1);
            let right = index.min(x.len() - 1);
            if (xq - x[left]).abs() <= (x[right] - xq).abs() {
                y[left]
            } else {
                y[right]
            }
        }
    }
}

pub(super) fn eval_pp_scalar(
    pp: &PiecewisePolynomial,
    series: usize,
    xq: f64,
    extrap: &Extrapolation,
) -> f64 {
    if !xq.is_finite() {
        return f64::NAN;
    }
    let Some(piece) = interval_index(&pp.breaks, xq, matches!(extrap, Extrapolation::Extrapolate))
    else {
        return out_of_range_value(extrap);
    };
    let t = xq - pp.breaks[piece];
    let rows = pp.pieces * pp.dim;
    let row = series * pp.pieces + piece;
    let mut acc = 0.0;
    for col in 0..pp.order {
        acc = acc * t + pp.coefs[row + col * rows];
    }
    acc
}

pub(super) fn interval_index(x: &[f64], xq: f64, allow_extrapolation: bool) -> Option<usize> {
    if xq < x[0] {
        return allow_extrapolation.then_some(0);
    }
    let last = x.len() - 1;
    if xq > x[last] {
        return allow_extrapolation.then_some(last - 1);
    }
    if xq == x[last] {
        return Some(last - 1);
    }
    match x.binary_search_by(|probe| probe.partial_cmp(&xq).unwrap()) {
        Ok(index) => Some(index.min(last - 1)),
        Err(index) if index > 0 && index < x.len() => Some(index - 1),
        _ => None,
    }
}

pub(super) fn out_of_range_value(extrap: &Extrapolation) -> f64 {
    match extrap {
        Extrapolation::Nan => f64::NAN,
        Extrapolation::Extrapolate => f64::NAN,
        Extrapolation::Value(value) => *value,
    }
}

#[path = "pp/coefficients.rs"]
mod coefficients;
use coefficients::{pchip_coefs_for_series, spline_coefs_for_series, validate_finite_coefs};

#[cfg(test)]
#[path = "pp/tests.rs"]
mod tests;
