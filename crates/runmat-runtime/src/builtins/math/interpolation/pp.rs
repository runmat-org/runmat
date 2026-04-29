use runmat_builtins::{StructValue, Tensor, Value};

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
    pub trailing_shape: Vec<usize>,
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
}

pub fn interp_error(name: &'static str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(name).build()
}

pub fn parse_method(value: &Value, name: &'static str) -> BuiltinResult<Option<InterpMethod>> {
    let Some(keyword) = keyword_of(value) else {
        return Ok(None);
    };
    let method = match keyword.as_str() {
        "linear" => InterpMethod::Linear,
        "nearest" => InterpMethod::Nearest,
        "spline" => InterpMethod::Spline,
        "pchip" | "cubic" => InterpMethod::Pchip,
        _ => {
            return Err(interp_error(
                name,
                format!("{name}: unsupported interpolation method '{keyword}'"),
            ))
        }
    };
    Ok(Some(method))
}

pub async fn parse_extrapolation(
    value: &Value,
    name: &'static str,
) -> BuiltinResult<Option<Extrapolation>> {
    if let Some(keyword) = keyword_of(value) {
        return match keyword.as_str() {
            "extrap" => Ok(Some(Extrapolation::Extrapolate)),
            _ => Ok(None),
        };
    }
    let gathered = dispatcher::gather_if_needed_async(value).await?;
    let Some(scalar) = tensor::scalar_f64_from_value_async(&gathered)
        .await
        .map_err(|err| interp_error(name, format!("{name}: {err}")))?
    else {
        return Ok(None);
    };
    Ok(Some(Extrapolation::Value(scalar)))
}

pub async fn query_points(value: Value, name: &'static str) -> BuiltinResult<QueryPoints> {
    let gathered = dispatcher::gather_if_needed_async(&value).await?;
    let tensor = tensor::value_into_tensor_for(name, gathered)
        .map_err(|err| interp_error(name, format!("{name}: {err}")))?;
    let shape = canonical_shape(&tensor.shape, tensor.data.len());
    Ok(QueryPoints {
        values: tensor.data,
        shape,
    })
}

pub async fn vector_from_value(
    value: Value,
    label: &str,
    name: &'static str,
) -> BuiltinResult<Vec<f64>> {
    let gathered = dispatcher::gather_if_needed_async(&value).await?;
    let tensor = tensor::value_into_tensor_for(name, gathered)
        .map_err(|err| interp_error(name, format!("{name}: {err}")))?;
    if !is_vector_shape(&tensor.shape) && tensor.data.len() > 1 {
        return Err(interp_error(
            name,
            format!("{name}: {label} must be a vector"),
        ));
    }
    Ok(tensor.data)
}

pub async fn series_from_values(
    x_value: Value,
    y_value: Value,
    name: &'static str,
) -> BuiltinResult<NumericSeries> {
    let x = vector_from_value(x_value, "X", name).await?;
    let y_host = dispatcher::gather_if_needed_async(&y_value).await?;
    let y_tensor = tensor::value_into_tensor_for(name, y_host)
        .map_err(|err| interp_error(name, format!("{name}: {err}")))?;
    series_from_tensor(x, y_tensor, name)
}

pub async fn implicit_series_from_values(
    y_value: Value,
    name: &'static str,
) -> BuiltinResult<NumericSeries> {
    let y_host = dispatcher::gather_if_needed_async(&y_value).await?;
    let y_tensor = tensor::value_into_tensor_for(name, y_host)
        .map_err(|err| interp_error(name, format!("{name}: {err}")))?;
    let n = first_non_singleton_len(&y_tensor).unwrap_or(y_tensor.data.len());
    let x: Vec<f64> = (1..=n).map(|value| value as f64).collect();
    series_from_tensor(x, y_tensor, name)
}

fn series_from_tensor(
    x: Vec<f64>,
    y_tensor: Tensor,
    name: &'static str,
) -> BuiltinResult<NumericSeries> {
    validate_breaks(&x, name)?;
    let n = x.len();
    if y_tensor.data.len() != n && !y_tensor.data.len().is_multiple_of(n) {
        return Err(interp_error(
            name,
            format!("{name}: Y length must match X or have X as its first dimension"),
        ));
    }

    let y_shape = canonical_shape(&y_tensor.shape, y_tensor.data.len());
    let (series, trailing_shape) = if y_tensor.data.len() == n && is_vector_shape(&y_shape) {
        (1, Vec::new())
    } else if y_shape.first().copied() == Some(n) {
        let series = y_tensor.data.len() / n;
        let trailing = if y_shape.len() > 1 {
            y_shape[1..].to_vec()
        } else {
            vec![series]
        };
        (series, trailing)
    } else if y_tensor.data.len() == n {
        (1, Vec::new())
    } else {
        return Err(interp_error(
            name,
            format!("{name}: size of Y must be compatible with X"),
        ));
    };

    Ok(NumericSeries {
        x,
        y: y_tensor.data,
        series,
        trailing_shape,
    })
}

pub fn validate_breaks(x: &[f64], name: &'static str) -> BuiltinResult<()> {
    if x.len() < 2 {
        return Err(interp_error(
            name,
            format!("{name}: X must contain at least two points"),
        ));
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(interp_error(name, format!("{name}: X must be finite")));
    }
    for pair in x.windows(2) {
        if pair[1] <= pair[0] {
            return Err(interp_error(
                name,
                format!("{name}: X must be strictly increasing"),
            ));
        }
    }
    Ok(())
}

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
    let n = series.x.len();
    let pieces = n - 1;
    let order = 4;
    let rows = pieces * series.series;
    let mut coefs = vec![0.0; rows * order];
    for s in 0..series.series {
        let y = &series.y[s * n..(s + 1) * n];
        let local = spline_coefs_for_series(&series.x, y)?;
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
    })
}

pub fn pp_to_value(pp: PiecewisePolynomial, name: &'static str) -> BuiltinResult<Value> {
    let breaks = Tensor::new(pp.breaks.clone(), vec![1, pp.breaks.len()])
        .map_err(|err| interp_error(name, format!("{name}: {err}")))?;
    let coefs = Tensor::new(pp.coefs.clone(), vec![pp.pieces * pp.dim, pp.order])
        .map_err(|err| interp_error(name, format!("{name}: {err}")))?;
    let mut st = StructValue::new();
    st.insert("form", Value::String("pp".to_string()));
    st.insert("breaks", Value::Tensor(breaks));
    st.insert("coefs", Value::Tensor(coefs));
    st.insert("pieces", Value::Num(pp.pieces as f64));
    st.insert("order", Value::Num(pp.order as f64));
    st.insert("dim", Value::Num(pp.dim as f64));
    Ok(Value::Struct(st))
}

pub async fn pp_from_value(value: Value, name: &'static str) -> BuiltinResult<PiecewisePolynomial> {
    let gathered = dispatcher::gather_if_needed_async(&value).await?;
    let Value::Struct(st) = gathered else {
        return Err(interp_error(
            name,
            format!("{name}: first argument must be a pp struct"),
        ));
    };
    let form = st
        .fields
        .get("form")
        .and_then(keyword_of)
        .ok_or_else(|| interp_error(name, format!("{name}: pp struct is missing form")))?;
    if form != "pp" {
        return Err(interp_error(
            name,
            format!("{name}: struct form must be 'pp'"),
        ));
    }
    let breaks_value = st
        .fields
        .get("breaks")
        .ok_or_else(|| interp_error(name, format!("{name}: pp struct is missing breaks")))?
        .clone();
    let breaks = vector_from_value(breaks_value, "breaks", name).await?;
    validate_breaks(&breaks, name)?;

    let coefs_value = st
        .fields
        .get("coefs")
        .ok_or_else(|| interp_error(name, format!("{name}: pp struct is missing coefs")))?
        .clone();
    let coefs_host = dispatcher::gather_if_needed_async(&coefs_value).await?;
    let coefs_tensor = tensor::value_into_tensor_for(name, coefs_host)
        .map_err(|err| interp_error(name, format!("{name}: {err}")))?;

    let pieces = scalar_field(&st, "pieces", name).await? as usize;
    let order = scalar_field(&st, "order", name).await? as usize;
    let dim = scalar_field(&st, "dim", name).await? as usize;
    if pieces + 1 != breaks.len() {
        return Err(interp_error(name, format!("{name}: malformed pp breaks")));
    }
    if order == 0 || dim == 0 {
        return Err(interp_error(
            name,
            format!("{name}: malformed pp dimensions"),
        ));
    }
    if coefs_tensor.data.len() != pieces * dim * order {
        return Err(interp_error(
            name,
            format!("{name}: malformed pp coefficients"),
        ));
    }
    Ok(PiecewisePolynomial {
        breaks,
        coefs: coefs_tensor.data,
        pieces,
        order,
        dim,
    })
}

async fn scalar_field(st: &StructValue, field: &str, name: &'static str) -> BuiltinResult<f64> {
    let value = st
        .fields
        .get(field)
        .ok_or_else(|| interp_error(name, format!("{name}: pp struct is missing {field}")))?;
    let Some(raw) = tensor::scalar_f64_from_value_async(value)
        .await
        .map_err(|err| interp_error(name, format!("{name}: {err}")))?
    else {
        return Err(interp_error(
            name,
            format!("{name}: pp field {field} must be scalar"),
        ));
    };
    if !raw.is_finite() || raw < 1.0 || (raw.round() - raw).abs() > 1e-9 {
        return Err(interp_error(
            name,
            format!("{name}: pp field {field} must be a positive integer"),
        ));
    }
    Ok(raw.round())
}

pub fn evaluate_pp(
    pp: &PiecewisePolynomial,
    query: &QueryPoints,
    extrap: &Extrapolation,
    name: &'static str,
) -> BuiltinResult<Value> {
    let mut out = vec![0.0; query.values.len() * pp.dim];
    for d in 0..pp.dim {
        for (q_index, &xq) in query.values.iter().enumerate() {
            out[q_index + d * query.values.len()] = eval_pp_scalar(pp, d, xq, extrap);
        }
    }
    tensor_from_pp_output(out, query, pp.dim, name)
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
    dim: usize,
    name: &'static str,
) -> BuiltinResult<Value> {
    let shape = if dim == 1 {
        query.shape.clone()
    } else {
        let mut shape = vec![dim];
        shape.extend(query.shape.iter().copied());
        shape
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
    canonical_shape(&tensor.shape, tensor.data.len())
        .into_iter()
        .find(|&dim| dim > 1)
}

fn is_vector_shape(shape: &[usize]) -> bool {
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

fn eval_pp_scalar(pp: &PiecewisePolynomial, series: usize, xq: f64, extrap: &Extrapolation) -> f64 {
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

fn interval_index(x: &[f64], xq: f64, allow_extrapolation: bool) -> Option<usize> {
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

fn out_of_range_value(extrap: &Extrapolation) -> f64 {
    match extrap {
        Extrapolation::Nan => f64::NAN,
        Extrapolation::Extrapolate => f64::NAN,
        Extrapolation::Value(value) => *value,
    }
}

fn spline_coefs_for_series(x: &[f64], y: &[f64]) -> BuiltinResult<Vec<[f64; 4]>> {
    let n = x.len();
    if n == 2 {
        let h = x[1] - x[0];
        return Ok(vec![[0.0, 0.0, (y[1] - y[0]) / h, y[0]]]);
    }
    if n == 3 {
        return Ok(quadratic_as_piecewise_cubic(x, y));
    }

    let h: Vec<f64> = x.windows(2).map(|pair| pair[1] - pair[0]).collect();
    let mut a = vec![vec![0.0; n]; n];
    let mut rhs = vec![0.0; n];

    a[0][0] = -h[1];
    a[0][1] = h[0] + h[1];
    a[0][2] = -h[0];
    for i in 1..(n - 1) {
        a[i][i - 1] = h[i - 1];
        a[i][i] = 2.0 * (h[i - 1] + h[i]);
        a[i][i + 1] = h[i];
        rhs[i] = 6.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1]);
    }
    a[n - 1][n - 3] = -h[n - 2];
    a[n - 1][n - 2] = h[n - 3] + h[n - 2];
    a[n - 1][n - 1] = -h[n - 3];

    let second = solve_dense(a, rhs)?;
    Ok(second_derivatives_to_coefs(x, y, &second))
}

fn quadratic_as_piecewise_cubic(x: &[f64], y: &[f64]) -> Vec<[f64; 4]> {
    let x0 = x[0];
    let x1 = x[1];
    let x2 = x[2];
    let y0 = y[0];
    let y1 = y[1];
    let y2 = y[2];
    let c2 =
        y0 / ((x0 - x1) * (x0 - x2)) + y1 / ((x1 - x0) * (x1 - x2)) + y2 / ((x2 - x0) * (x2 - x1));
    let c1 = -y0 * (x1 + x2) / ((x0 - x1) * (x0 - x2))
        - y1 * (x0 + x2) / ((x1 - x0) * (x1 - x2))
        - y2 * (x0 + x1) / ((x2 - x0) * (x2 - x1));
    let c0 = y0 * x1 * x2 / ((x0 - x1) * (x0 - x2))
        + y1 * x0 * x2 / ((x1 - x0) * (x1 - x2))
        + y2 * x0 * x1 / ((x2 - x0) * (x2 - x1));
    (0..2)
        .map(|i| {
            let xi = x[i];
            let a = c2 * xi * xi + c1 * xi + c0;
            let b = 2.0 * c2 * xi + c1;
            [0.0, c2, b, a]
        })
        .collect()
}

fn second_derivatives_to_coefs(x: &[f64], y: &[f64], second: &[f64]) -> Vec<[f64; 4]> {
    let mut coefs = Vec::with_capacity(x.len() - 1);
    for i in 0..(x.len() - 1) {
        let h = x[i + 1] - x[i];
        let a = y[i];
        let b = (y[i + 1] - y[i]) / h - h * (2.0 * second[i] + second[i + 1]) / 6.0;
        let c = second[i] / 2.0;
        let d = (second[i + 1] - second[i]) / (6.0 * h);
        coefs.push([d, c, b, a]);
    }
    coefs
}

fn pchip_coefs_for_series(x: &[f64], y: &[f64]) -> Vec<[f64; 4]> {
    let n = x.len();
    let h: Vec<f64> = x.windows(2).map(|pair| pair[1] - pair[0]).collect();
    let delta: Vec<f64> = y
        .windows(2)
        .zip(h.iter())
        .map(|(pair, &width)| (pair[1] - pair[0]) / width)
        .collect();

    let mut slopes = vec![0.0; n];
    if n == 2 {
        slopes[0] = delta[0];
        slopes[1] = delta[0];
    } else {
        slopes[0] = endpoint_slope(h[0], h[1], delta[0], delta[1]);
        slopes[n - 1] = endpoint_slope(h[n - 2], h[n - 3], delta[n - 2], delta[n - 3]);
        for i in 1..(n - 1) {
            if delta[i - 1] == 0.0 || delta[i] == 0.0 || delta[i - 1].signum() != delta[i].signum()
            {
                slopes[i] = 0.0;
            } else {
                let w1 = 2.0 * h[i] + h[i - 1];
                let w2 = h[i] + 2.0 * h[i - 1];
                slopes[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i]);
            }
        }
    }

    let mut coefs = Vec::with_capacity(n - 1);
    for i in 0..(n - 1) {
        let c0 = y[i];
        let c1 = slopes[i];
        let c2 = (3.0 * delta[i] - 2.0 * slopes[i] - slopes[i + 1]) / h[i];
        let c3 = (slopes[i] + slopes[i + 1] - 2.0 * delta[i]) / (h[i] * h[i]);
        coefs.push([c3, c2, c1, c0]);
    }
    coefs
}

fn endpoint_slope(h0: f64, h1: f64, del0: f64, del1: f64) -> f64 {
    let mut slope = ((2.0 * h0 + h1) * del0 - h0 * del1) / (h0 + h1);
    if slope.signum() != del0.signum() {
        slope = 0.0;
    } else if del0.signum() != del1.signum() && slope.abs() > (3.0 * del0).abs() {
        slope = 3.0 * del0;
    }
    slope
}

fn solve_dense(mut a: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> BuiltinResult<Vec<f64>> {
    let n = rhs.len();
    for col in 0..n {
        let mut pivot = col;
        let mut pivot_abs = a[col][col].abs();
        for row in (col + 1)..n {
            let candidate = a[row][col].abs();
            if candidate > pivot_abs {
                pivot = row;
                pivot_abs = candidate;
            }
        }
        if pivot_abs <= 1.0e-14 {
            return Err(interp_error(
                "spline",
                "spline: singular interpolation system",
            ));
        }
        if pivot != col {
            a.swap(pivot, col);
            rhs.swap(pivot, col);
        }
        for row in (col + 1)..n {
            let factor = a[row][col] / a[col][col];
            a[row][col] = 0.0;
            for k in (col + 1)..n {
                a[row][k] -= factor * a[col][k];
            }
            rhs[row] -= factor * rhs[col];
        }
    }

    let mut solution = vec![0.0; n];
    for row in (0..n).rev() {
        let mut acc = rhs[row];
        for col in (row + 1)..n {
            acc -= a[row][col] * solution[col];
        }
        solution[row] = acc / a[row][row];
    }
    Ok(solution)
}

fn validate_finite_coefs(coefs: &[f64], name: &'static str) -> BuiltinResult<()> {
    if coefs.iter().any(|value| !value.is_finite()) {
        return Err(interp_error(
            name,
            format!("{name}: interpolation coefficients must be finite"),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_series(y: &[f64]) -> NumericSeries {
        NumericSeries {
            x: (1..=y.len()).map(|v| v as f64).collect(),
            y: y.to_vec(),
            series: 1,
            trailing_shape: Vec::new(),
        }
    }

    #[test]
    fn pchip_preserves_monotone_samples() {
        let series = scalar_series(&[0.0, 1.0, 1.5, 1.75]);
        let pp = build_pchip_pp(&series, "pchip").expect("pchip");
        let query = QueryPoints {
            values: vec![1.25, 1.5, 2.5, 3.5],
            shape: vec![1, 4],
        };
        let value = evaluate_pp(&pp, &query, &Extrapolation::Extrapolate, "pchip").expect("ppval");
        let Value::Tensor(tensor) = value else {
            panic!("expected tensor");
        };
        assert!(tensor.data.windows(2).all(|pair| pair[1] >= pair[0]));
    }

    #[test]
    fn spline_matches_quadratic_for_three_points() {
        let series = scalar_series(&[1.0, 4.0, 9.0]);
        let pp = build_spline_pp(&series, "spline").expect("spline");
        let query = QueryPoints {
            values: vec![1.5, 2.5],
            shape: vec![1, 2],
        };
        let value = evaluate_pp(&pp, &query, &Extrapolation::Extrapolate, "spline").expect("ppval");
        let Value::Tensor(tensor) = value else {
            panic!("expected tensor");
        };
        assert!((tensor.data[0] - 2.25).abs() < 1e-10);
        assert!((tensor.data[1] - 6.25).abs() < 1e-10);
    }
}
