use super::*;

pub fn evaluate_linear_or_nearest(
    series: &NumericSeries,
    query: &QueryPoints,
    method: InterpMethod,
    extrap: &Extrapolation,
    name: &'static str,
) -> BuiltinResult<Value> {
    series.validate_layout(name)?;
    let len = query
        .values
        .len()
        .checked_mul(series.series)
        .ok_or_else(|| interp_error(name, format!("{name}: output size overflow")))?;
    let mut out = zeroed_buffer(len, name)?;
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

pub fn evaluate_pp(
    pp: &PiecewisePolynomial,
    query: &QueryPoints,
    extrap: &Extrapolation,
    layout: PpOutputLayout,
    name: &'static str,
) -> BuiltinResult<Value> {
    pp.validate_layout(name)?;
    let output_count = query
        .values
        .len()
        .checked_mul(pp.dim)
        .ok_or_else(|| interp_error(name, format!("{name}: output size overflow")))?;
    let mut out = zeroed_buffer(output_count, name)?;
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
    let shape = output_shape(&query.shape, series.series, &series.component_shape);
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

fn output_shape(query_shape: &[usize], series: usize, component_shape: &[usize]) -> Vec<usize> {
    if series == 1 {
        return query_shape.to_vec();
    }
    let mut shape = query_shape.to_vec();
    if component_shape.is_empty() {
        shape.push(series);
    } else {
        shape.extend(component_shape.iter().copied());
    }
    shape
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

pub(in crate::builtins::math::interpolation) fn eval_pp_scalar(
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
    let row = piece * pp.dim + series;
    let mut acc = 0.0;
    for col in 0..pp.order {
        acc = acc * t + pp.coefs[row + col * rows];
    }
    acc
}

pub(in crate::builtins::math::interpolation) fn interval_index(
    x: &[f64],
    xq: f64,
    allow_extrapolation: bool,
) -> Option<usize> {
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

pub(in crate::builtins::math::interpolation) fn out_of_range_value(extrap: &Extrapolation) -> f64 {
    match extrap {
        Extrapolation::Nan => f64::NAN,
        Extrapolation::Extrapolate => f64::NAN,
        Extrapolation::Value(value) => *value,
    }
}
