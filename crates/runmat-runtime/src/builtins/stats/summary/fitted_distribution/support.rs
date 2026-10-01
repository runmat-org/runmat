use super::*;

pub(super) async fn parse_shape_args(rest: &[Value]) -> BuiltinResult<Vec<usize>> {
    if rest.is_empty() {
        return Ok(vec![1, 1]);
    }
    let mut dims = Vec::new();
    for arg in rest {
        match extract_dims(arg, RANDOM_NAME).await {
            Ok(Some(values)) => dims.extend(values),
            Ok(None) => return Err(invalid("random: invalid size argument")),
            Err(err) => return Err(invalid(err)),
        }
    }
    Ok(normalize_dims(dims))
}

pub(super) fn normalize_dims(dims: Vec<usize>) -> Vec<usize> {
    if dims.is_empty() {
        vec![0, 0]
    } else if dims.len() == 1 {
        vec![dims[0], dims[0]]
    } else {
        dims
    }
}

pub(super) fn finish_for(
    builtin: &'static str,
    shape: Vec<usize>,
    data: Vec<f64>,
) -> BuiltinResult<Value> {
    if shape.iter().copied().product::<usize>() == 1 {
        return Ok(Value::Num(data.first().copied().unwrap_or(0.0)));
    }
    Tensor::new(data, shape)
        .map(tensor::tensor_into_value)
        .map_err(|err| internal_for(builtin, format!("{builtin}: {err}")))
}

pub(super) fn two(values: &[f64]) -> [f64; 2] {
    [values[0], values[1]]
}

pub(super) fn invert_positive(p: f64, initial_hi: f64, cdf: impl Fn(f64) -> f64) -> f64 {
    if p == 0.0 {
        return 0.0;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    let mut lo = 0.0;
    let mut hi = initial_hi.max(1.0);
    let mut iter = 0;
    while cdf(hi) < p {
        hi *= 2.0;
        iter += 1;
        if !hi.is_finite() || iter > 2048 {
            return f64::INFINITY;
        }
    }
    for _ in 0..160 {
        let mid = 0.5 * (lo + hi);
        if cdf(mid) >= p {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    0.5 * (lo + hi)
}

pub(super) fn digamma(mut x: f64) -> f64 {
    let mut result = 0.0;
    while x < 8.0 {
        result -= 1.0 / x;
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    result + x.ln() - 0.5 * inv - inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 / 252.0))
}

pub(super) fn trigamma(mut x: f64) -> f64 {
    let mut result = 0.0;
    while x < 8.0 {
        result += 1.0 / (x * x);
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    result + inv + 0.5 * inv2 + inv2 * inv / 6.0 - inv2 * inv2 * inv / 30.0
}
