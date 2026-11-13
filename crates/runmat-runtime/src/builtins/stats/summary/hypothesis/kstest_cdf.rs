//! Tabulated and named CDF normalization for the KS reference distribution.
use super::*;
pub(super) fn parse_kstest_cdf(value: &Value) -> BuiltinResult<CdfSpec> {
    match value {
        Value::Tensor(tensor) => parse_kstest_cdf_table(tensor),
        Value::String(runmat_value::StringElement::Text(text)) if KstestNamedCdf::from_public(text).is_some() => {
            Ok(CdfSpec::StandardNormal)
        }
        Value::CharArray(chars) if chars.rows == 1 => {
            let text = crate::builtins::stats::decode_utf16_with_error(
                chars.data(),
                || kstest_invalid_argument("kstest: CDF name is not valid UTF-16 text"),
            )?;
            if KstestNamedCdf::from_public(&text).is_some() {
                Ok(CdfSpec::StandardNormal)
            } else {
                Err(kstest_invalid_argument(format!(
                    "kstest: unsupported CDF '{text}'"
                )))
            }
        }
        Value::FunctionHandle(name) if KstestNamedCdf::from_public(name).is_some() => {
            Ok(CdfSpec::StandardNormal)
        }
        Value::Cell(cell) if cell.data().len() == 1 => parse_kstest_cdf(&cell.data()[0]),
        other => Err(kstest_invalid_argument(format!(
            "kstest: CDF must be a two-column numeric matrix or normcdf handle; probability distribution objects are not supported yet, got {other:?}"
        ))),
    }
}

pub(super) fn parse_kstest_cdf_table(tensor: &Tensor) -> BuiltinResult<CdfSpec> {
    ensure_exact_kstest_integer_tensor(tensor, "CDF table")?;
    let shape = tensor.shape();
    if shape.len() != 2 || shape[1] != 2 || shape[0] < 2 {
        return Err(kstest_invalid_argument(
            "kstest: CDF table must be an n-by-2 numeric matrix",
        ));
    }
    let rows = shape[0];
    let mut points = Vec::with_capacity(rows);
    for row in 0..rows {
        let x = tensor::tensor_value_f64(tensor, row);
        let f = tensor::tensor_value_f64(tensor, row + rows);
        if !x.is_finite() || !f.is_finite() || !(0.0..=1.0).contains(&f) {
            return Err(kstest_invalid_argument(
                "kstest: CDF table values must be finite and probabilities must be in [0,1]",
            ));
        }
        points.push((x, f));
    }
    points.sort_by(|(x_a, f_a), (x_b, f_b)| x_a.total_cmp(x_b).then_with(|| f_a.total_cmp(f_b)));

    let mut normalized = Vec::with_capacity(points.len());
    for (x, f) in points {
        if let Some((last_x, last_f)) = normalized.last_mut() {
            if *last_x == x {
                if f > *last_f {
                    *last_f = f;
                }
                continue;
            }
            if f < *last_f {
                return Err(kstest_invalid_argument(
                    "kstest: CDF table probabilities must be nondecreasing after sorting by x",
                ));
            }
        }
        normalized.push((x, f));
    }

    if normalized.len() < 2 {
        return Err(kstest_invalid_argument(
            "kstest: CDF table must contain at least two distinct x values",
        ));
    }
    for window in normalized.windows(2) {
        if window[1].1 < window[0].1 {
            return Err(kstest_invalid_argument(
                "kstest: CDF table probabilities must be nondecreasing",
            ));
        }
    }
    Ok(CdfSpec::Table(normalized))
}
