use super::filling::numeric::{summary_value, Summary};
use super::numeric::{first_nonsingleton_dim, scalar_text, scalar_usize, validate_matrix_dim};
use super::*;

#[derive(Clone, Copy)]
pub(super) struct MovingOptions {
    dim: Option<usize>,
    omit_nan: bool,
}

impl MovingOptions {
    pub(super) fn parse(args: &[Value]) -> BuiltinResult<Self> {
        let mut dim = None;
        let mut omit_nan = false;
        let mut idx = 0;
        while idx < args.len() {
            if let Some(text) = scalar_text(&args[idx]) {
                match text.to_ascii_lowercase().as_str() {
                    "omitnan" | "omitmissing" => omit_nan = true,
                    "includenan" | "includemissing" => omit_nan = false,
                    "dim" if idx + 1 < args.len() => {
                        dim = Some(scalar_usize(&args[idx + 1], "movmad dimension")?);
                        idx += 1;
                    }
                    "dim" => return Err(invalid_argument("movmad: 'dim' requires a value")),
                    other => {
                        return Err(invalid_argument(format!(
                            "movmad: unsupported option '{other}'"
                        )))
                    }
                }
            } else if matches!(args[idx], Value::Num(_) | Value::Int(_)) {
                dim = Some(scalar_usize(&args[idx], "movmad dimension")?);
            } else {
                return Err(invalid_argument(format!(
                    "movmad: unsupported option argument {:?}",
                    args[idx]
                )));
            }
            idx += 1;
        }
        if dim.is_some_and(|dim| dim != 1 && dim != 2) {
            return Err(invalid_argument("movmad: dimension must be 1 or 2"));
        }
        Ok(Self { dim, omit_nan })
    }
}

pub(super) fn moving_mad(
    tensor: Tensor,
    window: usize,
    options: MovingOptions,
) -> BuiltinResult<Value> {
    if window == 0 {
        return Err(invalid_argument("movmad: window length must be positive"));
    }
    let rows = tensor.rows();
    let cols = tensor.cols();
    let dim = options
        .dim
        .unwrap_or_else(|| first_nonsingleton_dim(rows, cols));
    validate_matrix_dim(dim, "movmad")?;
    let shape = tensor.shape.clone();
    let output_dtype = if tensor.integer_storage().is_some() {
        NumericDType::F64
    } else {
        tensor.numeric_dtype()
    };
    let values = tensor_utils::tensor_values_f64_cow(&tensor);
    let mut out = vec![f64::NAN; values.len()];
    if dim == 1 {
        for col in 0..cols {
            for row in 0..rows {
                out[row + col * rows] = moving_mad_at(
                    &values,
                    rows,
                    col * rows,
                    row,
                    rows,
                    1,
                    window,
                    options.omit_nan,
                );
            }
        }
    } else {
        for row in 0..rows {
            for col in 0..cols {
                out[row + col * rows] = moving_mad_at(
                    &values,
                    rows,
                    row,
                    col,
                    cols,
                    rows,
                    window,
                    options.omit_nan,
                );
            }
        }
    }
    Tensor::new_with_dtype(out, shape, output_dtype)
        .map(Value::Tensor)
        .map_err(internal_error)
}

pub(super) fn moving_mad_at(
    data: &[f64],
    _rows: usize,
    start: usize,
    pos: usize,
    len: usize,
    step: usize,
    window: usize,
    omit_nan: bool,
) -> f64 {
    let before = (window - 1) / 2;
    let after = window / 2;
    let lo = pos.saturating_sub(before);
    let hi = pos.saturating_add(after).saturating_add(1).min(len);
    let mut vals = Vec::new();
    for idx in lo..hi {
        let value = data[start + idx * step];
        if value.is_nan() && omit_nan {
            continue;
        }
        vals.push(value);
    }
    if vals.is_empty() || vals.iter().any(|value| value.is_nan()) {
        return f64::NAN;
    }
    let med = summary_value(vals.clone(), &Summary::Median);
    let mut devs: Vec<f64> = vals.into_iter().map(|value| (value - med).abs()).collect();
    summary_value(::std::mem::take(&mut devs), &Summary::Median)
}
