use runmat_value::{IntValue, NumericScalar, Value};

use super::{fspecial_argument_error, fspecial_error, BuiltinResult, FilterKind};

pub(super) fn parse_filter_kind(value: &Value) -> BuiltinResult<FilterKind> {
    let text = value_to_string(value)?
        .ok_or_else(|| fspecial_argument_error("first argument must be a string filter name"))?;
    let lower = text.to_ascii_lowercase();
    FilterKind::from_name(&text).ok_or_else(|| {
        fspecial_argument_error(format!("fspecial: filter type '{lower}' is not supported"))
    })
}

pub(super) fn ensure_arg_count(
    name: &str,
    args: crate::call::FunctionArgumentView<'_>,
    min: usize,
    max: usize,
) -> BuiltinResult<()> {
    if args.len() < min || args.len() > max {
        if min == max {
            Err(fspecial_argument_error(format!(
                "fspecial: '{name}' expects exactly {min} argument{}",
                if min == 1 { "" } else { "s" }
            )))
        } else {
            Err(fspecial_argument_error(format!(
                "fspecial: '{name}' expects between {min} and {max} arguments"
            )))
        }
    } else {
        Ok(())
    }
}

pub(super) fn parse_average_dims(arg: Option<&Value>) -> BuiltinResult<(usize, usize)> {
    match arg {
        None => Ok((3, 3)),
        Some(value) => {
            let dims = parse_lengths_strict(value, "fspecial: LENGTHS must be positive integers")?;
            match dims.len() {
                1 => Ok((dims[0], dims[0])),
                2 => Ok((dims[0], dims[1])),
                _ => Err(fspecial_error(
                    "fspecial: LENGTHS must be a scalar or two-element vector",
                )),
            }
        }
    }
}

pub(super) fn parse_disk_params(arg: Option<&Value>) -> BuiltinResult<(f64, usize)> {
    let radius = match arg {
        None => 5.0,
        Some(value) => to_positive_scalar(value, "fspecial: RADIUS must be a non-negative scalar")?,
    };
    if radius < 0.0 {
        return Err(fspecial_error("fspecial: RADIUS must be non-negative"));
    }
    let extent_raw = radius.ceil();
    if extent_raw > isize::MAX as f64 {
        return Err(fspecial_error("fspecial: RADIUS is too large"));
    }
    let extent = extent_raw as isize;
    let size = extent
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| fspecial_error("fspecial: RADIUS is too large"))?;
    Ok((radius, size))
}

pub(super) fn parse_gaussian_params(
    lengths: Option<&Value>,
    sigma_value: Option<&Value>,
) -> BuiltinResult<(usize, usize, f64)> {
    let dims = match lengths {
        None => vec![3, 3],
        Some(value) => parse_lengths_strict(
            value,
            "fspecial: LENGTHS must be positive integers for gaussian",
        )?,
    };
    let dims = match dims.len() {
        1 => vec![dims[0], dims[0]],
        2 => dims,
        _ => {
            return Err(fspecial_error(
                "fspecial: gaussian lengths must be a scalar or a two-element vector",
            ));
        }
    };
    let sigma = match sigma_value {
        None => 0.5,
        Some(value) => {
            let sigma = to_positive_scalar(value, "fspecial: SIGMA must be a positive scalar")?;
            if sigma <= 0.0 {
                return Err(fspecial_error("fspecial: SIGMA must be positive"));
            }
            sigma
        }
    };
    Ok((dims[0], dims[1], sigma))
}

pub(super) fn parse_laplacian_alpha(arg: Option<&Value>) -> BuiltinResult<f64> {
    match arg {
        None => Ok(0.2),
        Some(value) => {
            let alpha = to_scalar(value, "fspecial: ALPHA must be a scalar")?;
            if !(0.0..=1.0).contains(&alpha) {
                return Err(fspecial_error("fspecial: ALPHA must be between 0 and 1"));
            }
            Ok(alpha)
        }
    }
}

pub(super) fn parse_log_params(
    lengths: Option<&Value>,
    sigma_value: Option<&Value>,
) -> BuiltinResult<(usize, usize, f64)> {
    let dims = match lengths {
        None => vec![5, 5],
        Some(value) => {
            parse_lengths_strict(value, "fspecial: LENGTHS must be positive integers for log")?
        }
    };
    let dims = match dims.len() {
        1 => vec![dims[0], dims[0]],
        2 => dims,
        _ => {
            return Err(fspecial_error(
                "fspecial: log lengths must be a scalar or two-element vector",
            ));
        }
    };
    let sigma = match sigma_value {
        None => 0.5,
        Some(value) => {
            let sigma = to_positive_scalar(value, "fspecial: SIGMA must be a positive scalar")?;
            if sigma <= 0.0 {
                return Err(fspecial_error("fspecial: SIGMA must be positive"));
            }
            sigma
        }
    };
    Ok((dims[0], dims[1], sigma))
}

pub(super) fn parse_motion_params(
    length_value: Option<&Value>,
    angle_value: Option<&Value>,
) -> BuiltinResult<(usize, usize, f64, usize)> {
    let length = parse_motion_length(length_value)?;
    let kernel_size = if length % 2 == 1 {
        length
    } else {
        length
            .checked_add(1)
            .ok_or_else(|| fspecial_error("fspecial: LENGTH is too large"))?
    };
    let angle_deg = match angle_value {
        None => 0.0,
        Some(value) => to_scalar(value, "fspecial: ANGLE must be a scalar")?,
    };
    Ok((length, kernel_size, angle_deg, 8))
}

pub(super) fn parse_motion_length(value: Option<&Value>) -> BuiltinResult<usize> {
    let Some(value) = value else {
        return Ok(9);
    };
    if let Some(length) = integer_scalar_dimension(value)? {
        if length == 0 {
            return Err(fspecial_error("fspecial: LENGTH must be positive"));
        }
        return Ok(length);
    }

    let len = to_positive_scalar(value, "fspecial: LENGTH must be a positive scalar")?;
    if len <= 0.0 {
        return Err(fspecial_error("fspecial: LENGTH must be positive"));
    }
    let rounded = len.round();
    if !fits_platform_usize(rounded) {
        return Err(fspecial_error("fspecial: LENGTH is too large"));
    }
    let length = rounded as usize;
    if length == 0 {
        return Err(fspecial_error("fspecial: LENGTH must be at least 1"));
    }
    Ok(length)
}

pub(super) fn parse_unsharp_alpha(arg: Option<&Value>) -> BuiltinResult<f64> {
    match arg {
        None => Ok(0.2),
        Some(value) => {
            let alpha = to_scalar(value, "fspecial: ALPHA must be a scalar")?;
            if !(0.0..=1.0).contains(&alpha) {
                return Err(fspecial_error("fspecial: ALPHA must be between 0 and 1"));
            }
            Ok(alpha)
        }
    }
}

fn value_to_string(value: &Value) -> crate::BuiltinResult<Option<String>> {
    Ok(match value {
        Value::String(runmat_value::StringElement::Text(s)) => Some(s.as_str().to_owned()),
        Value::StringArray(sa) if sa.data().len() == 1 => sa.data()[0].as_str().map(str::to_owned),
        Value::CharArray(ca) if ca.rows == 1 => Some(
            runmat_value::decode_utf16_code_units(ca.data()).map_err(|source| {
                let mut error = fspecial_error(source.to_string());
                error.source = Some(Box::new(source));
                error
            })?,
        ),
        _ => None,
    })
}

pub(super) fn parse_lengths_strict(value: &Value, err: &str) -> BuiltinResult<Vec<usize>> {
    parse_lengths_inner(value, err, true)
}

fn parse_lengths_inner(
    value: &Value,
    err: &str,
    enforce_positive: bool,
) -> BuiltinResult<Vec<usize>> {
    match value {
        Value::Int(i) => {
            let len = parse_integer_dimension(i)?;
            if enforce_positive && len == 0 {
                return Err(fspecial_error(err));
            }
            Ok(vec![len])
        }
        Value::Bool(value) => {
            parse_numeric_dimension(if *value { 1.0 } else { 0.0 }).map(|dimension| vec![dimension])
        }
        Value::Num(n) => parse_numeric_dimension(*n).map(|d| vec![d]),
        Value::Tensor(tensor) => {
            let dims = (0..tensor.len())
                .map(|index| {
                    let value = tensor
                        .numeric_value_at(index)
                        .ok_or_else(|| fspecial_error(err))?;
                    if let Some(value) = value.into_int_value() {
                        parse_integer_dimension(&value)
                    } else {
                        parse_numeric_dimension(value.materialize_f64())
                    }
                })
                .collect::<Result<Vec<_>, _>>()?;
            if enforce_positive && dims.contains(&0) {
                return Err(fspecial_error(err));
            }
            Ok(dims)
        }
        Value::LogicalArray(logical) => {
            if logical.data.len() != logical.shape.iter().product::<usize>() {
                return Err(fspecial_error(err));
            }
            let dims = logical
                .data
                .iter()
                .map(|&v| parse_numeric_dimension(v as f64))
                .collect::<Result<Vec<_>, _>>()?;
            if enforce_positive && dims.contains(&0) {
                return Err(fspecial_error(err));
            }
            Ok(dims)
        }
        _ => Err(fspecial_error(err)),
    }
}

fn parse_integer_dimension(value: &IntValue) -> BuiltinResult<usize> {
    value
        .try_to_usize()
        .ok_or_else(|| fspecial_error("fspecial: dimensions must be non-negative"))
}

pub(super) fn parse_numeric_dimension(n: f64) -> BuiltinResult<usize> {
    if !n.is_finite() {
        return Err(fspecial_error("fspecial: dimensions must be finite"));
    }
    if n < 0.0 {
        return Err(fspecial_error("fspecial: dimensions must be non-negative"));
    }
    let rounded = n.round();
    if (rounded - n).abs() > f64::EPSILON {
        return Err(fspecial_error("fspecial: dimensions must be integers"));
    }
    if !fits_platform_usize(rounded) {
        return Err(fspecial_error(
            "fspecial: dimensions are outside the supported platform range",
        ));
    }
    Ok(rounded as usize)
}

fn integer_scalar_dimension(value: &Value) -> BuiltinResult<Option<usize>> {
    match value {
        Value::Int(value) => parse_integer_dimension(value).map(Some),
        Value::Tensor(tensor) => {
            let Some(storage) = tensor.integer_storage() else {
                return Ok(None);
            };
            if storage.len() != 1 {
                return Ok(None);
            }
            let value = storage
                .value_at(0)
                .ok_or_else(|| fspecial_error("fspecial: dimensions must be scalar"))?;
            parse_integer_dimension(&value).map(Some)
        }
        _ => Ok(None),
    }
}

fn fits_platform_usize(value: f64) -> bool {
    value < usize::MAX as f64 || (usize::BITS < 64 && value == usize::MAX as f64)
}

fn to_scalar(value: &Value, err: &str) -> BuiltinResult<f64> {
    match value {
        Value::Num(n) => Ok(*n),
        Value::Int(integer) => exact_integer_f64(integer, err),
        Value::Bool(value) => Ok(if *value { 1.0 } else { 0.0 }),
        Value::LogicalArray(logical) if logical.data.len() == 1 => {
            Ok(if logical.data[0] == 0 { 0.0 } else { 1.0 })
        }
        Value::Tensor(tensor) if tensor.len() == 1 => {
            let scalar = tensor
                .numeric_value_at(0)
                .ok_or_else(|| fspecial_error(err))?;
            numeric_scalar_f64(scalar, err)
        }
        _ => Err(fspecial_error(err)),
    }
}

fn numeric_scalar_f64(value: NumericScalar, err: &str) -> BuiltinResult<f64> {
    if let Some(integer) = value.into_int_value() {
        exact_integer_f64(&integer, err)
    } else {
        Ok(value.materialize_f64())
    }
}

fn exact_integer_f64(value: &IntValue, err: &str) -> BuiltinResult<f64> {
    if crate::builtins::common::validation::integer_is_exact_f64(value) {
        Ok(value.to_f64())
    } else {
        Err(fspecial_error(format!(
            "{err}; integer value must be exactly representable as double"
        )))
    }
}

fn to_positive_scalar(value: &Value, err: &str) -> BuiltinResult<f64> {
    let scalar = to_scalar(value, err)?;
    if scalar.is_nan() || scalar.is_infinite() {
        return Err(fspecial_error(err));
    }
    Ok(scalar)
}
