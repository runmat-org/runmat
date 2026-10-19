//! Typed grayscale/indexed argument and control admission.
use super::*;

pub(super) struct ParsedCall {
    pub(super) image: Value,
    pub(super) mode: ImhistMode,
}

pub(super) enum ImhistMode {
    Grayscale { bins: Option<usize> },
    Indexed { bins: usize },
}

pub(super) fn parse_call(image: Value, rest: &[Value]) -> BuiltinResult<ParsedCall> {
    match rest {
        [] => Ok(ParsedCall {
            image,
            mode: ImhistMode::Grayscale { bins: None },
        }),
        [second] => {
            if let Some(bins) = parse_optional_bin_count(second)? {
                Ok(ParsedCall {
                    image,
                    mode: ImhistMode::Grayscale { bins: Some(bins) },
                })
            } else {
                let bins = parse_colormap_bins(second)?;
                Ok(ParsedCall {
                    image,
                    mode: ImhistMode::Indexed { bins },
                })
            }
        }
        _ => Err(invalid(
            "expected imhist(I), imhist(I, n), or imhist(X, map)",
        )),
    }
}

pub(super) fn parse_optional_bin_count(value: &Value) -> BuiltinResult<Option<usize>> {
    if is_typed_integer_scalar(value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &IMHIST_TYPED_BIN_COUNT_EXTENSION,
            NAME,
        )?;
    }
    if let Some(bins) = integer_scalar_bin_count(value)? {
        validate_bin_count(bins)?;
        return Ok(Some(bins));
    }
    let Some(raw) = scalar_number(value) else {
        return Ok(None);
    };
    if !raw.is_finite() || raw < 1.0 || (raw.round() - raw).abs() > INTEGER_TOL {
        return Err(invalid("bin count must be a positive integer scalar"));
    }
    let rounded = raw.round();
    if !fits_platform_usize(rounded) {
        return Err(invalid("bin count is outside the supported platform range"));
    }
    if rounded > MAX_BINS as f64 {
        return Err(invalid(format!(
            "bin count {:.0} exceeds maximum supported bin count {MAX_BINS}",
            rounded
        )));
    }
    let bins = rounded as usize;
    validate_bin_count(bins)?;
    Ok(Some(bins))
}

pub(super) fn is_typed_integer_scalar(value: &Value) -> bool {
    match value {
        Value::Int(_) => true,
        Value::Tensor(tensor) if tensor_utils::is_scalar_tensor(tensor) => {
            tensor.integer_storage().is_some()
        }
        Value::GpuTensor(handle) => {
            runmat_accelerate_api::handle_integer_type(handle).is_some()
                && handle
                    .shape()
                    .iter()
                    .try_fold(1usize, |n, &dim| n.checked_mul(dim))
                    == Some(1)
        }
        _ => false,
    }
}

fn parse_colormap_bins(value: &Value) -> BuiltinResult<usize> {
    let tensor = Tensor::try_from(value)
        .map_err(|err| invalid(format!("colormap must be an Nx3 numeric array: {err}")))?;
    if tensor.shape().len() != 2 || tensor.cols() != 3 || tensor.rows() == 0 {
        return Err(invalid("colormap must be a non-empty Nx3 numeric array"));
    }
    if tensor.numeric_dtype() != runmat_value::NumericDType::F64 {
        return Err(invalid("colormap must be a double Nx3 numeric array"));
    }
    let values = tensor_utils::tensor_values_f64_cow(&tensor);
    if !values.iter().all(|value| value.is_finite()) {
        return Err(invalid("colormap values must be finite"));
    }
    if !values.iter().all(|value| (0.0..=1.0).contains(value)) {
        return Err(invalid("colormap values must be in the range [0, 1]"));
    }
    validate_bin_count(tensor.rows())?;
    Ok(tensor.rows())
}

pub(super) fn validate_bin_count(bins: usize) -> BuiltinResult<()> {
    if bins == 0 {
        return Err(invalid("bin count must be positive"));
    }
    if bins > MAX_BINS {
        return Err(invalid(format!(
            "bin count {bins} exceeds maximum supported bin count {MAX_BINS}"
        )));
    }
    Ok(())
}

pub(super) fn scalar_number(value: &Value) -> Option<f64> {
    match value {
        Value::Num(value) => Some(*value),
        Value::Int(value) => Some(value.to_f64()),
        Value::Bool(value) => Some(if *value { 1.0 } else { 0.0 }),
        Value::Tensor(tensor) if tensor_utils::is_scalar_tensor(tensor) => {
            Some(tensor_utils::tensor_value_f64(tensor, 0))
        }
        _ => None,
    }
}

fn integer_scalar_bin_count(value: &Value) -> BuiltinResult<Option<usize>> {
    let integer = match value {
        Value::Int(value) => Some(value.clone()),
        Value::Tensor(tensor) if tensor_utils::is_scalar_tensor(tensor) => tensor
            .integer_storage()
            .and_then(|storage| storage.value_at(0)),
        _ => None,
    };
    let Some(integer) = integer else {
        return Ok(None);
    };
    integer
        .try_to_usize()
        .ok_or_else(|| invalid("bin count is outside the supported platform range"))
        .map(Some)
}

fn fits_platform_usize(value: f64) -> bool {
    value < usize::MAX as f64 || (usize::BITS < 64 && value == usize::MAX as f64)
}
