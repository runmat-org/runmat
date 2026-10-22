use super::*;
use crate::builtins::common::tensor;

#[derive(Clone, Copy)]
pub(super) enum ControlRole {
    FileId,
    Size,
    Skip,
}

pub(super) fn preflight_control(value: &Value, role: ControlRole) -> BuiltinResult<()> {
    let integer_extension = match role {
        ControlRole::FileId => &FREAD_INTEGER_ID_EXTENSION,
        ControlRole::Size => &FREAD_INTEGER_SIZE_EXTENSION,
        ControlRole::Skip => &FREAD_INTEGER_SKIP_EXTENSION,
    };
    match value {
        Value::Int(_) => {
            crate::compatibility::ensure_builtin_extension_enabled(integer_extension, BUILTIN_NAME)
        }
        Value::Tensor(tensor) if tensor.integer_storage().is_some() => {
            crate::compatibility::ensure_builtin_extension_enabled(integer_extension, BUILTIN_NAME)
        }
        Value::Bool(_) | Value::LogicalArray(_) => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FREAD_LOGICAL_CONTROL_EXTENSION,
                BUILTIN_NAME,
            )
        }
        Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F32 => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FREAD_SINGLE_CONTROL_EXTENSION,
                BUILTIN_NAME,
            )
        }
        Value::GpuTensor(handle) => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &FREAD_RESIDENT_CONTROL_EXTENSION,
                BUILTIN_NAME,
            )?;
            let element_count = handle
                .shape()
                .iter()
                .try_fold(1usize, |count, extent| count.checked_mul(*extent))
                .ok_or_else(|| {
                    fread_error_with_detail(
                        &FREAD_ERROR_INVALID_INPUT,
                        "resident control shape exceeds platform limits",
                    )
                })?;
            let shape_valid = match role {
                ControlRole::FileId | ControlRole::Skip => element_count == 1,
                ControlRole::Size => element_count <= 2,
            };
            if !shape_valid {
                let error = match role {
                    ControlRole::Skip => &FREAD_ERROR_INVALID_OPTION,
                    ControlRole::FileId | ControlRole::Size => &FREAD_ERROR_INVALID_INPUT,
                };
                return Err(fread_error_with_detail(
                    error,
                    "resident control has invalid element count",
                ));
            }
            if runmat_accelerate_api::handle_is_logical(handle) {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &FREAD_LOGICAL_CONTROL_EXTENSION,
                    BUILTIN_NAME,
                )?;
            } else if runmat_accelerate_api::handle_integer_type(handle).is_some() {
                crate::compatibility::ensure_builtin_extension_enabled(
                    integer_extension,
                    BUILTIN_NAME,
                )?;
            } else if runmat_accelerate_api::handle_precision(handle)
                == Some(runmat_accelerate_api::ProviderPrecision::F32)
            {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &FREAD_SINGLE_CONTROL_EXTENSION,
                    BUILTIN_NAME,
                )?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

pub(super) fn parse_fid(value: &Value) -> Result<i32, String> {
    let number = match value {
        Value::Num(n) => *n,
        Value::Int(int) => {
            return int
                .try_to_i32()
                .ok_or_else(|| "file identifier is out of range".to_string());
        }
        Value::Tensor(t) if tensor::is_scalar_tensor(t) => {
            if let Some(int) = t.integer_storage().and_then(|storage| storage.value_at(0)) {
                return int
                    .try_to_i32()
                    .ok_or_else(|| "file identifier is out of range".to_string());
            }
            tensor::tensor_value_f64(t, 0)
        }
        _ => {
            return Err("file identifier must be numeric".to_string());
        }
    };
    if !number.is_finite() {
        return Err("file identifier must be finite".to_string());
    }
    let rounded = number.round();
    if (rounded - number).abs() > f64::EPSILON {
        return Err("file identifier must be an integer".to_string());
    }
    if rounded < i32::MIN as f64 || rounded > i32::MAX as f64 {
        return Err("file identifier is out of range".to_string());
    }
    Ok(rounded as i32)
}

pub(super) type ClassifiedArgs<'a> = (
    Option<&'a Value>,
    Option<&'a Value>,
    Option<&'a Value>,
    Option<&'a Value>,
    Option<&'a Value>,
);

pub(super) fn classify_arguments<'a>(
    args: crate::call::FunctionArgumentView<'a>,
) -> Result<ClassifiedArgs<'a>, String> {
    let mut filtered_indices = [0usize; 4];
    let mut filtered_count = 0usize;
    let mut like_proto: Option<&Value> = None;
    let mut i = 0usize;

    while i < args.len() {
        let value = &args[i];
        if matches_keyword(value, "like") {
            if like_proto.is_some() {
                return Err("multiple 'like' prototypes are not supported".to_string());
            }
            i += 1;
            let Some(proto_value) = args.get(i) else {
                return Err("expected prototype after 'like'".to_string());
            };
            like_proto = Some(proto_value);
            i += 1;
            continue;
        }
        if filtered_count < filtered_indices.len() {
            filtered_indices[filtered_count] = i;
        }
        filtered_count += 1;
        i += 1;
    }

    if filtered_count > filtered_indices.len() {
        return Err("too many input arguments".to_string());
    }

    let (size_idx, precision_idx, skip_idx, machine_idx) =
        classify_ordered_indices(args, &filtered_indices[..filtered_count])?;

    let size = size_idx.map(|index| args.get(index).expect("classified fread control index"));
    let precision =
        precision_idx.map(|index| args.get(index).expect("classified fread control index"));
    let skip = skip_idx.map(|index| args.get(index).expect("classified fread control index"));
    let machine = machine_idx.map(|index| args.get(index).expect("classified fread control index"));

    Ok((size, precision, skip, machine, like_proto))
}

pub(super) type ClassifiedIndices = (Option<usize>, Option<usize>, Option<usize>, Option<usize>);

pub(super) fn classify_ordered_indices(
    args: crate::call::FunctionArgumentView<'_>,
    indices: &[usize],
) -> Result<ClassifiedIndices, String> {
    let mut position = 0usize;
    let mut size_idx: Option<usize> = None;
    let mut precision_idx: Option<usize> = None;
    let mut skip_idx: Option<usize> = None;
    let mut machine_idx: Option<usize> = None;

    if let Some(&first_index) = indices.get(position) {
        let first = &args[first_index];
        if is_string_like(first) {
            precision_idx = Some(first_index);
        } else {
            size_idx = Some(first_index);
        }
        position += 1;
    }

    if let Some(&index) = indices.get(position) {
        let candidate = &args[index];
        if precision_idx.is_none() && is_string_like(candidate) {
            precision_idx = Some(index);
            position += 1;
        } else if is_numeric_like(candidate) {
            skip_idx = Some(index);
            position += 1;
        } else if is_string_like(candidate) {
            machine_idx = Some(index);
            position += 1;
        } else {
            return Err("invalid argument combination".to_string());
        }
    }

    if let Some(&index) = indices.get(position) {
        let candidate = &args[index];
        if skip_idx.is_none() && is_numeric_like(candidate) {
            skip_idx = Some(index);
            position += 1;
        } else if machine_idx.is_none() && is_string_like(candidate) {
            machine_idx = Some(index);
            position += 1;
        } else {
            return Err("invalid argument combination".to_string());
        }
    }

    if let Some(&index) = indices.get(position) {
        let candidate = &args[index];
        if machine_idx.is_none() && is_string_like(candidate) {
            machine_idx = Some(index);
            position += 1;
        } else {
            return Err("too many input arguments".to_string());
        }
    }

    if position < indices.len() {
        return Err("too many input arguments".to_string());
    }

    Ok((size_idx, precision_idx, skip_idx, machine_idx))
}

pub(super) fn is_string_like(value: &Value) -> bool {
    crate::builtins::common::validation::value_is_text_scalar(value)
}

pub(super) fn matches_keyword(value: &Value, keyword: &str) -> bool {
    extract_scalar_string(value)
        .map(|text| text.eq_ignore_ascii_case(keyword))
        .unwrap_or(false)
}

pub(super) fn is_numeric_like(value: &Value) -> bool {
    matches!(
        value,
        Value::Num(_)
            | Value::Int(_)
            | Value::Bool(_)
            | Value::Tensor(_)
            | Value::LogicalArray(_)
            | Value::GpuTensor(_)
    )
}

#[derive(Clone, Debug)]
pub(super) enum SizeSpec {
    All,
    Count(usize),
    Matrix { rows: usize, cols: Option<usize> },
}

impl SizeSpec {
    pub(super) fn element_limit(&self) -> Option<usize> {
        match self {
            SizeSpec::All => None,
            SizeSpec::Count(n) => Some(*n),
            SizeSpec::Matrix { rows, cols } => {
                if *rows == 0 {
                    return Some(0);
                }
                cols.map(|c| rows.saturating_mul(c))
            }
        }
    }
}

pub(super) fn parse_size(arg: Option<&Value>) -> Result<SizeSpec, String> {
    match arg {
        None => Ok(SizeSpec::All),
        Some(value @ (Value::String(_) | Value::CharArray(_) | Value::StringArray(_))) => {
            let text = extract_scalar_string(value)
                .ok_or_else(|| "size argument must be numeric or a size vector".to_string())?;
            parse_size_string(&text)
        }
        Some(Value::Tensor(t)) => parse_size_tensor(t),
        Some(Value::Int(int)) => Ok(SizeSpec::Count(int_to_usize(
            int,
            "size argument must be a non-negative integer",
        )?)),
        Some(value) => {
            let scalar = value_to_scalar(value, "size argument must be numeric or a size vector")?;
            scalar_to_size(scalar)
        }
    }
}

pub(super) fn parse_size_string(text: &str) -> Result<SizeSpec, String> {
    if text.trim().is_empty() {
        return Err("size argument must not be empty".to_string());
    }
    let lower = text.trim().to_ascii_lowercase();
    if lower == "inf" {
        Ok(SizeSpec::All)
    } else {
        let number = lower
            .parse::<f64>()
            .map_err(|_| "size argument must be numeric or 'inf'".to_string())?;
        scalar_to_size(number)
    }
}

pub(super) fn parse_size_tensor(t: &Tensor) -> Result<SizeSpec, String> {
    if let Some(storage) = t.integer_storage() {
        return match storage.len() {
            0 => Ok(SizeSpec::Count(0)),
            1 => {
                let value = storage.value_at(0).expect("one-element integer storage");
                Ok(SizeSpec::Count(value.try_to_usize().ok_or_else(|| {
                    "size argument must be a non-negative integer".to_string()
                })?))
            }
            2 => {
                let rows_value = storage
                    .value_at(0)
                    .expect("integer storage length matches tensor length");
                let cols_value = storage
                    .value_at(1)
                    .expect("integer storage length matches tensor length");
                let rows = rows_value.try_to_usize().ok_or_else(|| {
                    "size vector components must be non-negative integers or Inf".to_string()
                })?;
                let cols = cols_value.try_to_usize().ok_or_else(|| {
                    "size vector components must be non-negative integers or Inf".to_string()
                })?;
                Ok(SizeSpec::Matrix {
                    rows,
                    cols: Some(cols),
                })
            }
            _ => Err("size vector must contain at most two elements".to_string()),
        };
    }

    match t.len() {
        0 => Ok(SizeSpec::Count(0)),
        1 => scalar_to_size(tensor::tensor_value_f64(t, 0)),
        2 => {
            let rows = scalar_to_size_component(
                tensor::tensor_value_f64(t, 0),
                "size vector components must be non-negative integers or Inf",
            )?;
            let cols_raw = tensor::tensor_value_f64(t, 1);
            if cols_raw.is_infinite() && cols_raw.is_sign_positive() {
                Ok(SizeSpec::Matrix { rows, cols: None })
            } else {
                let cols = scalar_to_size_component(
                    cols_raw,
                    "size vector components must be non-negative integers or Inf",
                )?;
                Ok(SizeSpec::Matrix {
                    rows,
                    cols: Some(cols),
                })
            }
        }
        _ => Err("size vector must contain at most two elements".to_string()),
    }
}

pub(super) fn scalar_to_size(value: f64) -> Result<SizeSpec, String> {
    if value.is_infinite() && value.is_sign_positive() {
        return Ok(SizeSpec::All);
    }
    let count = scalar_to_size_component(value, "size argument must be a non-negative integer")?;
    Ok(SizeSpec::Count(count))
}

pub(super) fn scalar_to_size_component(value: f64, err: &str) -> Result<usize, String> {
    if !value.is_finite() {
        return Err(err.to_string());
    }
    if value < 0.0 {
        return Err(err.to_string());
    }
    let rounded = value.round();
    if (rounded - value).abs() > f64::EPSILON {
        return Err(err.to_string());
    }
    if rounded > usize::MAX as f64 || (usize::BITS == 64 && rounded == usize::MAX as f64) {
        return Err("size argument is too large".to_string());
    }
    Ok(rounded as usize)
}
