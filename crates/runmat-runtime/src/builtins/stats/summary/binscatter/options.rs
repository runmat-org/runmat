//! Typed chart option evaluation and exact structural controls.
use super::input::ensure_vector_tensor;
use super::*;
use runmat_accelerate_api::GpuTensorStorage;

pub(super) struct BinscatterOptionPlan {
    steps: Vec<PlannedOption>,
}

enum PlannedOption {
    Host(DecodedOption),
    Resident(BinscatterOptionKind, Value),
}

enum DecodedOption {
    NumBins([usize; 2]),
    XLimits(Tensor),
    YLimits(Tensor),
    ShowEmptyBins(bool),
    FaceAlpha(f64),
    DisplayName(String),
    DisplayStyle,
}

impl BinscatterOptionPlan {
    pub(super) fn admit(args: crate::call::BuiltinArguments) -> BuiltinResult<Self> {
        let mut steps = Vec::new();
        let mut values = args.into_iter().peekable();
        if values.peek().is_some_and(|first| !is_option_name(first)) {
            let value = values.next().expect("peeked positional NumBins");
            steps.push(plan_step(BinscatterOptionKind::NumBins, value)?);
        }
        while let Some(name) = values.next() {
            let key = tensor::value_to_string(&name)
                .ok_or_else(|| invalid("binscatter: expected name-value option"))?;
            let Some(value) = values.next() else {
                return Err(invalid(format!(
                    "binscatter: missing value for option '{key}'"
                )));
            };
            let kind = match BinscatterOptionKind::parse(&key) {
                Some(BinscatterOptionKind::UnsupportedMatlabOption) => {
                    return Err(invalid(format!(
                        "binscatter: option '{key}' is not supported by MATLAB binscatter"
                    )));
                }
                Some(kind) => kind,
                None => {
                    return Err(invalid(format!(
                        "binscatter: unrecognised option '{}'",
                        key.trim().to_ascii_lowercase()
                    )))
                }
            };
            steps.push(plan_step(kind, value)?);
        }
        Ok(Self { steps })
    }

    pub(super) async fn evaluate(self) -> BuiltinResult<BinscatterOptions> {
        let mut options = BinscatterOptions {
            show_empty_bins: false,
            face_alpha: DEFAULT_FACE_ALPHA,
            display_name: None,
            num_bins: None,
            x_limits: None,
            y_limits: None,
        };
        for step in self.steps {
            let decoded = match step {
                PlannedOption::Host(decoded) => decoded,
                PlannedOption::Resident(kind, value) => {
                    let gathered = gather_if_needed_async(&value)
                        .await
                        .map_err(|err| invalid_with_source(format!("binscatter: {err}"), err))?;
                    decode_option(kind, &gathered)?
                }
            };
            match decoded {
                DecodedOption::NumBins(value) => options.num_bins = Some(value),
                DecodedOption::XLimits(value) => options.x_limits = Some(value),
                DecodedOption::YLimits(value) => options.y_limits = Some(value),
                DecodedOption::ShowEmptyBins(value) => options.show_empty_bins = value,
                DecodedOption::FaceAlpha(value) => options.face_alpha = value,
                DecodedOption::DisplayName(value) => options.display_name = Some(value),
                DecodedOption::DisplayStyle => {}
            }
        }
        Ok(options)
    }
}

fn plan_step(kind: BinscatterOptionKind, value: Value) -> BuiltinResult<PlannedOption> {
    if let Value::GpuTensor(handle) = &value {
        if matches!(
            kind,
            BinscatterOptionKind::DisplayName | BinscatterOptionKind::DisplayStyle
        ) {
            return Err(invalid("binscatter: text options must be host values"));
        }
        if handle.physical().storage() != GpuTensorStorage::Real {
            return Err(invalid("binscatter: numeric options must be real"));
        }
        let len = handle
            .shape()
            .iter()
            .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
            .ok_or_else(|| invalid("binscatter: option shape overflows"))?;
        let vector = handle.shape().iter().filter(|&&dim| dim > 1).count() <= 1;
        let geometry_valid = match kind {
            BinscatterOptionKind::NumBins => vector && (len == 1 || len == 2),
            BinscatterOptionKind::XLimits | BinscatterOptionKind::YLimits => vector && len == 2,
            BinscatterOptionKind::ShowEmptyBins | BinscatterOptionKind::FaceAlpha => len == 1,
            BinscatterOptionKind::DisplayName
            | BinscatterOptionKind::DisplayStyle
            | BinscatterOptionKind::UnsupportedMatlabOption => false,
        };
        if !geometry_valid {
            return Err(invalid("binscatter: option has invalid shape"));
        }
        return Ok(PlannedOption::Resident(kind, value));
    }
    Ok(PlannedOption::Host(decode_option(kind, &value)?))
}

fn decode_option(kind: BinscatterOptionKind, value: &Value) -> BuiltinResult<DecodedOption> {
    match kind {
        BinscatterOptionKind::NumBins => Ok(DecodedOption::NumBins(parse_num_bins(value)?)),
        BinscatterOptionKind::XLimits => {
            Ok(DecodedOption::XLimits(parse_limits(value, "XLimits")?))
        }
        BinscatterOptionKind::YLimits => {
            Ok(DecodedOption::YLimits(parse_limits(value, "YLimits")?))
        }
        BinscatterOptionKind::ShowEmptyBins => Ok(DecodedOption::ShowEmptyBins(option_bool(
            value,
            "ShowEmptyBins",
        )?)),
        BinscatterOptionKind::FaceAlpha => {
            let alpha = option_scalar(value, "FaceAlpha")?;
            validate_face_alpha(alpha)?;
            Ok(DecodedOption::FaceAlpha(alpha))
        }
        BinscatterOptionKind::DisplayName => Ok(DecodedOption::DisplayName(
            tensor::value_to_string(value)
                .ok_or_else(|| invalid("binscatter: DisplayName must be text"))?,
        )),
        BinscatterOptionKind::DisplayStyle => {
            let style = tensor::value_to_string(value)
                .ok_or_else(|| invalid("binscatter: DisplayStyle must be a string"))?;
            if !style.trim().eq_ignore_ascii_case("tile") {
                return Err(invalid("binscatter: only DisplayStyle 'tile' is supported"));
            }
            Ok(DecodedOption::DisplayStyle)
        }
        BinscatterOptionKind::UnsupportedMatlabOption => {
            Err(invalid("binscatter: unsupported option"))
        }
    }
}

fn is_option_name(value: &Value) -> bool {
    matches!(
        value,
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_)
    )
}

pub(crate) fn parse_num_bins(value: &Value) -> BuiltinResult<[usize; 2]> {
    let known_len = match value {
        Value::Tensor(tensor) => Some(tensor.len()),
        Value::LogicalArray(array) => Some(array.len()),
        _ => None,
    };
    if known_len.is_some_and(|len| len != 1 && len != 2) {
        return Err(invalid(
            "binscatter: NumBins must be a scalar or two-element vector",
        ));
    }
    if let Some(values) = integer_values(value) {
        let bins = match values.as_slice() {
            [n] => [positive_bin_count_integer(n, "NumBins")?; 2],
            [nx, ny] => [
                positive_bin_count_integer(nx, "NumBins")?,
                positive_bin_count_integer(ny, "NumBins")?,
            ],
            _ => {
                return Err(invalid(
                    "binscatter: NumBins must be a scalar or two-element vector",
                ))
            }
        };
        return Ok(bins);
    }
    let values = numeric_vector(value, "NumBins")?;
    let bins = match values.as_slice() {
        [n] => [positive_bin_count(*n, "NumBins")?; 2],
        [nx, ny] => [
            positive_bin_count(*nx, "NumBins")?,
            positive_bin_count(*ny, "NumBins")?,
        ],
        _ => {
            return Err(invalid(
                "binscatter: NumBins must be a scalar or two-element vector",
            ))
        }
    };
    Ok(bins)
}

pub(crate) fn parse_limits(value: &Value, name: &str) -> BuiltinResult<Tensor> {
    let tensor = tensor::value_to_tensor(value)
        .map_err(|_| invalid(format!("binscatter: {name} must be numeric")))?;
    ensure_vector_tensor(&tensor, name)?;
    if tensor.len() != 2 {
        return Err(invalid(format!(
            "binscatter: {name} must be a two-element vector"
        )));
    }
    let lo = tensor
        .numeric_value_at(0)
        .ok_or_else(|| invalid(format!("binscatter: {name} is empty")))?;
    let hi = tensor
        .numeric_value_at(1)
        .ok_or_else(|| invalid(format!("binscatter: {name} is empty")))?;
    if let (Some(lo), Some(hi)) = (integer_i128(lo), integer_i128(hi)) {
        if hi <= lo {
            return Err(invalid(format!(
                "binscatter: {name} must contain finite increasing limits"
            )));
        }
    } else {
        validate_limits(lo.materialize_f64(), hi.materialize_f64(), name)?;
    }
    Ok(tensor)
}

fn numeric_vector(value: &Value, name: &str) -> BuiltinResult<Vec<f64>> {
    if let Value::Tensor(tensor) = value {
        return Ok((0..tensor.len())
            .map(|index| {
                tensor
                    .numeric_value_at(index)
                    .expect("numeric tensor element remains present")
                    .materialize_f64()
            })
            .collect());
    }
    let tensor = tensor::value_to_tensor(value)
        .map_err(|_| invalid(format!("binscatter: {name} must be numeric")))?;
    Ok((0..tensor.len())
        .map(|index| {
            tensor
                .numeric_value_at(index)
                .expect("numeric tensor element remains present")
                .materialize_f64()
        })
        .collect())
}

fn positive_bin_count(value: f64, name: &str) -> BuiltinResult<usize> {
    if !value.is_finite() || value < 1.0 || value.fract() != 0.0 {
        return Err(invalid(format!(
            "binscatter: {name} entries must be positive integers"
        )));
    }
    if value > MAX_NUM_BINS as f64 {
        return Err(invalid(format!(
            "binscatter: {name} entries must be no greater than {MAX_NUM_BINS}"
        )));
    }
    let count = value as usize;
    if count > MAX_NUM_BINS {
        return Err(invalid(format!(
            "binscatter: {name} entries must be no greater than {MAX_NUM_BINS}"
        )));
    }
    Ok(count)
}

fn positive_bin_count_integer(value: &IntValue, name: &str) -> BuiltinResult<usize> {
    let Some(count) = value.try_to_usize() else {
        return Err(invalid(format!(
            "binscatter: {name} entries must be positive integers"
        )));
    };
    if count == 0 {
        return Err(invalid(format!(
            "binscatter: {name} entries must be positive integers"
        )));
    }
    if count > MAX_NUM_BINS {
        return Err(invalid(format!(
            "binscatter: {name} entries must be no greater than {MAX_NUM_BINS}"
        )));
    }
    Ok(count)
}

pub(super) fn validate_limits(lo: f64, hi: f64, name: &str) -> BuiltinResult<(f64, f64)> {
    if !lo.is_finite() || !hi.is_finite() || hi <= lo {
        return Err(invalid(format!(
            "binscatter: {name} must contain finite increasing limits"
        )));
    }
    Ok((lo, hi))
}

pub(crate) fn option_bool(value: &Value, name: &str) -> BuiltinResult<bool> {
    if let Some(value) = integer_scalar(value) {
        return integer_bool(&value, name);
    }
    match value {
        Value::Bool(value) => Ok(*value),
        Value::Num(value) => numeric_bool(*value, name),
        _ => {
            if let Some(text) = tensor::value_to_string(value) {
                match text.trim().to_ascii_lowercase().as_str() {
                    "on" | "true" | "1" => Ok(true),
                    "off" | "false" | "0" => Ok(false),
                    _ => Err(invalid(format!("binscatter: {name} must be logical"))),
                }
            } else {
                Err(invalid(format!("binscatter: {name} must be logical")))
            }
        }
    }
}

fn numeric_bool(value: f64, name: &str) -> BuiltinResult<bool> {
    if value == 0.0 {
        Ok(false)
    } else if value == 1.0 {
        Ok(true)
    } else {
        Err(invalid(format!("binscatter: {name} must be 0 or 1")))
    }
}

fn integer_bool(value: &IntValue, name: &str) -> BuiltinResult<bool> {
    match value.try_to_usize() {
        Some(0) => Ok(false),
        Some(1) => Ok(true),
        _ => Err(invalid(format!("binscatter: {name} must be 0 or 1"))),
    }
}

pub(super) fn option_scalar(value: &Value, name: &str) -> BuiltinResult<f64> {
    if let Some(value) = integer_scalar(value) {
        return Ok(value.to_f64());
    }
    match value {
        Value::Num(value) => Ok(*value),
        Value::Bool(value) => Ok(if *value { 1.0 } else { 0.0 }),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            Ok(tensor::tensor_value_f64(tensor, 0))
        }
        _ => Err(invalid(format!("binscatter: {name} must be a scalar"))),
    }
}

pub(super) fn integer_scalar(value: &Value) -> Option<IntValue> {
    match value {
        Value::Int(value) => Some(value.clone()),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => tensor
            .integer_storage()
            .and_then(|storage| storage.value_at(0)),
        _ => None,
    }
}

fn integer_values(value: &Value) -> Option<Vec<IntValue>> {
    match value {
        Value::Int(value) => Some(vec![value.clone()]),
        Value::Tensor(tensor) => tensor
            .integer_storage()
            .map(|storage| storage.exact_values()),
        _ => None,
    }
}

pub(super) fn tensor_values_f64(tensor: &Tensor) -> Vec<f64> {
    tensor::tensor_values_f64(tensor)
}

pub(super) fn integer_i128(value: NumericScalar) -> Option<i128> {
    match value {
        NumericScalar::I8(value) => Some(i128::from(value)),
        NumericScalar::I16(value) => Some(i128::from(value)),
        NumericScalar::I32(value) => Some(i128::from(value)),
        NumericScalar::I64(value) => Some(i128::from(value)),
        NumericScalar::U8(value) => Some(i128::from(value)),
        NumericScalar::U16(value) => Some(i128::from(value)),
        NumericScalar::U32(value) => Some(i128::from(value)),
        NumericScalar::U64(value) => Some(i128::from(value)),
        NumericScalar::F64(_) | NumericScalar::F32(_) => None,
    }
}
