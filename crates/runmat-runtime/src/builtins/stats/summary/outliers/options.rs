//! Numeric-array outlier option parsing and exact structural controls.
use super::*;

pub(super) async fn parse_detection_options(
    operation: OutlierOperation,
    rest: Vec<Value>,
) -> BuiltinResult<DetectionOptions> {
    let builtin = operation.name();
    let rest = gather_values(rest).await?;
    let mut options = DetectionOptions::default();
    let mut idx = 0usize;
    while idx < rest.len() {
        let arg = &rest[idx];
        if let Some(keyword) = keyword_of(arg) {
            match keyword.to_ascii_lowercase().as_str() {
                "median" => {
                    options.method = DetectionMethod::Median;
                    options.method_was_specified = true;
                }
                "mean" => {
                    options.method = DetectionMethod::Mean;
                    options.method_was_specified = true;
                }
                "quartiles" => {
                    options.method = DetectionMethod::Quartiles;
                    options.method_was_specified = true;
                }
                "percentiles" => {
                    options.method_was_specified = true;
                    idx += 1;
                    if idx >= rest.len() {
                        return Err(invalid_argument(
                            builtin,
                            "percentiles requires [lower upper]",
                        ));
                    }
                    let bounds = numeric_vector(builtin, &rest[idx])?;
                    if bounds.len() != 2 {
                        return Err(invalid_argument(
                            builtin,
                            "percentiles must be a two-element vector",
                        ));
                    }
                    if !(0.0..=100.0).contains(&bounds[0])
                        || !(0.0..=100.0).contains(&bounds[1])
                        || bounds[0] >= bounds[1]
                    {
                        return Err(invalid_argument(
                            builtin,
                            "percentiles must satisfy 0 <= p1 < p2 <= 100",
                        ));
                    }
                    options.method =
                        DetectionMethod::Percentiles(bounds[0] / 100.0, bounds[1] / 100.0);
                }
                "movmedian" | "movmean" => {
                    options.method_was_specified = true;
                    let is_median = keyword.eq_ignore_ascii_case("movmedian");
                    idx += 1;
                    if idx >= rest.len() {
                        return Err(invalid_argument(builtin, "moving method requires a window"));
                    }
                    let window = scalar_usize(builtin, &rest[idx], "window")?;
                    if window == 0 {
                        return Err(invalid_argument(builtin, "window must be positive"));
                    }
                    options.method = if is_median {
                        DetectionMethod::MovingMedian(window)
                    } else {
                        DetectionMethod::MovingMean(window)
                    };
                }
                "thresholdfactor" => {
                    idx += 1;
                    if idx >= rest.len() {
                        return Err(invalid_argument(
                            builtin,
                            "ThresholdFactor requires a numeric scalar",
                        ));
                    }
                    let factor = scalar_number(&rest[idx]).ok_or_else(|| {
                        invalid_argument(builtin, "ThresholdFactor must be a numeric scalar")
                    })?;
                    if !(factor.is_finite() && factor >= 0.0) {
                        return Err(invalid_argument(
                            builtin,
                            "ThresholdFactor must be a finite nonnegative scalar",
                        ));
                    }
                    options.threshold_factor = Some(factor);
                }
                "outlierlocations" if operation == OutlierOperation::Fill => {
                    idx += 1;
                    if idx >= rest.len() {
                        return Err(invalid_argument(
                            builtin,
                            "OutlierLocations requires a logical mask value",
                        ));
                    }
                    options.forced_mask = Some(logical_mask(builtin, &rest[idx])?);
                }
                "samplepoints" | "data variables" | "datavariables" | "outlierlocations" => {
                    idx += 1;
                    if idx >= rest.len() {
                        return Err(invalid_argument(
                            builtin,
                            format!("{keyword} requires a value"),
                        ));
                    }
                    return Err(unsupported(
                        builtin,
                        format!(
                            "{keyword} is not supported for numeric-array outlier detection yet"
                        ),
                    ));
                }
                other => {
                    return Err(invalid_argument(
                        builtin,
                        format!("unsupported option or method '{other}'"),
                    ));
                }
            }
        } else {
            options.dim = Some(parse_dim(builtin, arg)?);
        }
        idx += 1;
    }
    if matches!(options.method, DetectionMethod::Percentiles(_, _))
        && options.threshold_factor.is_some()
    {
        return Err(invalid_argument(
            builtin,
            "ThresholdFactor is not supported with the percentiles method",
        ));
    }
    if options.forced_mask.is_some() && options.method_was_specified {
        return Err(invalid_argument(
            builtin,
            "OutlierLocations cannot be combined with a specified find method",
        ));
    }
    Ok(options)
}

pub(super) fn parse_fill_method(rest: &mut Vec<Value>) -> BuiltinResult<FillMethod> {
    let value = rest.remove(0);
    if let Some(number) = scalar_number(&value) {
        return Ok(FillMethod::Constant(number));
    }
    let Some(keyword) = keyword_of(&value) else {
        return Err(invalid_argument(
            FILLOUTLIERS_NAME,
            "filloutliers: fill method must be text or numeric scalar",
        ));
    };
    match keyword.to_ascii_lowercase().as_str() {
        "center" => Ok(FillMethod::Center),
        "clip" => Ok(FillMethod::Clip),
        "previous" => Ok(FillMethod::Previous),
        "next" => Ok(FillMethod::Next),
        "nearest" => Ok(FillMethod::Nearest),
        "linear" => Ok(FillMethod::Linear),
        "constant" => Err(invalid_argument(
            FILLOUTLIERS_NAME,
            "filloutliers: use a numeric scalar fill method for constant replacement",
        )),
        "spline" | "pchip" | "makima" => Err(unsupported(
            FILLOUTLIERS_NAME,
            format!("filloutliers: interpolation method '{keyword}' is not supported yet"),
        )),
        other => Err(invalid_argument(
            FILLOUTLIERS_NAME,
            format!("filloutliers: unsupported fill method '{other}'"),
        )),
    }
}

fn parse_dim(builtin: &'static str, value: &Value) -> BuiltinResult<usize> {
    tensor::parse_dimension(value, builtin).map_err(|err| invalid_argument(builtin, err))
}

pub(super) fn logical_mask(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<(Vec<u8>, Vec<usize>)> {
    match value {
        Value::Bool(flag) => Ok((vec![u8::from(*flag)], vec![1, 1])),
        Value::LogicalArray(mask) => Ok((
            mask.data.iter().map(|flag| u8::from(*flag != 0)).collect(),
            mask.shape.clone(),
        )),
        Value::Tensor(tensor) => Ok((
            tensor
                .materialize_f64()
                .into_iter()
                .map(|value| u8::from(value != 0.0 && !value.is_nan()))
                .collect(),
            tensor.shape.clone(),
        )),
        Value::Num(value) => Ok((vec![u8::from(*value != 0.0 && !value.is_nan())], vec![1, 1])),
        Value::Int(value) => Ok((vec![u8::from(!value.is_zero())], vec![1, 1])),
        other => Err(invalid_argument(
            builtin,
            format!("OutlierLocations must be logical or numeric, got {other:?}"),
        )),
    }
}

pub(super) fn numeric_vector(builtin: &'static str, value: &Value) -> BuiltinResult<Vec<f64>> {
    let tensor = tensor::value_into_tensor_for(builtin, value.clone())
        .map_err(|err| invalid_argument(builtin, format!("{builtin}: {err}")))?;
    Ok(tensor_values_f64(&tensor))
}

pub(super) fn scalar_usize(
    builtin: &'static str,
    value: &Value,
    label: &str,
) -> BuiltinResult<usize> {
    if let Some(value) = integer_scalar(value) {
        return value.try_to_usize().ok_or_else(|| {
            invalid_argument(builtin, format!("{label} must be a nonnegative integer"))
        });
    }
    let number = scalar_number(value)
        .ok_or_else(|| invalid_argument(builtin, format!("{label} must be numeric")))?;
    if !(number.is_finite() && number >= 0.0 && number.fract() == 0.0) {
        return Err(invalid_argument(
            builtin,
            format!("{label} must be a nonnegative integer"),
        ));
    }
    if number > usize::MAX as f64 || (usize::BITS == 64 && number == usize::MAX as f64) {
        return Err(invalid_argument(
            builtin,
            format!("{label} exceeds platform integer limits"),
        ));
    }
    Ok(number as usize)
}

pub(super) fn scalar_number(value: &Value) -> Option<f64> {
    if let Some(value) = integer_scalar(value) {
        return Some(value.to_f64());
    }
    match value {
        Value::Num(value) => Some(*value),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            Some(tensor::tensor_value_f64(tensor, 0))
        }
        _ => None,
    }
}

fn integer_scalar(value: &Value) -> Option<IntValue> {
    match value {
        Value::Int(value) => Some(value.clone()),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => tensor
            .integer_storage()
            .and_then(|storage| storage.value_at(0)),
        _ => None,
    }
}

fn tensor_values_f64(tensor: &Tensor) -> Vec<f64> {
    tensor::tensor_values_f64(tensor)
}
