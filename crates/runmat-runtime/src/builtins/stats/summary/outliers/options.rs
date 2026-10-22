//! Numeric-array outlier option parsing and exact structural controls.
use super::*;

pub(super) struct DecodedOptions {
    tokens: Vec<Option<OutlierOption>>,
}

impl DecodedOptions {
    pub(super) fn new(
        operation: OutlierOperation,
        rest: crate::call::FunctionArgumentView<'_>,
    ) -> BuiltinResult<Self> {
        let mut tokens = Vec::with_capacity(rest.len());
        for (index, value) in rest.iter().enumerate() {
            tokens.push(if operation == OutlierOperation::Fill && index == 0 {
                None
            } else {
                keyword_of(value)?.and_then(|keyword| OutlierOption::parse(&keyword))
            });
        }
        Ok(Self { tokens })
    }

    pub(super) fn at(&self, index: usize) -> Option<OutlierOption> {
        self.tokens.get(index).copied().flatten()
    }

    pub(super) fn after_fill(self) -> Vec<Option<OutlierOption>> {
        self.tokens.into_iter().skip(1).collect()
    }

    pub(super) fn all(self) -> Vec<Option<OutlierOption>> {
        self.tokens
    }
}

pub(super) async fn preflight_host_options(
    operation: OutlierOperation,
    rest: crate::call::FunctionArgumentView<'_>,
    decoded: &DecodedOptions,
) -> BuiltinResult<()> {
    let builtin = operation.name();
    let start = if operation == OutlierOperation::Fill {
        let Some(fill) = rest.first() else {
            return Err(invalid_argument(
                builtin,
                "filloutliers: fill method is required",
            ));
        };
        if let Some(keyword) = keyword_of(fill)? {
            parse_fill_keyword(&keyword)?;
        } else if !matches!(fill, Value::GpuTensor(_)) && scalar_number(fill).is_none() {
            return Err(invalid_argument(
                builtin,
                "filloutliers: fill method must be text or numeric scalar",
            ));
        }
        1
    } else {
        0
    };
    let mut method_specified = false;
    let mut forced_mask = false;
    let mut percentiles = false;
    let mut threshold_factor = false;
    let mut index = start;
    while index < rest.len() {
        let Some(keyword) = keyword_of(&rest[index])? else {
            if !matches!(&rest[index], Value::GpuTensor(_)) {
                parse_dim(builtin, &rest[index])?;
            }
            index += 1;
            continue;
        };
        let Some(option) = decoded.at(index) else {
            return Err(invalid_argument(
                builtin,
                format!("unsupported option or method '{keyword}'"),
            ));
        };
        if option.unsupported_array_form()
            || (option == OutlierOption::OutlierLocations && operation == OutlierOperation::Detect)
        {
            return Err(unsupported(
                builtin,
                format!("{keyword} is not supported for numeric-array outlier detection yet"),
            ));
        }
        if option.consumes_value() && rest.get(index + 1).is_none() {
            return Err(invalid_argument(
                builtin,
                format!("{keyword} requires a value"),
            ));
        }
        if let Some(control) = rest.get(index + 1).filter(|_| option.consumes_value()) {
            if !matches!(control, Value::GpuTensor(_)) {
                match option {
                    OutlierOption::Percentiles => {
                        ensure_host_integer_exact(control, builtin).await?;
                        percentile_bounds(builtin, control)?;
                    }
                    OutlierOption::MovingMedian | OutlierOption::MovingMean => {
                        if scalar_usize(builtin, control, "window")? == 0 {
                            return Err(invalid_argument(builtin, "window must be positive"));
                        }
                    }
                    OutlierOption::ThresholdFactor => {
                        ensure_host_integer_exact(control, builtin).await?;
                        threshold_factor_value(builtin, control)?;
                    }
                    OutlierOption::OutlierLocations => {
                        logical_mask(builtin, control)?;
                    }
                    OutlierOption::SamplePoints
                    | OutlierOption::DataVariables
                    | OutlierOption::MaxNumOutliers
                    | OutlierOption::Median
                    | OutlierOption::Mean
                    | OutlierOption::Quartiles
                    | OutlierOption::Grubbs
                    | OutlierOption::Gesd => {}
                }
            }
        }
        match option {
            OutlierOption::Median
            | OutlierOption::Mean
            | OutlierOption::Quartiles
            | OutlierOption::Percentiles
            | OutlierOption::MovingMedian
            | OutlierOption::MovingMean => method_specified = true,
            OutlierOption::OutlierLocations => forced_mask = true,
            OutlierOption::ThresholdFactor
            | OutlierOption::SamplePoints
            | OutlierOption::DataVariables
            | OutlierOption::MaxNumOutliers
            | OutlierOption::Grubbs
            | OutlierOption::Gesd => {}
        }
        percentiles |= option == OutlierOption::Percentiles;
        threshold_factor |= option == OutlierOption::ThresholdFactor;
        index += if option.consumes_value() { 2 } else { 1 };
    }
    if percentiles && threshold_factor {
        return Err(invalid_argument(
            builtin,
            "ThresholdFactor is not supported with the percentiles method",
        ));
    }
    if forced_mask && method_specified {
        return Err(invalid_argument(
            builtin,
            "OutlierLocations cannot be combined with a specified find method",
        ));
    }
    Ok(())
}

async fn ensure_host_integer_exact(value: &Value, builtin: &'static str) -> BuiltinResult<()> {
    if is_typed_integer_value(value)
        && !crate::builtins::common::validation::native_integer_value_is_exact_f64_async(value)
            .await?
    {
        return Err(invalid_argument(
            builtin,
            "typed-integer threshold value must be exactly representable as double",
        ));
    }
    Ok(())
}

fn percentile_bounds(builtin: &'static str, value: &Value) -> BuiltinResult<(f64, f64)> {
    let bounds = numeric_vector(builtin, value)?;
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
    Ok((bounds[0] / 100.0, bounds[1] / 100.0))
}

fn threshold_factor_value(builtin: &'static str, value: &Value) -> BuiltinResult<f64> {
    let factor = scalar_number(value)
        .ok_or_else(|| invalid_argument(builtin, "ThresholdFactor must be a numeric scalar"))?;
    if !(factor.is_finite() && factor >= 0.0) {
        return Err(invalid_argument(
            builtin,
            "ThresholdFactor must be a finite nonnegative scalar",
        ));
    }
    Ok(factor)
}

pub(super) async fn parse_detection_options(
    operation: OutlierOperation,
    rest: crate::call::BuiltinArguments,
    decoded: Vec<Option<OutlierOption>>,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<DetectionOptions> {
    let rest = gather_values(rest, runtime).await?;
    parse_gathered_detection_options(operation, rest.view(), decoded)
}

pub(super) fn parse_gathered_detection_options(
    operation: OutlierOperation,
    rest: crate::call::FunctionArgumentView<'_>,
    decoded: Vec<Option<OutlierOption>>,
) -> BuiltinResult<DetectionOptions> {
    let builtin = operation.name();
    let mut options = DetectionOptions::default();
    let mut idx = 0usize;
    while idx < rest.len() {
        let arg = &rest[idx];
        if let Some(keyword) = keyword_of(arg)? {
            let option = decoded.get(idx).copied().flatten();
            match option {
                Some(OutlierOption::Median) => {
                    options.method = DetectionMethod::Median;
                    options.method_was_specified = true;
                }
                Some(OutlierOption::Mean) => {
                    options.method = DetectionMethod::Mean;
                    options.method_was_specified = true;
                }
                Some(OutlierOption::Quartiles) => {
                    options.method = DetectionMethod::Quartiles;
                    options.method_was_specified = true;
                }
                Some(OutlierOption::Percentiles) => {
                    options.method_was_specified = true;
                    idx += 1;
                    if idx >= rest.len() {
                        return Err(invalid_argument(
                            builtin,
                            "percentiles requires [lower upper]",
                        ));
                    }
                    let (lower, upper) = percentile_bounds(builtin, &rest[idx])?;
                    options.method = DetectionMethod::Percentiles(lower, upper);
                }
                Some(OutlierOption::MovingMedian | OutlierOption::MovingMean) => {
                    options.method_was_specified = true;
                    let is_median = option == Some(OutlierOption::MovingMedian);
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
                Some(OutlierOption::ThresholdFactor) => {
                    idx += 1;
                    if idx >= rest.len() {
                        return Err(invalid_argument(
                            builtin,
                            "ThresholdFactor requires a numeric scalar",
                        ));
                    }
                    options.threshold_factor = Some(threshold_factor_value(builtin, &rest[idx])?);
                }
                Some(OutlierOption::OutlierLocations) if operation == OutlierOperation::Fill => {
                    idx += 1;
                    if idx >= rest.len() {
                        return Err(invalid_argument(
                            builtin,
                            "OutlierLocations requires a logical mask value",
                        ));
                    }
                    options.forced_mask = Some(logical_mask(builtin, &rest[idx])?);
                }
                Some(
                    OutlierOption::SamplePoints
                    | OutlierOption::DataVariables
                    | OutlierOption::MaxNumOutliers
                    | OutlierOption::OutlierLocations,
                ) => {
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
                Some(OutlierOption::Grubbs | OutlierOption::Gesd) => {
                    return Err(unsupported(
                        builtin,
                        format!(
                            "{keyword} is not supported for numeric-array outlier detection yet"
                        ),
                    ));
                }
                None => {
                    return Err(invalid_argument(
                        builtin,
                        format!("unsupported option or method '{keyword}'"),
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

pub(super) fn parse_fill_method(value: &Value) -> BuiltinResult<FillMethod> {
    if let Some(number) = scalar_number(value) {
        return Ok(FillMethod::Constant(number));
    }
    let Some(keyword) = keyword_of(value)? else {
        return Err(invalid_argument(
            FILLOUTLIERS_NAME,
            "filloutliers: fill method must be text or numeric scalar",
        ));
    };
    parse_fill_keyword(&keyword)
}

fn parse_fill_keyword(keyword: &str) -> BuiltinResult<FillMethod> {
    match keyword.trim().to_ascii_lowercase().as_str() {
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
            mask.shape.to_vec(),
        )),
        Value::Tensor(tensor) => Ok((
            tensor
                .materialize_f64()
                .into_iter()
                .map(|value| u8::from(value != 0.0 && !value.is_nan()))
                .collect(),
            tensor.shape().to_vec(),
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
