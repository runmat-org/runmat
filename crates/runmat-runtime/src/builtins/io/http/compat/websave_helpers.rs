use super::*;
pub(super) fn preflight_websave_rest(
    args: crate::call::FunctionArgumentView<'_>,
    prepared_api_key: Option<&super::super::api_key::PreparedApiKey>,
) -> BuiltinResult<()> {
    let mut headers = Vec::new();
    let mut timeout = DEFAULT_TIMEOUT;
    let mut user_agent = USER_AGENT.to_string();
    let mut index = 0;
    if let Some(Value::Struct(options)) = args.first() {
        if let Some(prepared) = prepared_api_key {
            prepared.preflight_known(options).map_err(|err| {
                compat_error_with_source("websave", format!("websave: {err}"), err)
            })?;
        }
        if !crate::dispatcher::value_contains_gpu(&args[0]) {
            apply_websave_options(options, &mut headers, &mut timeout, &mut user_agent)?;
        }
        index = 1;
    }
    if !(args.len() - index).is_multiple_of(2) {
        return Err(compat_error(
            "websave",
            "websave: name-value arguments must be paired",
        ));
    }
    while index < args.len() {
        if !crate::dispatcher::value_contains_gpu(&args[index]) {
            let name = scalar_text(&args[index], "websave", "name")?;
            let value = &args[index + 1];
            if !crate::dispatcher::value_contains_gpu(value) {
                match name.to_ascii_lowercase().as_str() {
                    "timeout" => {
                        let _ = numeric_seconds(value, "websave Timeout")?;
                    }
                    "useragent" => {
                        let _ = scalar_text(value, "websave", "UserAgent")?;
                    }
                    "headerfields" => {
                        return Err(compat_error(
                            "websave",
                            "websave: HeaderFields must be supplied through weboptions",
                        ));
                    }
                    _ => {
                        let _ = query_value_to_string(value, &name)?;
                    }
                }
            }
        }
        index += 2;
    }
    Ok(())
}

pub(super) fn parse_websave_rest(
    url: &mut Url,
    headers: &mut Vec<(String, String)>,
    timeout: &mut Duration,
    user_agent: &mut String,
    args: crate::call::FunctionArgumentView<'_>,
    prepared_api_key: Option<&super::super::api_key::PreparedApiKey>,
) -> BuiltinResult<()> {
    let mut idx = 0usize;
    if let Some(Value::Struct(options)) = args.first() {
        if let Some(prepared) = prepared_api_key {
            if let Some(header) = prepared.finish(options).map_err(|err| {
                compat_error_with_source("websave", format!("websave: {err}"), err)
            })? {
                headers.push(header);
            }
        }
        apply_websave_options(options, headers, timeout, user_agent)?;
        idx = 1;
    }
    if !(args.len() - idx).is_multiple_of(2) {
        return Err(compat_error(
            "websave",
            "websave: name-value arguments must be paired",
        ));
    }
    while idx < args.len() {
        let name = scalar_text(&args[idx], "websave", "name")?;
        let value = &args[idx + 1];
        match name.to_ascii_lowercase().as_str() {
            "timeout" => {
                *timeout = duration_from_seconds(numeric_seconds(value, "websave Timeout")?)
            }
            "useragent" => *user_agent = scalar_text(value, "websave", "UserAgent")?,
            "headerfields" => {
                return Err(compat_error(
                    "websave",
                    "websave: HeaderFields must be supplied through weboptions",
                ));
            }
            _ => {
                let value = query_value_to_string(value, &name)?;
                url.query_pairs_mut().append_pair(&name, &value);
            }
        }
        idx += 2;
    }
    Ok(())
}

pub(super) fn apply_websave_options(
    options: &runmat_value::StructValue,
    headers: &mut Vec<(String, String)>,
    timeout: &mut Duration,
    user_agent: &mut String,
) -> BuiltinResult<()> {
    if let Some(value) = options.fields.get("Timeout") {
        *timeout = duration_from_seconds(numeric_seconds(value, "websave Timeout")?);
    }
    if let Some(value) = options.fields.get("UserAgent") {
        let ua = scalar_text(value, "websave", "UserAgent")?;
        if !ua.is_empty() {
            *user_agent = ua;
        }
    }
    if let Some(value) = options.fields.get("HeaderFields") {
        append_header_fields(value, headers)?;
    }
    Ok(())
}

pub(super) fn numeric_seconds(value: &Value, label: &str) -> BuiltinResult<f64> {
    match value {
        Value::Num(v) if *v > 0.0 && (!v.is_finite() || *v <= MAX_TIMEOUT_SECONDS) => Ok(*v),
        Value::Int(v) if v.to_i64() > 0 && v.to_f64() <= MAX_TIMEOUT_SECONDS => Ok(v.to_f64()),
        Value::Tensor(tensor) if tensor.len() == 1 => {
            let value = tensor
                .numeric_value_at(0)
                .expect("timeout tensor index must exist");
            let seconds = format_numeric_scalar(value).parse::<f64>().map_err(|_| {
                compat_error(
                    "websave",
                    format!("websave: {label} must be a positive numeric scalar"),
                )
            })?;
            if seconds > 0.0 && (!seconds.is_finite() || seconds <= MAX_TIMEOUT_SECONDS) {
                Ok(seconds)
            } else {
                Err(compat_error(
                    "websave",
                    format!("websave: {label} is outside the supported timeout range"),
                ))
            }
        }
        Value::String(runmat_value::StringElement::Text(text)) => parse_positive_seconds(text),
        Value::CharArray(array) if array.rows == 1 => {
            let text = crate::builtins::io::utf16::decode_utf16_with_error(array.data(), || {
                compat_error(
                    "websave",
                    format!("websave: {label} contains malformed UTF-16 text"),
                )
            })?;
            parse_positive_seconds(&text)
        }
        _ => Err(compat_error(
            "websave",
            format!("websave: {label} must be a positive finite scalar"),
        )),
    }
}

pub(super) fn parse_positive_seconds(text: &str) -> BuiltinResult<f64> {
    let seconds: f64 = text.trim().parse().map_err(|_| {
        compat_error(
            "websave",
            "websave: Timeout must be a positive finite scalar",
        )
    })?;
    if seconds > 0.0 && (!seconds.is_finite() || seconds <= MAX_TIMEOUT_SECONDS) {
        Ok(seconds)
    } else {
        Err(compat_error(
            "websave",
            "websave: Timeout must be a positive finite scalar",
        ))
    }
}

pub(super) fn duration_from_seconds(seconds: f64) -> Duration {
    Duration::from_secs_f64(if seconds.is_infinite() {
        MAX_TIMEOUT_SECONDS
    } else {
        seconds
    })
}

pub(super) fn query_value_to_string(value: &Value, name: &str) -> BuiltinResult<String> {
    match value {
        Value::String(_) | Value::CharArray(_) | Value::StringArray(_) => {
            scalar_text(value, "websave", name)
        }
        Value::Num(value) => Ok(value.to_string()),
        Value::Int(value) => Ok(value.decimal_string()),
        Value::Bool(value) => Ok(if *value { "true" } else { "false" }.to_string()),
        Value::Tensor(tensor)
            if tensor.shape().len() <= 2 && (tensor.rows() == 1 || tensor.cols() == 1) =>
        {
            Ok((0..tensor.len())
                .map(|index| {
                    format_numeric_scalar(
                        tensor
                            .numeric_value_at(index)
                            .expect("query tensor index must exist"),
                    )
                })
                .collect::<Vec<_>>()
                .join(","))
        }
        Value::LogicalArray(array)
            if array.shape.len() <= 2
                && (array.shape.first().copied().unwrap_or(1) == 1
                    || array.shape.get(1).copied().unwrap_or(1) == 1) =>
        {
            Ok(array
                .data
                .iter()
                .map(|value| if *value != 0 { "true" } else { "false" })
                .collect::<Vec<_>>()
                .join(","))
        }
        _ => Err(compat_error(
            "websave",
            format!("websave: query value '{name}' must be text or a numeric, logical, or datetime scalar or vector"),
        )),
    }
}

pub(super) fn format_numeric_scalar(value: NumericScalar) -> String {
    match value {
        NumericScalar::F64(value) => value.to_string(),
        NumericScalar::F32(value) => value.to_string(),
        NumericScalar::I8(value) => value.to_string(),
        NumericScalar::I16(value) => value.to_string(),
        NumericScalar::I32(value) => value.to_string(),
        NumericScalar::I64(value) => value.to_string(),
        NumericScalar::U8(value) => value.to_string(),
        NumericScalar::U16(value) => value.to_string(),
        NumericScalar::U32(value) => value.to_string(),
        NumericScalar::U64(value) => value.to_string(),
    }
}

pub(super) fn append_header_fields(
    value: &Value,
    headers: &mut Vec<(String, String)>,
) -> BuiltinResult<()> {
    match value {
        Value::Struct(st) => {
            for (name, value) in &st.fields {
                let header_value = scalar_text(value, "websave", "HeaderFields value")?;
                if !name.is_empty() && !header_value.is_empty() {
                    headers.push((name.to_owned(), header_value));
                }
            }
            Ok(())
        }
        Value::Cell(cell) if cell.cols == 2 => {
            for row in 0..cell.rows {
                let name = scalar_text(&cell.data()[row * cell.cols], "websave", "header name")?;
                let value =
                    scalar_text(&cell.data()[row * cell.cols + 1], "websave", "header value")?;
                if !name.is_empty() && !value.is_empty() {
                    headers.push((name, value));
                }
            }
            Ok(())
        }
        Value::Cell(_) => Err(compat_error(
            "websave",
            "websave: HeaderFields cell array must have two columns",
        )),
        _ => Err(compat_error(
            "websave",
            "websave: HeaderFields must be a struct or two-column cell array",
        )),
    }
}

pub(super) fn path_from_value(value: &Value, name: &str) -> BuiltinResult<PathBuf> {
    let text = scalar_text(value, name, "filename")?;
    Ok(PathBuf::from(
        expand_user_path(text.trim(), name).map_err(|err| compat_error(name, err))?,
    ))
}
