use super::*;
pub(super) fn parse_content_type(value: &Value) -> BuiltinResult<HttpResponseMode> {
    let text = expect_string_scalar(
        value,
        "webwrite: ContentType must be a character vector or string scalar",
    )?;
    HttpResponseMode::decode(&text, HttpResponseDialect::WebWrite).ok_or_else(|| {
        webwrite_error("webwrite: ContentType must be 'auto', 'json', 'text', or 'binary'")
    })
}

pub(super) fn parse_timeout(value: &Value) -> BuiltinResult<Duration> {
    let seconds = numeric_scalar(
        value,
        "webwrite: Timeout must be a positive scalar numeric value within the supported range or Inf",
    )?;
    if seconds <= 0.0 || (seconds.is_finite() && seconds > MAX_TIMEOUT_SECONDS) {
        return Err(webwrite_error(
            "webwrite: Timeout must be a positive scalar numeric value within the supported range or Inf",
        ));
    }
    Ok(Duration::from_secs_f64(if seconds.is_infinite() {
        MAX_TIMEOUT_SECONDS
    } else {
        seconds
    }))
}

pub(super) fn parse_request_method(value: &Value) -> BuiltinResult<HttpMethod> {
    let text = expect_string_scalar(
        value,
        "webwrite: RequestMethod must be a character vector or string scalar",
    )?;
    match text.trim().to_ascii_lowercase().as_str() {
        "auto" => Ok(HttpMethod::Post),
        "post" => Ok(HttpMethod::Post),
        "put" => Ok(HttpMethod::Put),
        "patch" => Ok(HttpMethod::Patch),
        "delete" => Ok(HttpMethod::Delete),
        other => Err(webwrite_error(format!(
            "webwrite: unsupported RequestMethod '{}'; expected auto, post, put, patch, or delete",
            other
        ))),
    }
}

pub(super) fn parse_header_fields(value: &Value) -> BuiltinResult<Vec<(String, String)>> {
    match value {
        Value::Struct(struct_value) => {
            let mut headers = Vec::with_capacity(struct_value.fields.len());
            for (key, val) in &struct_value.fields {
                let header_value = expect_string_scalar(
                    val,
                    "webwrite: header values must be character vectors or string scalars",
                )?;
                headers.push((key.to_owned(), header_value));
            }
            Ok(headers)
        }
        Value::Cell(cell) => {
            if cell.cols != 2 {
                return Err(webwrite_error(
                    "webwrite: HeaderFields cell array must have exactly two columns",
                ));
            }
            let mut headers = Vec::with_capacity(cell.rows);
            for row in 0..cell.rows {
                let name = cell
                    .get(row, 0)
                    .map_err(|err| webwrite_error(format!("webwrite: {err}")))?;
                let value = cell
                    .get(row, 1)
                    .map_err(|err| webwrite_error(format!("webwrite: {err}")))?;
                let header_name = expect_string_scalar(
                    &name,
                    "webwrite: header names must be character vectors or string scalars",
                )?;
                if header_name.trim().is_empty() {
                    return Err(webwrite_error("webwrite: header names must not be empty"));
                }
                let header_value = expect_string_scalar(
                    &value,
                    "webwrite: header values must be character vectors or string scalars",
                )?;
                headers.push((header_name, header_value));
            }
            Ok(headers)
        }
        _ => Err(webwrite_error(
            "webwrite: HeaderFields must be a struct or two-column cell array",
        )),
    }
}

pub(super) fn map_json_error(err: RuntimeError) -> RuntimeError {
    let message = if let Some(rest) = err.message().strip_prefix("jsondecode: ") {
        format!("webwrite: failed to parse JSON response ({rest})")
    } else {
        format!(
            "webwrite: failed to parse JSON response ({})",
            err.message()
        )
    };
    webwrite_error_with_source(&WEBWRITE_ERROR_RESPONSE_JSON, message, err)
}

pub(super) fn numeric_scalar(value: &Value, context: &str) -> BuiltinResult<f64> {
    match value {
        Value::Num(n) => Ok(*n),
        Value::Int(i) => Ok(i.to_f64()),
        Value::Tensor(tensor) => {
            if tensor_utils::is_scalar_tensor(tensor) {
                Ok(tensor_utils::tensor_value_f64(tensor, 0))
            } else {
                Err(webwrite_error(context))
            }
        }
        _ => Err(webwrite_error(context)),
    }
}

pub(super) fn scalar_to_string(value: &Value) -> BuiltinResult<String> {
    match value {
        Value::String(runmat_value::StringElement::Text(s)) => Ok(s.as_str().to_owned()),
        Value::CharArray(ca) if ca.rows == 1 => {
            crate::builtins::io::utf16::decode_utf16_with_error(ca.data(), || {
                webwrite_error("webwrite: value contains malformed UTF-16 text")
            })
        }
        Value::StringArray(sa) if sa.data().len() == 1 && !sa.data()[0].is_missing() => Ok(sa
            .data()[0]
            .as_str()
            .expect("nonmissing scalar text")
            .to_owned()),
        Value::Num(n) => Ok(format!("{}", n)),
        Value::Int(i) => Ok(i.decimal_string()),
        Value::Bool(b) => Ok(if *b { "true".into() } else { "false".into() }),
        Value::Tensor(tensor) => {
            if tensor_utils::is_scalar_tensor(tensor) {
                Ok(tensor
                    .integer_storage()
                    .and_then(|storage| storage.value_at(0))
                    .map_or_else(
                        || format!("{}", tensor_utils::tensor_value_f64(tensor, 0)),
                        |value| value.decimal_string(),
                    ))
            } else {
                Err(webwrite_error(
                    "webwrite: expected scalar value for text payload",
                ))
            }
        }
        Value::LogicalArray(array) => {
            if array.len() == 1 {
                Ok(if array.data[0] != 0 {
                    "true".into()
                } else {
                    "false".into()
                })
            } else {
                Err(webwrite_error(
                    "webwrite: expected scalar value for text payload",
                ))
            }
        }
        _ => Err(webwrite_error(
            "webwrite: unsupported value type for text payload",
        )),
    }
}

pub(super) fn expect_string_scalar(value: &Value, context: &str) -> BuiltinResult<String> {
    match value {
        Value::String(runmat_value::StringElement::Text(s)) => Ok(s.as_str().to_owned()),
        Value::CharArray(ca) if ca.rows == 1 => {
            crate::builtins::io::utf16::decode_utf16_with_error(ca.data(), || {
                webwrite_error(context)
            })
        }
        Value::StringArray(sa) if sa.data().len() == 1 && !sa.data()[0].is_missing() => Ok(sa
            .data()[0]
            .as_str()
            .expect("nonmissing scalar text")
            .to_owned()),
        _ => Err(webwrite_error(context)),
    }
}

pub(super) fn value_to_query_string(value: &Value, name: &str) -> BuiltinResult<String> {
    match value {
        Value::String(runmat_value::StringElement::Text(s)) => Ok(s.as_str().to_owned()),
        Value::CharArray(ca) if ca.rows == 1 => {
            crate::builtins::io::utf16::decode_utf16_with_error(ca.data(), || {
                webwrite_error(format!(
                    "webwrite: query parameter {name} contains malformed UTF-16"
                ))
            })
        }
        Value::StringArray(sa) if sa.data().len() == 1 && !sa.data()[0].is_missing() => Ok(sa
            .data()[0]
            .as_str()
            .expect("nonmissing scalar text")
            .to_owned()),
        Value::Num(n) => Ok(format!("{}", n)),
        Value::Int(i) => Ok(i.decimal_string()),
        Value::Bool(b) => Ok(if *b { "true".into() } else { "false".into() }),
        Value::Tensor(tensor) => {
            if tensor.shape().len() > 2 || (tensor.rows() > 1 && tensor.cols() > 1) {
                Err(webwrite_error(format!(
                    "webwrite: query parameter '{}' must be a scalar or vector",
                    name
                )))
            } else {
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
        }
        Value::LogicalArray(array) => {
            if array.shape.len() > 2
                || (array.shape.first().copied().unwrap_or(1) > 1
                    && array.shape.get(1).copied().unwrap_or(1) > 1)
            {
                Err(webwrite_error(format!(
                    "webwrite: query parameter '{}' must be a scalar or vector",
                    name
                )))
            } else {
                Ok(array
                    .data
                    .iter()
                    .map(|value| if *value != 0 { "true" } else { "false" })
                    .collect::<Vec<_>>()
                    .join(","))
            }
        }
        _ => Err(webwrite_error(format!(
            "webwrite: unsupported value type for query parameter '{}'",
            name
        ))),
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
