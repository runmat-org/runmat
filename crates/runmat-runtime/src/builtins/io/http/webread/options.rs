use super::*;
pub(super) fn parse_header_fields(value: &Value) -> BuiltinResult<Vec<(String, String)>> {
    match value {
        Value::Struct(struct_value) => {
            let mut headers = Vec::with_capacity(struct_value.fields.len());
            for (key, val) in &struct_value.fields {
                let header_value = expect_string_scalar(
                    val,
                    "webread: header values must be character vectors or string scalars",
                )?;
                headers.push((key.to_owned(), header_value));
            }
            Ok(headers)
        }
        Value::Cell(cell) => {
            if cell.cols != 2 {
                return Err(webread_error(
                    "webread: HeaderFields cell array must have exactly two columns",
                ));
            }
            let mut headers = Vec::with_capacity(cell.rows);
            for row in 0..cell.rows {
                let name = cell
                    .get(row, 0)
                    .map_err(|err| webread_error(format!("webread: {err}")))?;
                let value = cell
                    .get(row, 1)
                    .map_err(|err| webread_error(format!("webread: {err}")))?;
                let header_name = expect_string_scalar(
                    &name,
                    "webread: header names must be character vectors or string scalars",
                )?;
                if header_name.trim().is_empty() {
                    return Err(webread_error("webread: header names must not be empty"));
                }
                let header_value = expect_string_scalar(
                    &value,
                    "webread: header values must be character vectors or string scalars",
                )?;
                headers.push((header_name, header_value));
            }
            Ok(headers)
        }
        _ => Err(webread_error(
            "webread: HeaderFields must be provided as a struct or cell array of name/value pairs",
        )),
    }
}

pub(super) fn parse_content_type(value: &Value) -> BuiltinResult<HttpResponseMode> {
    let text = expect_string_scalar(
        value,
        "webread: ContentType must be a character vector or string scalar",
    )?;
    HttpResponseMode::decode(&text, HttpResponseDialect::WebRead).ok_or_else(|| {
        webread_error(format!(
            "webread: unsupported ContentType '{}'; use 'auto', 'json', 'text', or 'binary'",
            text.trim().to_ascii_lowercase()
        ))
    })
}

pub(super) fn parse_timeout(value: &Value) -> BuiltinResult<Duration> {
    let seconds = numeric_scalar(value, "webread: Timeout must be a finite, positive scalar")?;
    if seconds <= 0.0 || (seconds.is_finite() && seconds > MAX_TIMEOUT_SECONDS) {
        return Err(webread_error(
            "webread: Timeout must be a positive scalar within the supported range or Inf",
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
        "webread: RequestMethod must be a character vector or string scalar",
    )?;
    let lower = text.trim().to_ascii_lowercase();
    match lower.as_str() {
        "get" | "auto" => Ok(HttpMethod::Get),
        other => Err(webread_error(format!(
            "webread: RequestMethod '{}' is not supported; expected 'auto' or 'get'",
            other
        ))),
    }
}

pub(super) fn numeric_scalar(value: &Value, context: &str) -> BuiltinResult<f64> {
    match value {
        Value::Num(n) => Ok(*n),
        Value::Int(i) => Ok(i.to_f64()),
        Value::Tensor(tensor) => {
            if tensor_utils::is_scalar_tensor(tensor) {
                Ok(tensor_utils::tensor_value_f64(tensor, 0))
            } else {
                Err(webread_error(context))
            }
        }
        _ => Err(webread_error(context)),
    }
}

pub(super) fn expect_string_scalar(value: &Value, context: &str) -> BuiltinResult<String> {
    match value {
        Value::String(runmat_value::StringElement::Text(s)) => Ok(s.as_str().to_owned()),
        Value::CharArray(ca) if ca.rows == 1 => {
            crate::builtins::io::utf16::decode_utf16_with_error(ca.data(), || {
                webread_error(context)
            })
        }
        Value::StringArray(sa) if sa.data().len() == 1 && !sa.data()[0].is_missing() => Ok(sa
            .data()[0]
            .as_str()
            .expect("nonmissing scalar text")
            .to_owned()),
        _ => Err(webread_error(context)),
    }
}

pub(super) fn value_to_query_string(value: &Value, name: &str) -> BuiltinResult<String> {
    match value {
        Value::String(runmat_value::StringElement::Text(s)) => Ok(s.as_str().to_owned()),
        Value::CharArray(ca) if ca.rows == 1 => {
            crate::builtins::io::utf16::decode_utf16_with_error(ca.data(), || {
                webread_error(format!(
                    "webread: query parameter {name} contains malformed UTF-16"
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
                Err(webread_error(format!(
                    "webread: query parameter '{}' must be a scalar or vector",
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
                Err(webread_error(format!(
                    "webread: query parameter '{}' must be a scalar or vector",
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
        _ => Err(webread_error(format!(
            "webread: unsupported value type for query parameter '{}'",
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

#[derive(Clone, Copy, Debug)]
pub(super) enum ResolvedContentType {
    Text,
    Json,
    Binary,
}

#[derive(Clone, Debug)]
pub(super) struct WebReadOptions {
    pub(super) content_type: HttpResponseMode,
    pub(super) timeout: Duration,
    pub(super) headers: Vec<(String, String)>,
    pub(super) api_key_header: Option<(String, String)>,
    pub(super) user_agent: Option<String>,
    pub(super) username: Option<String>,
    pub(super) password: Option<String>,
    pub(super) method: HttpMethod,
}

impl Default for WebReadOptions {
    fn default() -> Self {
        Self {
            content_type: HttpResponseMode::Auto,
            timeout: Duration::from_secs_f64(DEFAULT_TIMEOUT_SECONDS),
            headers: Vec::new(),
            api_key_header: None,
            user_agent: None,
            username: None,
            password: None,
            method: HttpMethod::Get,
        }
    }
}

impl WebReadOptions {
    pub(super) fn resolve_content_type(&self, header: Option<&str>) -> ResolvedContentType {
        match self.content_type {
            HttpResponseMode::Json => ResolvedContentType::Json,
            HttpResponseMode::Text => ResolvedContentType::Text,
            HttpResponseMode::Binary => ResolvedContentType::Binary,
            HttpResponseMode::Auto => infer_content_type(header),
        }
    }
}

pub(super) fn infer_content_type(header: Option<&str>) -> ResolvedContentType {
    if let Some(raw) = header {
        let mime = raw
            .split(';')
            .next()
            .map(|part| part.trim().to_ascii_lowercase())
            .unwrap_or_default();
        if mime == "application/json" || mime == "text/json" || mime.ends_with("+json") {
            ResolvedContentType::Json
        } else if mime.starts_with("text/")
            || mime == "application/xml"
            || mime.ends_with("+xml")
            || mime == "application/xhtml+xml"
            || mime == "application/javascript"
            || mime == "application/x-www-form-urlencoded"
        {
            ResolvedContentType::Text
        } else {
            ResolvedContentType::Binary
        }
    } else {
        ResolvedContentType::Text
    }
}
