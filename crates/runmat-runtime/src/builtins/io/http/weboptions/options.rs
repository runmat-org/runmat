use super::*;

pub(super) fn default_options_struct() -> super::builder::Options {
    super::builder::Options::new()
}

pub(super) fn set_option_field(
    options: &mut super::builder::Options,
    field: WebOptionField,
    value: &Value,
) -> BuiltinResult<()> {
    match field {
        WebOptionField::ContentType => {
            let canonical = parse_content_type_option(value)?;
            options.set(field, Value::from(canonical));
            Ok(())
        }
        WebOptionField::Timeout => {
            let seconds = numeric_scalar(
                value,
                "weboptions: Timeout must be a finite, positive scalar",
                &WEBOPTIONS_ERROR_INVALID_OPTION_VALUE,
            )?;
            if seconds <= 0.0 || (seconds.is_finite() && seconds > MAX_TIMEOUT_SECONDS) {
                return Err(weboptions_error_with(
                    &WEBOPTIONS_ERROR_INVALID_OPTION_VALUE,
                    "weboptions: Timeout must be a positive numeric scalar within the supported range or Inf",
                ));
            }
            options.set(field, Value::Num(seconds));
            Ok(())
        }
        WebOptionField::KeyName => {
            let key = expect_string_scalar(
                value,
                "weboptions: KeyName must be a character vector or string scalar",
                &WEBOPTIONS_ERROR_INVALID_OPTION_VALUE,
            )?;
            options.set(field, Value::from(key));
            Ok(())
        }
        WebOptionField::KeyValue => {
            let supported = match value {
                Value::String(_) | Value::Num(_) | Value::Int(_) | Value::Bool(_) => true,
                Value::CharArray(array) => array.rows == 1,
                Value::StringArray(array) => array.data().len() == 1,
                Value::Tensor(tensor) => tensor::is_scalar_tensor(tensor),
                Value::LogicalArray(array) => array.len() == 1,
                _ => false,
            };
            if !supported {
                return Err(weboptions_error_with(
                    &WEBOPTIONS_ERROR_INVALID_OPTION_VALUE,
                    "weboptions: KeyValue must be a text, numeric, or logical scalar",
                ));
            }
            options.set(field, value.clone());
            Ok(())
        }
        WebOptionField::HeaderFields => {
            let canonical = canonical_header_fields(value)?;
            options.set(field, canonical);
            Ok(())
        }
        WebOptionField::UserAgent => {
            let ua = expect_string_scalar(
                value,
                "weboptions: UserAgent must be a character vector or string scalar",
                &WEBOPTIONS_ERROR_INVALID_OPTION_VALUE,
            )?;
            options.set(field, Value::from(ua));
            Ok(())
        }
        WebOptionField::Username => {
            let username = expect_string_scalar(
                value,
                "weboptions: Username must be a character vector or string scalar",
                &WEBOPTIONS_ERROR_INVALID_OPTION_VALUE,
            )?;
            options.set(field, Value::from(username));
            Ok(())
        }
        WebOptionField::Password => {
            let password = expect_string_scalar(
                value,
                "weboptions: Password must be a character vector or string scalar",
                &WEBOPTIONS_ERROR_INVALID_OPTION_VALUE,
            )?;
            options.set(field, Value::from(password));
            Ok(())
        }
        WebOptionField::RequestMethod => {
            let method = parse_request_method_option(value)?;
            options.set(field, Value::from(method));
            Ok(())
        }
        WebOptionField::MediaType => {
            let media = expect_string_scalar(
                value,
                "weboptions: MediaType must be a character vector or string scalar",
                &WEBOPTIONS_ERROR_INVALID_OPTION_VALUE,
            )?;
            options.set(field, Value::from(media));
            Ok(())
        }
        WebOptionField::QueryParameters => {
            let qp = canonical_query_parameters(value)?;
            options.set(field, qp);
            Ok(())
        }
    }
}

fn parse_content_type_option(value: &Value) -> BuiltinResult<String> {
    let text = expect_string_scalar(
        value,
        "weboptions: ContentType must be a character vector or string scalar",
        &WEBOPTIONS_ERROR_INVALID_OPTION_VALUE,
    )?;
    match text.trim().to_ascii_lowercase().as_str() {
        "auto" => Ok("auto".to_string()),
        "json" => Ok("json".to_string()),
        "text" | "char" | "string" => Ok("text".to_string()),
        "binary" | "raw" | "octet-stream" => Ok("binary".to_string()),
        other => Err(weboptions_error_with(
            &WEBOPTIONS_ERROR_INVALID_OPTION_VALUE,
            format!(
                "weboptions: unsupported ContentType '{}'; use 'auto', 'json', 'text', or 'binary'",
                other
            ),
        )),
    }
}

fn parse_request_method_option(value: &Value) -> BuiltinResult<String> {
    let text = expect_string_scalar(
        value,
        "weboptions: RequestMethod must be a character vector or string scalar",
        &WEBOPTIONS_ERROR_INVALID_OPTION_VALUE,
    )?;
    let lower = text.trim().to_ascii_lowercase();
    match lower.as_str() {
        "auto" | "get" | "post" | "put" | "patch" | "delete" => Ok(lower),
        _ => Err(weboptions_error_with(
            &WEBOPTIONS_ERROR_INVALID_OPTION_VALUE,
            format!(
                "weboptions: unsupported RequestMethod '{}'; expected auto, get, post, put, patch, or delete",
                text
            ),
        )),
    }
}

fn canonical_header_fields(value: &Value) -> BuiltinResult<Value> {
    match value {
        Value::Struct(struct_value) => {
            let mut out = runmat_value::UnboundedStructValueBuilder::new();
            for (key, val) in &struct_value.fields {
                let header_value = expect_string_scalar(
                    val,
                    "weboptions: HeaderFields values must be character vectors or string scalars",
                    &WEBOPTIONS_ERROR_INVALID_OPTION_VALUE,
                )?;
                if header_value.trim().is_empty() {
                    return Err(weboptions_error_with(
                        &WEBOPTIONS_ERROR_INVALID_OPTION_VALUE,
                        "weboptions: header values must not be empty",
                    ));
                }
                if key.trim().is_empty() {
                    return Err(weboptions_error_with(
                        &WEBOPTIONS_ERROR_INVALID_OPTION_VALUE,
                        "weboptions: header names must not be empty",
                    ));
                }
                out.insert(key, Value::from(header_value));
            }
            Ok(Value::Struct(out.finish()))
        }
        Value::Cell(cell) => {
            if cell.cols != 2 {
                return Err(weboptions_error_with(
                    &WEBOPTIONS_ERROR_INVALID_OPTION_VALUE,
                    "weboptions: HeaderFields cell array must have exactly two columns",
                ));
            }
            let mut out = runmat_value::UnboundedStructValueBuilder::new();
            for row in 0..cell.rows {
                let name_val = cell.get(row, 0).map_err(|err| {
                    weboptions_error_with(
                        &WEBOPTIONS_ERROR_INVALID_OPTION_VALUE,
                        format!("weboptions: {err}"),
                    )
                })?;
                let value_val = cell.get(row, 1).map_err(|err| {
                    weboptions_error_with(
                        &WEBOPTIONS_ERROR_INVALID_OPTION_VALUE,
                        format!("weboptions: {err}"),
                    )
                })?;

                let name = expect_string_scalar(
                    &name_val,
                    "weboptions: header names must be character vectors or string scalars",
                    &WEBOPTIONS_ERROR_INVALID_OPTION_VALUE,
                )?;
                if name.trim().is_empty() {
                    return Err(weboptions_error_with(
                        &WEBOPTIONS_ERROR_INVALID_OPTION_VALUE,
                        "weboptions: header names must not be empty",
                    ));
                }
                let header_value = expect_string_scalar(
                    &value_val,
                    "weboptions: header values must be character vectors or string scalars",
                    &WEBOPTIONS_ERROR_INVALID_OPTION_VALUE,
                )?;
                if header_value.trim().is_empty() {
                    return Err(weboptions_error_with(
                        &WEBOPTIONS_ERROR_INVALID_OPTION_VALUE,
                        "weboptions: header values must not be empty",
                    ));
                }
                out.insert(name, Value::from(header_value));
            }
            Ok(Value::Struct(out.finish()))
        }
        _ => Err(weboptions_error_with(
            &WEBOPTIONS_ERROR_INVALID_OPTION_VALUE,
            "weboptions: HeaderFields must be a struct or two-column cell array",
        )),
    }
}

fn canonical_query_parameters(value: &Value) -> BuiltinResult<Value> {
    match value {
        Value::Struct(struct_value) => {
            let mut out = runmat_value::UnboundedStructValueBuilder::new();
            for (key, val) in &struct_value.fields {
                out.insert(key, val.clone());
            }
            Ok(Value::Struct(out.finish()))
        }
        Value::Cell(cell) => {
            if cell.cols != 2 {
                return Err(weboptions_error_with(
                    &WEBOPTIONS_ERROR_INVALID_OPTION_VALUE,
                    "weboptions: QueryParameters cell array must have exactly two columns",
                ));
            }
            let mut out = runmat_value::UnboundedStructValueBuilder::new();
            for row in 0..cell.rows {
                let name_val = cell.get(row, 0).map_err(|err| {
                    weboptions_error_with(
                        &WEBOPTIONS_ERROR_INVALID_OPTION_VALUE,
                        format!("weboptions: {err}"),
                    )
                })?;
                let value_val = cell.get(row, 1).map_err(|err| {
                    weboptions_error_with(
                        &WEBOPTIONS_ERROR_INVALID_OPTION_VALUE,
                        format!("weboptions: {err}"),
                    )
                })?;
                let name = expect_string_scalar(
                    &name_val,
                    "weboptions: query parameter names must be character vectors or string scalars",
                    &WEBOPTIONS_ERROR_INVALID_OPTION_VALUE,
                )?;
                out.insert(name, value_val);
            }
            Ok(Value::Struct(out.finish()))
        }
        _ => Err(weboptions_error_with(
            &WEBOPTIONS_ERROR_INVALID_OPTION_VALUE,
            "weboptions: QueryParameters must be a struct or two-column cell array",
        )),
    }
}

pub(super) fn validate_credentials(options: &StructValue) -> BuiltinResult<()> {
    let username = string_field(options, "Username")?.unwrap_or_default();
    let password = string_field(options, "Password")?.unwrap_or_default();
    if !password.trim().is_empty() && username.trim().is_empty() {
        return Err(weboptions_error_with(
            &WEBOPTIONS_ERROR_INVALID_CREDENTIALS,
            "weboptions: Password requires a Username option",
        ));
    }
    Ok(())
}

fn string_field(options: &StructValue, field: &str) -> BuiltinResult<Option<String>> {
    let Some(value) = options.fields.get(field) else {
        return Ok(None);
    };
    match value {
        Value::String(runmat_value::StringElement::Text(text)) => {
            Ok(Some(text.as_str().to_owned()))
        }
        Value::CharArray(ca) if ca.rows == 1 => {
            crate::builtins::io::utf16::decode_utf16_with_error(ca.data(), || {
                weboptions_error_with(
                    &WEBOPTIONS_ERROR_INVALID_OPTION_VALUE,
                    format!("weboptions: {field} contains malformed UTF-16 text"),
                )
            })
            .map(Some)
        }
        Value::StringArray(sa) if sa.data().len() == 1 && !sa.data()[0].is_missing() => Ok(Some(
            sa.data()[0]
                .text_ref()
                .expect("checked nonmissing scalar string")
                .as_str()
                .to_owned(),
        )),
        _ => Ok(None),
    }
}

fn numeric_scalar(
    value: &Value,
    context: &str,
    error: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<f64> {
    match value {
        Value::Num(n) => Ok(*n),
        Value::Int(i) => Ok(i.to_f64()),
        Value::Tensor(tensor) => {
            if tensor::is_scalar_tensor(tensor) {
                Ok(tensor::tensor_value_f64(tensor, 0))
            } else {
                Err(weboptions_error_with(error, context))
            }
        }
        _ => Err(weboptions_error_with(error, context)),
    }
}

pub(super) fn expect_string_scalar(
    value: &Value,
    context: &str,
    error: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<String> {
    match value {
        Value::String(runmat_value::StringElement::Text(s)) => Ok(s.as_str().to_owned()),
        Value::CharArray(ca) if ca.rows == 1 => {
            crate::builtins::io::utf16::decode_utf16_with_error(ca.data(), || {
                weboptions_error_with(error, context)
            })
        }
        Value::StringArray(sa) if sa.data().len() == 1 && !sa.data()[0].is_missing() => Ok(sa
            .data()[0]
            .as_str()
            .expect("nonmissing scalar text")
            .to_owned()),
        _ => Err(weboptions_error_with(error, context)),
    }
}
