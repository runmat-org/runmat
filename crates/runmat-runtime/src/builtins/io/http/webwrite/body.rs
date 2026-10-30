use super::*;
pub(super) async fn prepare_request_body(
    data: Value,
    options: &WebWriteOptions,
) -> BuiltinResult<PreparedBody> {
    let format = match options.request_format {
        RequestFormat::Auto => guess_request_format(&data),
        set => set,
    };
    let content_type = options
        .media_type
        .clone()
        .or_else(|| default_content_type_for(format));
    let bytes = match format {
        RequestFormat::Form => encode_form_payload(&data)?,
        RequestFormat::Json => encode_json_payload(&data).await?,
        RequestFormat::Text => encode_text_payload(&data)?,
        RequestFormat::Binary => encode_binary_payload(&data)?,
        RequestFormat::Auto => encode_json_payload(&data).await?,
    };
    Ok(PreparedBody {
        bytes,
        content_type,
    })
}

pub(super) fn encode_form_payload(value: &Value) -> BuiltinResult<Vec<u8>> {
    let mut pairs = Vec::new();
    match value {
        Value::Struct(struct_value) => {
            for (key, val) in &struct_value.fields {
                let text = value_to_query_string(val, key)?;
                pairs.push((key.to_owned(), text));
            }
        }
        Value::Cell(cell) => {
            append_query_from_cell(cell, &mut pairs)?;
        }
        Value::CharArray(_)
        | Value::String(_)
        | Value::Num(_)
        | Value::Int(_)
        | Value::Tensor(_) => {
            // Allow scalar text/numeric by mapping to a default "data" key.
            let text = scalar_to_string(value)?;
            pairs.push(("data".to_string(), text));
        }
        _ => {
            return Err(webwrite_error(
                "webwrite: form payloads must be structs, two-column cell arrays, or scalars",
            ))
        }
    }

    let encoded = encode_form_pairs(&pairs);
    Ok(encoded.into_bytes())
}

pub(super) fn encode_form_pairs(pairs: &[(String, String)]) -> String {
    let mut result = String::new();
    for (idx, (name, value)) in pairs.iter().enumerate() {
        if idx > 0 {
            result.push('&');
        }
        result.push_str(&url_encode_component(name));
        result.push('=');
        result.push_str(&url_encode_component(value));
    }
    result
}

pub(super) fn url_encode_component(input: &str) -> String {
    let mut out = String::new();
    for byte in input.bytes() {
        match byte {
            b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'*' => {
                out.push(byte as char);
            }
            b' ' => out.push('+'),
            _ => {
                out.push('%');
                out.push(hex_digit(byte >> 4));
                out.push(hex_digit(byte & 0xF));
            }
        }
    }
    out
}

pub(super) fn hex_digit(nibble: u8) -> char {
    match nibble {
        0..=9 => (b'0' + nibble) as char,
        10..=15 => (b'A' + (nibble - 10)) as char,
        _ => unreachable!(),
    }
}

pub(super) async fn encode_json_payload(value: &Value) -> BuiltinResult<Vec<u8>> {
    let encoded = call_builtin_async("jsonencode", std::slice::from_ref(value))
        .await
        .map_err(|flow| {
            remap_webwrite_flow(&WEBWRITE_ERROR_FLOW, flow, |err| {
                format!("webwrite: {}", err.message())
            })
        })?;
    let text = expect_string_scalar(
        &encoded,
        "webwrite: jsonencode returned unexpected value; expected text scalar",
    )?;
    Ok(text.into_bytes())
}

pub(super) fn encode_text_payload(value: &Value) -> BuiltinResult<Vec<u8>> {
    let text = scalar_to_string(value)?;
    Ok(text.into_bytes())
}

pub(super) fn encode_binary_payload(value: &Value) -> BuiltinResult<Vec<u8>> {
    match value {
        Value::Tensor(tensor) => tensor_f64_to_bytes(tensor),
        Value::Num(n) => Ok(vec![float_to_byte(*n)?]),
        Value::Int(i) => Ok(vec![int_to_byte(i.to_i64())?]),
        Value::Bool(b) => Ok(vec![if *b { 1 } else { 0 }]),
        Value::LogicalArray(array) => Ok(array.data.to_vec()),
        Value::CharArray(ca) => {
            let mut bytes = Vec::with_capacity(ca.data().len());
            for ch in ca.data() {
                let code = ch.raw() as u32;
                if code > 0xFF {
                    return Err(webwrite_error(
                        "webwrite: character codes exceed 255 for binary payload",
                    ));
                }
                bytes.push(code as u8);
            }
            Ok(bytes)
        }
        Value::String(runmat_value::StringElement::Text(s)) => Ok(s.as_bytes().to_vec()),
        Value::StringArray(sa) => {
            if sa.data().len() == 1 && !sa.data()[0].is_missing() {
                Ok(sa.data()[0]
                    .as_str()
                    .expect("checked nonmissing scalar string")
                    .as_bytes()
                    .to_vec())
            } else {
                Err(webwrite_error(
                    "webwrite: binary payload string arrays must be scalar",
                ))
            }
        }
        _ => Err(webwrite_error(
            "webwrite: unsupported value for binary payload",
        )),
    }
}

pub(super) fn tensor_f64_to_bytes(tensor: &Tensor) -> BuiltinResult<Vec<u8>> {
    let values = tensor_utils::tensor_values_f64_cow(tensor);
    let mut bytes = Vec::with_capacity(values.len());
    for value in values.iter().copied() {
        bytes.push(float_to_byte(value)?);
    }
    Ok(bytes)
}

pub(super) fn float_to_byte(value: f64) -> BuiltinResult<u8> {
    if !value.is_finite() {
        return Err(webwrite_error(
            "webwrite: binary payload values must be finite",
        ));
    }
    let rounded = value.round();
    if (value - rounded).abs() > 1e-9 {
        return Err(webwrite_error(
            "webwrite: binary payload values must be integers in 0..255",
        ));
    }
    let int_val = rounded as i64;
    int_to_byte(int_val)
}

pub(super) fn int_to_byte(value: i64) -> BuiltinResult<u8> {
    if !(0..=255).contains(&value) {
        return Err(webwrite_error(
            "webwrite: binary payload values must be in the range 0..255",
        ));
    }

    Ok(value as u8)
}
