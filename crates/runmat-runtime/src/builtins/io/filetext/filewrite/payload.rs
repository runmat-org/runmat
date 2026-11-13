use super::options::FileEncoding;
use super::{
    filewrite_error_with_detail, FILEWRITE_ERROR_ENCODE, FILEWRITE_ERROR_INTERNAL,
    FILEWRITE_ERROR_INVALID_DATA,
};
use crate::builtins::common::tensor as tensor_utils;
use crate::BuiltinResult;
use runmat_value::{CharArray, IntValue, LogicalArray, StringArray, Tensor, Utf16CodeUnit, Value};

enum Payload {
    Text(String),
    Bytes(Vec<u8>),
}

pub(super) fn prepare_payload(data: &Value, encoding: FileEncoding) -> BuiltinResult<Vec<u8>> {
    let payload = extract_payload(data)?;
    match payload {
        Payload::Text(chars) => encode_text(chars, encoding),
        Payload::Bytes(bytes) => encode_bytes(bytes, encoding),
    }
}

fn extract_payload(data: &Value) -> BuiltinResult<Payload> {
    match data {
        Value::CharArray(ca) => Ok(Payload::Text(char_array_to_text(ca)?)),
        Value::String(runmat_value::StringElement::Text(s)) => {
            Ok(Payload::Text(s.as_str().to_owned()))
        }
        Value::StringArray(sa) => Ok(Payload::Text(string_array_to_text(sa)?)),
        Value::Num(n) => {
            let byte = float_to_byte(*n)?;
            Ok(Payload::Bytes(vec![byte]))
        }
        Value::Int(i) => {
            let byte = int_value_to_byte(i)?;
            Ok(Payload::Bytes(vec![byte]))
        }
        Value::Bool(flag) => Ok(Payload::Bytes(vec![if *flag { 1 } else { 0 }])),
        Value::Tensor(t) => Ok(Payload::Bytes(tensor_to_bytes(t)?)),
        Value::LogicalArray(la) => Ok(Payload::Bytes(logical_to_bytes(la))),
        Value::Cell(_) => Err(filewrite_error_with_detail(
            &FILEWRITE_ERROR_INVALID_DATA,
            "cell arrays are not supported inputs",
        )),
        Value::GpuTensor(_) => Err(filewrite_error_with_detail(
            &FILEWRITE_ERROR_INTERNAL,
            "GPU tensor should be gathered before filewrite payload extraction",
        )),
        Value::Complex(_, _) | Value::ComplexTensor(_) => Err(filewrite_error_with_detail(
            &FILEWRITE_ERROR_INVALID_DATA,
            "complex data must be converted to text before writing",
        )),
        other => Err(filewrite_error_with_detail(
            &FILEWRITE_ERROR_INVALID_DATA,
            format!("unsupported data type {other:?}; expected text or uint8-compatible array"),
        )),
    }
}

fn char_array_to_text(ca: &CharArray) -> BuiltinResult<String> {
    if ca.rows <= 1 {
        return crate::builtins::io::utf16::decode_utf16_with_error(ca.data(), || {
            filewrite_error_with_detail(
                &FILEWRITE_ERROR_INVALID_DATA,
                "character input contains malformed UTF-16 text",
            )
        });
    }
    let mut out = Vec::with_capacity(ca.rows * (ca.cols + 1));
    for row in 0..ca.rows {
        for col in 0..ca.cols {
            out.push(ca.data()[row * ca.cols + col]);
        }
        if row + 1 < ca.rows {
            out.push(Utf16CodeUnit::from_raw(b'\n' as u16));
        }
    }
    crate::builtins::io::utf16::decode_utf16_with_error(&out, || {
        filewrite_error_with_detail(
            &FILEWRITE_ERROR_INVALID_DATA,
            "character input contains malformed UTF-16 text",
        )
    })
}

fn string_array_to_text(sa: &StringArray) -> BuiltinResult<String> {
    if sa.data().is_empty() {
        return Ok(String::new());
    }
    let mut combined = String::new();
    for (idx, entry) in sa.data().iter().enumerate() {
        combined.push_str(entry.as_str().ok_or_else(|| {
            filewrite_error_with_detail(
                &FILEWRITE_ERROR_INVALID_DATA,
                "missing string cannot be encoded as plain text",
            )
        })?);
        if idx + 1 < sa.data().len() {
            combined.push('\n');
        }
    }
    Ok(combined)
}

pub(super) fn tensor_to_bytes(tensor: &Tensor) -> BuiltinResult<Vec<u8>> {
    let len = tensor_utils::tensor_element_len(tensor);
    let mut out = Vec::with_capacity(len);
    if let Some(storage) = tensor.integer_storage() {
        for idx in 0..len {
            let value = storage
                .value_at(idx)
                .expect("integer storage mirrors tensor shape");
            match int_value_to_byte(&value) {
                Ok(byte) => out.push(byte),
                Err(msg) => {
                    return Err(filewrite_error_with_detail(
                        &FILEWRITE_ERROR_INVALID_DATA,
                        format!("numeric element {} {}", idx, msg),
                    ));
                }
            }
        }
        return Ok(out);
    }
    let values = tensor_utils::tensor_values_f64_cow(tensor);
    for (idx, value) in values.iter().copied().enumerate() {
        match float_to_byte(value) {
            Ok(byte) => out.push(byte),
            Err(msg) => {
                return Err(filewrite_error_with_detail(
                    &FILEWRITE_ERROR_INVALID_DATA,
                    format!("numeric element {} {}", idx, msg),
                ));
            }
        }
    }
    Ok(out)
}

fn logical_to_bytes(array: &LogicalArray) -> Vec<u8> {
    array
        .data
        .iter()
        .map(|value| if *value != 0 { 1 } else { 0 })
        .collect()
}

fn float_to_byte(value: f64) -> BuiltinResult<u8> {
    if !value.is_finite() {
        return Err(filewrite_error_with_detail(
            &FILEWRITE_ERROR_INVALID_DATA,
            format!("value {value} is not finite; cannot write as raw byte"),
        ));
    }
    let rounded = value.round();
    if (value - rounded).abs() > 1e-9 {
        return Err(filewrite_error_with_detail(
            &FILEWRITE_ERROR_INVALID_DATA,
            format!("value {value} is not an integer in the range 0..255"),
        ));
    }
    let int = rounded as i128;
    if !(0..=255).contains(&int) {
        return Err(filewrite_error_with_detail(
            &FILEWRITE_ERROR_INVALID_DATA,
            format!("value {value} is not in the range 0..255"),
        ));
    }
    Ok(int as u8)
}

fn int_value_to_byte(value: &IntValue) -> BuiltinResult<u8> {
    match value {
        IntValue::I8(v) => signed_to_byte(*v as i64),
        IntValue::I16(v) => signed_to_byte(*v as i64),
        IntValue::I32(v) => signed_to_byte(*v as i64),
        IntValue::I64(v) => signed_to_byte(*v),
        IntValue::U8(v) => Ok(*v),
        IntValue::U16(v) => unsigned_to_byte(*v as u64),
        IntValue::U32(v) => unsigned_to_byte(*v as u64),
        IntValue::U64(v) => unsigned_to_byte(*v),
    }
}

fn signed_to_byte(value: i64) -> BuiltinResult<u8> {
    if !(0..=255).contains(&value) {
        return Err(filewrite_error_with_detail(
            &FILEWRITE_ERROR_INVALID_DATA,
            format!("value {value} is not in the range 0..255"),
        ));
    }
    Ok(value as u8)
}

fn unsigned_to_byte(value: u64) -> BuiltinResult<u8> {
    if value > 255 {
        return Err(filewrite_error_with_detail(
            &FILEWRITE_ERROR_INVALID_DATA,
            format!("value {value} is not in the range 0..255"),
        ));
    }
    Ok(value as u8)
}

fn encode_text(text: String, encoding: FileEncoding) -> BuiltinResult<Vec<u8>> {
    match encoding {
        FileEncoding::Auto | FileEncoding::Utf8 => Ok(text.into_bytes()),
        FileEncoding::Ascii => encode_ascii_chars(&text),
        FileEncoding::Latin1 | FileEncoding::Raw => encode_latin_chars(&text, encoding),
    }
}

fn encode_ascii_chars(text: &str) -> BuiltinResult<Vec<u8>> {
    let mut out = Vec::with_capacity(text.len());
    for ch in text.chars() {
        if ch as u32 > 0x7F {
            return Err(filewrite_error_with_detail(
                &FILEWRITE_ERROR_ENCODE,
                format!(
                    "character '{}' (U+{:04X}) cannot be encoded as ASCII",
                    ch, ch as u32
                ),
            ));
        }
        out.push(ch as u8);
    }
    Ok(out)
}

fn encode_latin_chars(text: &str, encoding: FileEncoding) -> BuiltinResult<Vec<u8>> {
    let mut out = Vec::with_capacity(text.len());
    for ch in text.chars() {
        if ch as u32 > 0xFF {
            return Err(filewrite_error_with_detail(
                &FILEWRITE_ERROR_ENCODE,
                format!(
                    "character '{}' (U+{:04X}) cannot be encoded as {}",
                    ch,
                    ch as u32,
                    match encoding {
                        FileEncoding::Latin1 => "Latin-1",
                        FileEncoding::Raw => "raw bytes",
                        _ => unreachable!(),
                    }
                ),
            ));
        }
        out.push(ch as u8);
    }
    Ok(out)
}

fn encode_bytes(bytes: Vec<u8>, encoding: FileEncoding) -> BuiltinResult<Vec<u8>> {
    if matches!(encoding, FileEncoding::Ascii) {
        for (idx, byte) in bytes.iter().enumerate() {
            if *byte > 0x7F {
                return Err(filewrite_error_with_detail(
                    &FILEWRITE_ERROR_ENCODE,
                    format!("byte 0x{byte:02X} at index {idx} cannot be encoded as ASCII"),
                ));
            }
        }
    }
    Ok(bytes)
}
