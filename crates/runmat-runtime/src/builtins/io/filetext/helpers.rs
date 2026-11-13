use std::io::{Read, Seek, SeekFrom};

use encoding_rs::Encoding;
use runmat_value::{CharArray, Tensor, Value};

use crate::{build_runtime_error, BuiltinResult, RuntimeError};

pub(crate) fn extract_scalar_string(value: &Value) -> Option<String> {
    if !crate::builtins::common::validation::value_is_text_scalar(value) {
        return None;
    }
    match value {
        Value::String(runmat_value::StringElement::Text(s)) => Some(s.as_str().to_owned()),
        Value::CharArray(ca) => ca.row_string(),
        Value::StringArray(sa) => sa.data()[0].as_str().map(str::to_owned),
        _ => None,
    }
}

pub(crate) fn char_array_value(text: &str) -> Value {
    Value::CharArray(CharArray::new_row(text))
}

pub(crate) fn normalize_encoding_label(label: &str) -> String {
    label.trim().to_ascii_lowercase()
}

pub(crate) struct LineRead {
    pub(crate) data: Vec<u8>,
    pub(crate) terminators: Vec<u8>,
    pub(crate) eof_before_any: bool,
    pub(crate) encountered_eof: bool,
}

pub(crate) fn read_text_line<R: Read + Seek>(
    file: &mut R,
    limit: Option<usize>,
    builtin_name: &'static str,
) -> BuiltinResult<LineRead> {
    let mut data = Vec::new();
    let mut terminators = Vec::new();
    let mut eof_before_any = false;
    let mut encountered_eof = false;

    let max_bytes = limit.unwrap_or(usize::MAX);
    if max_bytes == 0 {
        return Ok(LineRead {
            data,
            terminators,
            eof_before_any,
            encountered_eof,
        });
    }

    let mut first_attempt = true;
    let mut buffer = [0u8; 1];
    loop {
        if data.len() >= max_bytes {
            break;
        }

        let read = file.read(&mut buffer).map_err(|err| {
            build_runtime_error(format!("{builtin_name}: failed to read from file: {err}"))
                .with_builtin(builtin_name)
                .with_source(err)
                .build()
        })?;
        if read == 0 {
            encountered_eof = true;
            if data.is_empty() && first_attempt {
                eof_before_any = true;
            }
            break;
        }
        first_attempt = false;
        let byte = buffer[0];

        if byte == b'\n' {
            let mut newline = [0u8; 2];
            newline[0] = b'\n';
            let mut newline_len = 1usize;
            let mut consumed = 1i64;

            let mut next = [0u8; 1];
            let read_next = file.read(&mut next).map_err(|err| {
                build_runtime_error(format!("{builtin_name}: failed to read from file: {err}"))
                    .with_builtin(builtin_name)
                    .with_source(err)
                    .build()
            })?;
            if read_next == 0 {
                encountered_eof = true;
            } else if next[0] == b'\r' {
                newline[1] = b'\r';
                newline_len = 2;
                consumed = 2;
            } else {
                seek_current(file, -1, builtin_name)?;
            }

            if data.len().saturating_add(newline_len) > max_bytes {
                seek_current(file, -consumed, builtin_name)?;
                if read_next == 0 {
                    encountered_eof = false;
                }
            } else {
                data.extend_from_slice(&newline[..newline_len]);
                terminators.extend_from_slice(&newline[..newline_len]);
            }
            break;
        } else if byte == b'\r' {
            let mut newline = [0u8; 2];
            newline[0] = b'\r';
            let mut newline_len = 1usize;
            let mut consumed = 1i64;

            let mut next = [0u8; 1];
            let read_next = file.read(&mut next).map_err(|err| {
                build_runtime_error(format!("{builtin_name}: failed to read from file: {err}"))
                    .with_builtin(builtin_name)
                    .with_source(err)
                    .build()
            })?;
            if read_next == 0 {
                encountered_eof = true;
            }
            if read_next > 0 {
                if next[0] == b'\n' {
                    newline[1] = b'\n';
                    newline_len = 2;
                    consumed = 2;
                } else {
                    seek_current(file, -1, builtin_name)?;
                }
            }

            if data.len().saturating_add(newline_len) > max_bytes {
                seek_current(file, -consumed, builtin_name)?;
                if read_next == 0 {
                    encountered_eof = false;
                }
            } else {
                data.extend_from_slice(&newline[..newline_len]);
                terminators.extend_from_slice(&newline[..newline_len]);
            }
            break;
        } else {
            data.push(byte);
        }
    }

    Ok(LineRead {
        data,
        terminators,
        eof_before_any,
        encountered_eof,
    })
}

fn seek_current<R: Seek>(
    file: &mut R,
    offset: i64,
    builtin_name: &'static str,
) -> BuiltinResult<()> {
    file.seek(SeekFrom::Current(offset)).map_err(|err| {
        build_runtime_error(format!("{builtin_name}: failed to seek in file: {err}"))
            .with_builtin(builtin_name)
            .with_source(err)
            .build()
    })?;
    Ok(())
}

pub(crate) fn bytes_to_char_array(
    bytes: &[u8],
    encoding: &str,
    builtin_name: &'static str,
) -> BuiltinResult<Value> {
    let chars = decode_bytes(bytes, encoding, builtin_name)?;
    let cols = chars.iter().map(|value| value.len_utf16()).sum();
    let char_array = CharArray::from_unicode_scalars(chars, 1, cols).map_err(|e| {
        text_io_error(
            builtin_name,
            format!("{builtin_name}: failed to build char array: {e}"),
        )
    })?;
    Ok(Value::CharArray(char_array))
}

pub(crate) fn decode_bytes(
    bytes: &[u8],
    encoding: &str,
    builtin_name: &'static str,
) -> BuiltinResult<Vec<char>> {
    use super::encoding::TextEncoding;
    match TextEncoding::parse(encoding)
        .map_err(|detail| text_io_error(builtin_name, format!("{builtin_name}: {detail}")))?
    {
        TextEncoding::Bytes => Ok(bytes.iter().map(|&byte| char::from(byte)).collect()),
        TextEncoding::Ascii => decode_ascii(bytes, builtin_name),
        TextEncoding::Encoded(encoding) => decode_with_encoding(bytes, encoding, builtin_name),
    }
}

fn decode_with_encoding(
    bytes: &[u8],
    enc: &'static Encoding,
    builtin_name: &'static str,
) -> BuiltinResult<Vec<char>> {
    let (cow, _, had_errors) = enc.decode(bytes);
    if had_errors {
        return Err(text_io_error(
            builtin_name,
            format!(
                "{builtin_name}: unable to decode bytes using encoding '{}'",
                enc.name()
            ),
        ));
    }
    Ok(cow.chars().collect())
}

fn decode_ascii(bytes: &[u8], builtin_name: &'static str) -> BuiltinResult<Vec<char>> {
    if let Some(byte) = bytes.iter().find(|&&b| b > 0x7F) {
        return Err(text_io_error(
            builtin_name,
            format!(
                "{builtin_name}: byte value {} is outside the ASCII range",
                byte
            ),
        ));
    }
    Ok(bytes
        .iter()
        .map(|&b| char::from_u32(b as u32).unwrap())
        .collect())
}

pub(crate) fn numeric_row(bytes: &[u8], builtin_name: &'static str) -> BuiltinResult<Value> {
    let data: Vec<f64> = bytes.iter().map(|&b| b as f64).collect();
    let tensor = Tensor::new(data, vec![1, bytes.len()]).map_err(|e| {
        text_io_error(
            builtin_name,
            format!("{builtin_name}: failed to construct numeric array: {e}"),
        )
    })?;
    Ok(Value::Tensor(tensor))
}

pub(crate) fn empty_numeric_row() -> Value {
    let tensor = Tensor::new(Vec::new(), vec![0, 0]).unwrap_or_else(|_| Tensor::zeros(vec![0, 0]));
    Value::Tensor(tensor)
}

fn text_io_error(builtin_name: &'static str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(builtin_name)
        .build()
}

#[cfg(test)]
mod typed_string_controls {
    use super::*;

    #[test]
    fn scalar_text_controls_reject_missing_without_reserving_any_literal_spelling() {
        for element in [
            "<missing>".into(),
            "<MISSING>".into(),
            "".into(),
            runmat_value::StringElement::Missing,
        ] {
            let expected = element.as_str().map(str::to_owned);
            assert_eq!(
                extract_scalar_string(&Value::String(element.clone())),
                expected
            );
            let array =
                runmat_value::StringArray::from_elements_unbounded(vec![element], vec![1, 1])
                    .unwrap();
            assert_eq!(extract_scalar_string(&Value::StringArray(array)), expected);
        }
    }
}
