use std::io::{Read, Seek, SeekFrom};

use encoding_rs::{Encoding, UTF_8};
use runmat_builtins::{CharArray, Tensor, Value};
use runmat_filesystem::File;

use crate::{build_runtime_error, BuiltinResult, RuntimeError};

pub(crate) fn extract_scalar_string(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::CharArray(ca) if ca.rows == 1 => Some(ca.data.iter().collect()),
        Value::StringArray(sa) if sa.data.len() == 1 => Some(sa.data[0].clone()),
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
}

pub(crate) fn read_text_line(
    file: &mut File,
    limit: Option<usize>,
    builtin_name: &'static str,
) -> BuiltinResult<LineRead> {
    let mut data = Vec::new();
    let mut terminators = Vec::new();
    let mut eof_before_any = false;

    let max_bytes = limit.unwrap_or(usize::MAX);
    if max_bytes == 0 {
        return Ok(LineRead {
            data,
            terminators,
            eof_before_any,
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
            if data.is_empty() && first_attempt {
                eof_before_any = true;
            }
            break;
        }
        first_attempt = false;
        let byte = buffer[0];

        if byte == b'\n' {
            if data.len().saturating_add(1) > max_bytes {
                seek_current(file, -1, builtin_name)?;
            } else {
                data.push(b'\n');
                terminators.push(b'\n');
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
    })
}

fn seek_current(file: &mut File, offset: i64, builtin_name: &'static str) -> BuiltinResult<()> {
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
    let cols = chars.len();
    let char_array = CharArray::new(chars, 1, cols).map_err(|e| {
        text_io_error(
            builtin_name,
            format!("{builtin_name}: failed to build char array: {e}"),
        )
    })?;
    Ok(Value::CharArray(char_array))
}

fn decode_bytes(
    bytes: &[u8],
    encoding: &str,
    builtin_name: &'static str,
) -> BuiltinResult<Vec<char>> {
    let label = encoding.trim();
    if label.is_empty() || label.eq_ignore_ascii_case("utf-8") || label.eq_ignore_ascii_case("utf8")
    {
        return decode_with_encoding(bytes, UTF_8, builtin_name);
    }
    if label.eq_ignore_ascii_case("binary") {
        return Ok(bytes
            .iter()
            .map(|&b| char::from_u32(b as u32).unwrap())
            .collect());
    }
    if label.eq_ignore_ascii_case("latin1")
        || label.eq_ignore_ascii_case("latin-1")
        || label.eq_ignore_ascii_case("iso-8859-1")
    {
        return Ok(bytes
            .iter()
            .map(|&b| char::from_u32(b as u32).unwrap())
            .collect());
    }
    if label.eq_ignore_ascii_case("windows-1252") || label.eq_ignore_ascii_case("cp1252") {
        return decode_with_encoding(bytes, encoding_rs::WINDOWS_1252, builtin_name);
    }
    if label.eq_ignore_ascii_case("shift_jis")
        || label.eq_ignore_ascii_case("shift-jis")
        || label.eq_ignore_ascii_case("sjis")
    {
        return decode_with_encoding(bytes, encoding_rs::SHIFT_JIS, builtin_name);
    }
    if label.eq_ignore_ascii_case("us-ascii")
        || label.eq_ignore_ascii_case("ascii")
        || label.eq_ignore_ascii_case("us_ascii")
        || label.eq_ignore_ascii_case("usascii")
    {
        return decode_ascii(bytes, builtin_name);
    }
    if label.eq_ignore_ascii_case("system") {
        let fallback = system_default_encoding_label();
        if fallback.eq_ignore_ascii_case("binary") {
            return Ok(bytes
                .iter()
                .map(|&b| char::from_u32(b as u32).unwrap())
                .collect());
        }
        return decode_bytes(bytes, fallback, builtin_name);
    }

    if let Some(enc) = Encoding::for_label(label.as_bytes()) {
        return decode_with_encoding(bytes, enc, builtin_name);
    }

    Err(text_io_error(
        builtin_name,
        format!("{builtin_name}: unsupported encoding '{encoding}'"),
    ))
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

fn system_default_encoding_label() -> &'static str {
    #[cfg(windows)]
    {
        "windows-1252"
    }
    #[cfg(not(windows))]
    {
        "utf-8"
    }
}

fn text_io_error(builtin_name: &'static str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(builtin_name)
        .build()
}
