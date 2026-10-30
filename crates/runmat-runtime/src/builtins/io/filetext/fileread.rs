//! MATLAB-compatible `fileread` builtin for RunMat.

use std::path::{Path, PathBuf};

use runmat_builtins::{
    BuiltinErrorDescriptor, FILEREAD_ERROR_DECODE, FILEREAD_ERROR_INTERNAL,
    FILEREAD_ERROR_INVALID_INPUT, FILEREAD_ERROR_INVALID_OPTION, FILEREAD_ERROR_IO,
};
#[cfg(test)]
use runmat_builtins::{BuiltinIntegerAuditKind, FILEREAD_DESCRIPTOR, FILEREAD_INTEGER_AUDIT};
use runmat_macros::runtime_builtin;
use runmat_value::{CharArray, Value};

use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_filesystem as fs;

const BUILTIN_NAME: &str = "fileread";

fn fileread_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    fileread_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn fileread_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn fileread_error_with_source(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
    source: impl crate::runtime_error::RuntimeErrorSourceBound,
) -> RuntimeError {
    let mut builder = build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .with_source(source);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FileEncoding {
    Auto,
    Utf8,
    Ascii,
    Latin1,
    Raw,
}

impl FileEncoding {
    fn from_label(label: &str) -> Option<Self> {
        let trimmed = label.trim();
        if trimmed.is_empty() {
            return None;
        }
        match trimmed.to_ascii_lowercase().as_str() {
            "auto" | "default" | "system" | "native" => Some(FileEncoding::Auto),
            "utf-8" | "utf8" | "utf_8" | "unicode" => Some(FileEncoding::Utf8),
            "ascii" | "us-ascii" | "us_ascii" | "usascii" => Some(FileEncoding::Ascii),
            "latin1" | "latin-1" | "latin_1" | "iso-8859-1" | "iso8859-1" | "iso88591" => {
                Some(FileEncoding::Latin1)
            }
            "raw" | "bytes" | "byte" | "binary" => Some(FileEncoding::Raw),
            _ => None,
        }
    }
}

#[runtime_builtin(
    name = "fileread",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::filetext::fileread"
)]
async fn fileread_builtin(
    path: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    let encoding = parse_encoding_args(rest.view())?;
    let resolved = resolve_path(&path)?;
    let bytes = read_all(&resolved).await?;
    let chars = decode_bytes(bytes, encoding)?;
    let cols = chars.iter().map(|value| value.len_utf16()).sum();
    let char_array = CharArray::from_unicode_scalars(chars, 1, cols)
        .map_err(|e| fileread_error_with_detail(&FILEREAD_ERROR_INTERNAL, &e))?;
    Ok(Value::CharArray(char_array))
}

fn parse_encoding_args(args: crate::call::FunctionArgumentView<'_>) -> BuiltinResult<FileEncoding> {
    match args.len() {
        0 => Ok(FileEncoding::Auto),
        1 => {
            if is_encoding_keyword(&args[0])? {
                return Err(fileread_error_with_detail(
                    &FILEREAD_ERROR_INVALID_OPTION,
                    "missing encoding value after 'Encoding' keyword",
                ));
            }
            encoding_from_value(&args[0])
        }
        2 => {
            if !is_encoding_keyword(&args[0])? {
                return Err(fileread_error_with_detail(
                    &FILEREAD_ERROR_INVALID_OPTION,
                    "expected 'Encoding' keyword before encoding value",
                ));
            }
            encoding_from_value(&args[1])
        }
        _ => Err(fileread_error_with_detail(
            &FILEREAD_ERROR_INVALID_INPUT,
            "too many input arguments",
        )),
    }
}

fn encoding_from_value(value: &Value) -> BuiltinResult<FileEncoding> {
    let label = encoding_name(value)?;
    match FileEncoding::from_label(&label) {
        Some(enc) => Ok(enc),
        None => {
            if label.trim().is_empty() {
                Err(fileread_error_with_detail(
                    &FILEREAD_ERROR_INVALID_OPTION,
                    "encoding name must not be empty",
                ))
            } else {
                Err(fileread_error_with_detail(
                    &FILEREAD_ERROR_INVALID_OPTION,
                    format!("unsupported encoding '{}'", label),
                ))
            }
        }
    }
}

fn is_encoding_keyword(value: &Value) -> BuiltinResult<bool> {
    let text = encoding_name(value)?;
    Ok(text.eq_ignore_ascii_case("encoding"))
}

fn encoding_name(value: &Value) -> BuiltinResult<std::borrow::Cow<'_, str>> {
    match value {
        Value::String(runmat_value::StringElement::Text(s)) => {
            Ok(std::borrow::Cow::Borrowed(s.as_str()))
        }
        Value::CharArray(ca) if ca.rows == 1 => runmat_value::decode_utf16_code_units(ca.data())
            .map(std::borrow::Cow::Owned)
            .map_err(|source| {
                fileread_error_with_detail(&FILEREAD_ERROR_INVALID_OPTION, source.to_string())
            }),
        Value::CharArray(_) => Err(fileread_error_with_detail(
            &FILEREAD_ERROR_INVALID_OPTION,
            "encoding name must be a 1-by-N character vector",
        )),
        Value::StringArray(sa) => {
            if sa.data().len() == 1 && !sa.data()[0].is_missing() {
                Ok(std::borrow::Cow::Borrowed(
                    sa.data()[0].as_str().expect("checked nonmissing string"),
                ))
            } else {
                Err(fileread_error_with_detail(
                    &FILEREAD_ERROR_INVALID_OPTION,
                    "encoding inputs must be scalar string arrays",
                ))
            }
        }
        other => Err(fileread_error_with_detail(
            &FILEREAD_ERROR_INVALID_OPTION,
            format!("expected encoding as string scalar or character vector, got {other:?}"),
        )),
    }
}

fn resolve_path(value: &Value) -> BuiltinResult<PathBuf> {
    match value {
        Value::String(runmat_value::StringElement::Text(s)) => normalize_path(s),
        Value::CharArray(ca) if ca.rows == 1 => runmat_value::decode_utf16_code_units(ca.data())
            .map_err(|source| {
                fileread_error_with_detail(&FILEREAD_ERROR_INVALID_INPUT, source.to_string())
            })
            .and_then(|path| normalize_path(&path)),
        Value::CharArray(_) => Err(fileread_error_with_detail(
            &FILEREAD_ERROR_INVALID_INPUT,
            "expected a 1-by-N character vector for the file name",
        )),
        Value::StringArray(sa) => {
            if sa.data().len() == 1 && !sa.data()[0].is_missing() {
                normalize_path(sa.data()[0].as_str().expect("checked nonmissing string"))
            } else {
                Err(fileread_error_with_detail(
                    &FILEREAD_ERROR_INVALID_INPUT,
                    "string array inputs must be scalar",
                ))
            }
        }
        other => Err(fileread_error_with_detail(
            &FILEREAD_ERROR_INVALID_INPUT,
            format!("expected filename as string scalar or character vector, got {other:?}"),
        )),
    }
}

fn normalize_path(raw: &str) -> BuiltinResult<PathBuf> {
    if raw.is_empty() {
        return Err(fileread_error_with_detail(
            &FILEREAD_ERROR_INVALID_INPUT,
            "filename must not be empty",
        ));
    }
    Ok(Path::new(raw).to_path_buf())
}

async fn read_all(path: &Path) -> BuiltinResult<Vec<u8>> {
    fs::read_async(path).await.map_err(|err| {
        fileread_error_with_source(
            format!(
                "{}: unable to read '{}': {}",
                FILEREAD_ERROR_IO.message,
                path.display(),
                err
            ),
            &FILEREAD_ERROR_IO,
            err,
        )
    })
}

fn decode_bytes(bytes: Vec<u8>, encoding: FileEncoding) -> BuiltinResult<Vec<char>> {
    match encoding {
        FileEncoding::Auto => match String::from_utf8(bytes) {
            Ok(text) => Ok(text.chars().collect()),
            Err(err) => Ok(bytes_to_chars(err.into_bytes())),
        },
        FileEncoding::Utf8 => decode_utf8(bytes),
        FileEncoding::Ascii => decode_ascii(bytes),
        FileEncoding::Latin1 | FileEncoding::Raw => Ok(bytes_to_chars(bytes)),
    }
}

fn decode_utf8(bytes: Vec<u8>) -> BuiltinResult<Vec<char>> {
    match String::from_utf8(bytes) {
        Ok(text) => Ok(text.chars().collect()),
        Err(err) => {
            let utf8_err = err.utf8_error();
            let offset = utf8_err.valid_up_to();
            let detail = match utf8_err.error_len() {
                Some(len) => {
                    let plural = if len == 1 { "" } else { "s" };
                    format!(
                        "invalid UTF-8 sequence of {len} byte{plural} starting at offset {offset}"
                    )
                }
                None => format!("incomplete UTF-8 sequence at end of data (after offset {offset})"),
            };
            Err(fileread_error_with_detail(
                &FILEREAD_ERROR_DECODE,
                format!("unable to decode file as UTF-8: {detail}"),
            ))
        }
    }
}

fn decode_ascii(bytes: Vec<u8>) -> BuiltinResult<Vec<char>> {
    for (idx, byte) in bytes.iter().enumerate() {
        if *byte > 0x7F {
            return Err(fileread_error_with_detail(
                &FILEREAD_ERROR_DECODE,
                format!("byte 0x{byte:02X} at offset {idx} is not valid ASCII"),
            ));
        }
    }
    Ok(bytes_to_chars(bytes))
}

fn bytes_to_chars(bytes: Vec<u8>) -> Vec<char> {
    bytes.into_iter().map(char::from).collect()
}

#[cfg(test)]
pub(crate) mod tests;
