use super::*;
use crate::builtins::common::tensor;
use crate::builtins::io::filetext::registry::{self, FileInfo, FileOperation};
use crate::console::{record_console_output, ConsoleStream};
use runmat_builtins::{
    FPRINTF_ERROR_INVALID_IDENTIFIER, FPRINTF_ERROR_INVALID_INPUT, FPRINTF_ERROR_IO,
};
use std::io::Write;
#[derive(Clone, Copy)]
pub(super) enum SpecialStream {
    Stdout,
    Stderr,
}

pub(super) enum OutputTarget {
    Stdout,
    Stderr,
    File {
        operation: Box<FileOperation>,
        encoding: String,
    },
}

impl OutputTarget {
    pub(super) fn encoding_label(&self) -> Option<&str> {
        match self {
            OutputTarget::Stdout | OutputTarget::Stderr => None,
            OutputTarget::File { encoding, .. } => Some(encoding.as_str()),
        }
    }

    pub(super) fn write(
        &mut self,
        runtime: Option<&crate::context::RuntimeContext>,
        bytes: &[u8],
    ) -> BuiltinResult<()> {
        match self {
            OutputTarget::Stdout => record_console_chunk(ConsoleStream::Stdout, bytes),
            OutputTarget::Stderr => record_console_chunk(ConsoleStream::Stderr, bytes),
            OutputTarget::File { operation, .. } => {
                operation.check_authority(runtime).map_err(|source| {
                    fprintf_error_with_source(
                        format!("fprintf: file authority invalidated ({source})"),
                        &FPRINTF_ERROR_IO,
                        source,
                    )
                })?;
                operation.file_mut().write_all(bytes).map_err(|err| {
                    fprintf_error_with_source(
                        format!(
                            "{}: failed to write to file ({err})",
                            FPRINTF_ERROR_IO.message
                        ),
                        &FPRINTF_ERROR_IO,
                        err,
                    )
                })?;
                operation.check_authority(runtime).map_err(|source| {
                    fprintf_error_with_source(
                        format!("fprintf: file authority invalidated ({source})"),
                        &FPRINTF_ERROR_IO,
                        source,
                    )
                })
            }
        }
    }
}

fn record_console_chunk(stream: ConsoleStream, bytes: &[u8]) -> BuiltinResult<()> {
    if bytes.is_empty() {
        return Ok(());
    }
    let text = String::from_utf8_lossy(bytes).to_string();
    record_console_output(stream, text)
}

pub(super) fn target_from_fid(
    runtime: Option<&crate::context::RuntimeContext>,
    fid: i32,
) -> BuiltinResult<OutputTarget> {
    if fid < 0 {
        return Err(fprintf_error_with_detail(
            &FPRINTF_ERROR_INVALID_INPUT,
            "file identifier must be non-negative",
        ));
    }
    match fid {
        0 => Err(fprintf_error_with_detail(
            &FPRINTF_ERROR_INVALID_IDENTIFIER,
            "file identifier 0 (stdin) is not writable",
        )),
        1 => Ok(OutputTarget::Stdout),
        2 => Ok(OutputTarget::Stderr),
        _ => {
            let operation = registry::acquire_file_in(fid, runtime).map_err(|err| match err {
                registry::FileOperationError::InvalidIdentifier
                | registry::FileOperationError::ForeignSession => fprintf_error_with_message(
                    FPRINTF_ERROR_INVALID_IDENTIFIER.message,
                    &FPRINTF_ERROR_INVALID_IDENTIFIER,
                ),
                other => {
                    let source = other.into_io_error();
                    fprintf_error_with_source(
                        format!("fprintf: file identifier is unavailable ({source})"),
                        &FPRINTF_ERROR_IO,
                        source,
                    )
                }
            })?;
            ensure_writable(&operation.info)?;
            let encoding = operation.info.encoding.clone();
            Ok(OutputTarget::File {
                operation: Box::new(operation),
                encoding,
            })
        }
    }
}

pub(super) fn parse_fid(value: &Value) -> Result<i32, String> {
    let scalar = match value {
        Value::Num(n) => *n,
        Value::Int(int) => {
            return int
                .try_to_i32()
                .ok_or_else(|| "fprintf: file identifier is out of range".to_string());
        }
        Value::Tensor(t) => {
            if tensor::is_scalar_tensor(t) {
                if let Some(int) = t.integer_storage().and_then(|storage| storage.value_at(0)) {
                    return int
                        .try_to_i32()
                        .ok_or_else(|| "fprintf: file identifier is out of range".to_string());
                }
                tensor::tensor_value_f64(t, 0)
            } else {
                return Err("fprintf: file identifier must be numeric".to_string());
            }
        }
        _ => return Err("fprintf: file identifier must be numeric".to_string()),
    };
    if !scalar.is_finite() {
        return Err("fprintf: file identifier must be finite".to_string());
    }
    if (scalar.fract().abs()) > f64::EPSILON {
        return Err("fprintf: file identifier must be an integer".to_string());
    }
    if scalar < i32::MIN as f64 || scalar > i32::MAX as f64 {
        return Err("fprintf: file identifier is out of range".to_string());
    }
    Ok(scalar as i32)
}

fn ensure_writable(info: &FileInfo) -> BuiltinResult<()> {
    if info.access.write {
        Ok(())
    } else {
        Err(fprintf_error_with_detail(
            &FPRINTF_ERROR_INVALID_IDENTIFIER,
            "file is not open for writing",
        ))
    }
}

pub(super) fn match_stream_label(value: &Value) -> BuiltinResult<Option<SpecialStream>> {
    let candidate = match value {
        Value::String(runmat_value::StringElement::Text(s)) => s.trim().to_string(),
        Value::CharArray(ca) if ca.rows == 1 => {
            crate::builtins::io::utf16::decode_utf16_with_error(ca.data(), || {
                fprintf_error_with_detail(
                    &FPRINTF_ERROR_INVALID_INPUT,
                    "fprintf: stream label contains malformed UTF-16 text",
                )
            })?
            .trim()
            .to_string()
        }
        Value::StringArray(sa) if sa.data().len() == 1 && !sa.data()[0].is_missing() => sa.data()
            [0]
        .text_ref()
        .expect("checked nonmissing scalar string")
        .trim()
        .to_string(),
        _ => return Ok(None),
    };
    Ok(match candidate.to_ascii_lowercase().as_str() {
        "stdout" => Some(SpecialStream::Stdout),
        "stderr" => Some(SpecialStream::Stderr),
        _ => None,
    })
}
