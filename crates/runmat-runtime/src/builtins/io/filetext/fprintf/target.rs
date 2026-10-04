use super::*;
#[derive(Clone, Copy)]
pub(super) enum SpecialStream {
    Stdout,
    Stderr,
}

pub(super) enum OutputTarget {
    Stdout,
    Stderr,
    File {
        handle: SharedFileHandle,
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

    pub(super) fn write(&self, bytes: &[u8]) -> BuiltinResult<()> {
        match self {
            OutputTarget::Stdout => {
                record_console_chunk(ConsoleStream::Stdout, bytes);
                Ok(())
            }
            OutputTarget::Stderr => {
                record_console_chunk(ConsoleStream::Stderr, bytes);
                Ok(())
            }
            OutputTarget::File { handle, .. } => {
                let mut guard = handle.lock().map_err(|_| {
                    fprintf_error_with_detail(
                        &FPRINTF_ERROR_INTERNAL,
                        "failed to lock file handle (poisoned mutex)",
                    )
                })?;
                let file = guard.as_mut().ok_or_else(|| {
                    fprintf_error_with_message(
                        FPRINTF_ERROR_INVALID_IDENTIFIER.message,
                        &FPRINTF_ERROR_INVALID_IDENTIFIER,
                    )
                })?;
                file.write_all(bytes).map_err(|err| {
                    fprintf_error_with_source(
                        format!(
                            "{}: failed to write to file ({err})",
                            FPRINTF_ERROR_IO.message
                        ),
                        &FPRINTF_ERROR_IO,
                        err,
                    )
                })
            }
        }
    }
}

fn record_console_chunk(stream: ConsoleStream, bytes: &[u8]) {
    if bytes.is_empty() {
        return;
    }
    let text = String::from_utf8_lossy(bytes).to_string();
    record_console_output(stream, text);
}

pub(super) fn target_from_fid(fid: i32) -> BuiltinResult<OutputTarget> {
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
            let info = registry::info_for(fid).ok_or_else(|| {
                fprintf_error_with_message(
                    FPRINTF_ERROR_INVALID_IDENTIFIER.message,
                    &FPRINTF_ERROR_INVALID_IDENTIFIER,
                )
            })?;
            ensure_writable(&info)?;
            let handle = registry::shared_handle(fid).ok_or_else(|| {
                fprintf_error_with_message(
                    FPRINTF_ERROR_INVALID_IDENTIFIER.message,
                    &FPRINTF_ERROR_INVALID_IDENTIFIER,
                )
            })?;
            Ok(OutputTarget::File {
                handle,
                encoding: info.encoding.clone(),
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
            if t.shape == vec![1, 1] && tensor::is_scalar_tensor(t) {
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
    let permission = info.permission.to_ascii_lowercase();
    if permission.contains('w') || permission.contains('a') || permission.contains('+') {
        Ok(())
    } else {
        Err(fprintf_error_with_detail(
            &FPRINTF_ERROR_INVALID_IDENTIFIER,
            "file is not open for writing",
        ))
    }
}

pub(super) fn match_stream_label(value: &Value) -> Option<SpecialStream> {
    let candidate = match value {
        Value::String(s) => s.trim().to_string(),
        Value::CharArray(ca) if ca.rows == 1 => {
            ca.data.iter().collect::<String>().trim().to_string()
        }
        Value::StringArray(sa) if sa.data.len() == 1 => sa.data[0].trim().to_string(),
        _ => return None,
    };
    match candidate.to_ascii_lowercase().as_str() {
        "stdout" => Some(SpecialStream::Stdout),
        "stderr" => Some(SpecialStream::Stderr),
        _ => None,
    }
}
