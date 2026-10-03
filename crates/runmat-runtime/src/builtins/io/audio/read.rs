use super::*;

#[runtime_builtin(
    name = "audioread",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::audio"
)]
pub(super) async fn audioread_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    filename: Value,
    args: Vec<Value>,
) -> BuiltinResult<Value> {
    if crate::output_count::current_output_count().is_some_and(|count| count > 2) {
        return Err(audioread_error_with(
            &AUDIOREAD_ERROR_ARGUMENT,
            "audioread: too many output arguments",
        ));
    }
    if matches!(&filename, Value::GpuTensor(_))
        || args.iter().any(|arg| matches!(arg, Value::GpuTensor(_)))
    {
        return Err(audioread_error_with(
            &AUDIOREAD_ERROR_ARGUMENT,
            "audioread: gpuArray inputs are not supported",
        ));
    }
    let options = parse_audioread_options(&args)?;
    let path = resolve_audioread_path(&filename)?;
    let input = FsInputLease::capture(path, &AUDIOREAD_ERROR_IO, AUDIOREAD_BUILTIN_NAME)?;
    let mut cooperation = AudioCooperation::new(runtime.as_ref(), AUDIOREAD_BUILTIN_NAME);
    cooperation.completed(0).await?;
    let decoded = decode_wave_stream(&input, options, &mut cooperation).await?;

    match crate::output_count::current_output_count() {
        None => Ok(Value::Tensor(decoded.samples)),
        Some(0) => Ok(Value::OutputList(Vec::new())),
        Some(1) => Ok(Value::OutputList(vec![Value::Tensor(decoded.samples)])),
        Some(2) => Ok(Value::OutputList(vec![
            Value::Tensor(decoded.samples),
            Value::Num(decoded.sample_rate),
        ])),
        Some(_) => Err(audioread_error_with(
            &AUDIOREAD_ERROR_ARGUMENT,
            "audioread: too many output arguments",
        )),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct AudioreadOptions {
    pub(super) range: Option<AudioSampleRange>,
    pub(super) datatype: AudioReadDataType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct AudioSampleRange {
    pub(super) first: usize,
    pub(super) last: Option<usize>,
}

#[derive(Debug, Clone)]
pub(super) struct DecodedAudio {
    pub(super) samples: Tensor,
    pub(super) sample_rate: f64,
}

pub(super) fn parse_audioread_options(args: &[Value]) -> BuiltinResult<AudioreadOptions> {
    let mut range = None;
    let mut datatype = AudioReadDataType::Double;
    match args {
        [] => {}
        [single] => {
            if let Some(text) = scalar_text(single) {
                datatype = parse_audioread_datatype(&text)?;
            } else {
                range = Some(parse_sample_range(single)?);
            }
        }
        [samples, datatype] => {
            range = Some(parse_sample_range(samples)?);
            let text = scalar_text(datatype).ok_or_else(|| {
                audioread_error_with(
                    &AUDIOREAD_ERROR_ARGUMENT,
                    "audioread: datatype must be \"native\" or \"double\"",
                )
            })?;
            datatype = parse_audioread_datatype(&text)?;
        }
        _ => {
            return Err(audioread_error_with(
                &AUDIOREAD_ERROR_ARGUMENT,
                "audioread: too many input arguments",
            ));
        }
    }
    Ok(AudioreadOptions { range, datatype })
}

pub(super) fn parse_audioread_datatype(value: &str) -> BuiltinResult<AudioReadDataType> {
    AudioReadDataType::parse(value).ok_or_else(|| {
        audioread_error_with(
            &AUDIOREAD_ERROR_ARGUMENT,
            "audioread: datatype must be \"native\" or \"double\"",
        )
    })
}

pub(super) fn parse_sample_range(value: &Value) -> BuiltinResult<AudioSampleRange> {
    let Value::Tensor(tensor) = value else {
        return Err(audioread_error_with(
            &AUDIOREAD_ERROR_ARGUMENT,
            "audioread: sample range must be a two-element double vector",
        ));
    };
    if tensor.numeric_dtype() != NumericDType::F64 || tensor.len() != 2 {
        return Err(audioread_error_with(
            &AUDIOREAD_ERROR_ARGUMENT,
            "audioread: sample range must be a two-element double vector",
        ));
    }
    let values = tensor.materialize_f64();
    let first = parse_positive_integer(values[0], "sample range")?;
    let last = if values[1] == f64::INFINITY {
        None
    } else {
        Some(parse_positive_integer(values[1], "sample range")?)
    };
    if last.is_some_and(|last| first > last) {
        return Err(audioread_error_with(
            &AUDIOREAD_ERROR_ARGUMENT,
            "audioread: sample range first sample must be less than or equal to last sample",
        ));
    }
    Ok(AudioSampleRange { first, last })
}

pub(super) fn parse_positive_integer(value: f64, label: &str) -> BuiltinResult<usize> {
    if !value.is_finite() || value < 1.0 || value.fract() != 0.0 {
        return Err(audioread_error_with(
            &AUDIOREAD_ERROR_ARGUMENT,
            format!("audioread: {label} must be a positive integer"),
        ));
    }
    if value > usize::MAX as f64 {
        return Err(audioread_error_with(
            &AUDIOREAD_ERROR_ARGUMENT,
            format!("audioread: {label} is too large"),
        ));
    }
    Ok(value as usize)
}

pub(super) fn scalar_text(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::CharArray(ca) if ca.rows == 1 => Some(ca.data.iter().collect()),
        Value::StringArray(sa) if sa.data.len() == 1 => Some(sa.data[0].clone()),
        _ => None,
    }
}

pub(super) fn unsupported_sample_coding(bytes: &[u8]) -> String {
    if bytes.starts_with(b"fLaC")
        || bytes.starts_with(b"FORM")
        || bytes.starts_with(b"OggS")
        || parse_mp3(bytes, bytes.len() as u64).is_some()
    {
        "sample decoding is currently implemented for WAV/RF64 PCM and IEEE-float audio; compressed containers are metadata-only".to_string()
    } else {
        "unsupported audio format".to_string()
    }
}

pub(super) fn native_wave_dtype(fmt: WaveFormat) -> NumericDType {
    match (fmt.effective_format_tag(), fmt.effective_bits_per_sample()) {
        (0x0001, 8) => NumericDType::U8,
        (0x0003, 32) => NumericDType::F32,
        _ => NumericDType::F64,
    }
}

pub(super) fn decode_wave_sample(
    bytes: &[u8],
    fmt: WaveFormat,
    native: bool,
) -> Result<f64, String> {
    match (fmt.effective_format_tag(), fmt.effective_bits_per_sample()) {
        (0x0001, 8) => {
            let raw = bytes[0] as f64;
            Ok(if native { raw } else { (raw - 128.0) / 128.0 })
        }
        (0x0001, 16) => {
            let raw = i16::from_le_bytes([bytes[0], bytes[1]]);
            Ok(if native {
                raw as f64
            } else {
                raw as f64 / 32768.0
            })
        }
        (0x0001, 24) => {
            let raw = sign_extend_24(bytes);
            Ok(if native {
                raw as f64
            } else {
                raw as f64 / 8_388_608.0
            })
        }
        (0x0001, 32) => {
            let raw = i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            Ok(if native {
                raw as f64
            } else {
                raw as f64 / 2_147_483_648.0
            })
        }
        (0x0003, 32) => Ok(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f64),
        (0x0003, 64) => Ok(f64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ])),
        (tag, bits) => Err(format!(
            "unsupported WAVE sample encoding tag 0x{tag:04x} with {bits} bits per sample"
        )),
    }
}

pub(super) fn sign_extend_24(bytes: &[u8]) -> i32 {
    let mut value = ((bytes[2] as i32) << 16) | ((bytes[1] as i32) << 8) | bytes[0] as i32;
    if value & 0x0080_0000 != 0 {
        value |= !0x00ff_ffff;
    }
    value
}

pub(super) fn resolve_audioread_path(value: &Value) -> BuiltinResult<PathBuf> {
    match value {
        Value::String(s) => normalize_audioread_path(s),
        Value::CharArray(ca) if ca.rows == 1 => {
            let text: String = ca.data.iter().collect();
            normalize_audioread_path(&text)
        }
        Value::StringArray(sa) if sa.data.len() == 1 => normalize_audioread_path(&sa.data[0]),
        _ => Err(audioread_error(&AUDIOREAD_ERROR_ARGUMENT)),
    }
}

pub(super) fn normalize_audioread_path(raw: &str) -> BuiltinResult<PathBuf> {
    if raw.trim().is_empty() {
        return Err(audioread_error_with(
            &AUDIOREAD_ERROR_ARGUMENT,
            "audioread: filename must not be empty",
        ));
    }
    let expanded = expand_user_path(raw, AUDIOREAD_BUILTIN_NAME)
        .map_err(|msg| audioread_error_with(&AUDIOREAD_ERROR_ARGUMENT, msg))?;
    Ok(Path::new(&expanded).to_path_buf())
}
