use super::*;

#[runtime_builtin(
    name = "audioread",
    binding_variant = "default",
    builtin_path = "crate::builtins::io::audio"
)]
pub(super) async fn audioread_builtin(filename: Value, args: Vec<Value>) -> BuiltinResult<Value> {
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
    let bytes = fs::read_async(&path).await.map_err(|err| {
        audioread_error_with_source(
            &AUDIOREAD_ERROR_IO,
            format!("audioread: unable to read \"{}\" ({err})", path.display()),
            err,
        )
    })?;
    let decoded = decode_audio_samples(&bytes, options).map_err(|message| {
        audioread_error_with(&AUDIOREAD_ERROR_FORMAT, format!("audioread: {message}"))
    })?;

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

pub(super) fn decode_audio_samples(
    bytes: &[u8],
    options: AudioreadOptions,
) -> Result<DecodedAudio, String> {
    if bytes.starts_with(b"RIFF") || bytes.starts_with(b"RF64") {
        decode_wave_samples(bytes, options)
    } else if bytes.starts_with(b"fLaC")
        || bytes.starts_with(b"FORM")
        || bytes.starts_with(b"OggS")
        || parse_mp3(bytes, bytes.len() as u64).is_some()
    {
        Err("sample decoding is currently implemented for WAV/RF64 PCM and IEEE-float audio; compressed containers are metadata-only".to_string())
    } else {
        Err("unsupported audio format".to_string())
    }
}

pub(super) fn decode_wave_samples(
    bytes: &[u8],
    options: AudioreadOptions,
) -> Result<DecodedAudio, String> {
    let parsed = parse_wave_container(bytes)?;
    let fmt = parsed.format;
    let channels = fmt.channels as usize;
    let block_align = fmt.block_align as usize;
    let bytes_per_channel = usize::from(fmt.bits_per_sample.div_ceil(8));
    if channels == 0 || block_align == 0 || bytes_per_channel == 0 {
        return Err("WAVE fmt chunk has invalid sample layout".to_string());
    }
    let minimum_block_align = channels
        .checked_mul(bytes_per_channel)
        .ok_or_else(|| "WAVE channel layout overflows platform limits".to_string())?;
    if block_align < minimum_block_align {
        return Err("WAVE block alignment is smaller than the channel sample width".to_string());
    }
    if parsed.data_len % block_align != 0 {
        return Err("WAVE data chunk is not aligned to whole sample frames".to_string());
    }
    let total_frames = parsed.data_len / block_align;
    let (start_frame, end_frame_exclusive) = match options.range {
        Some(AudioSampleRange { first, last }) => {
            let last = last.unwrap_or(total_frames);
            if first > total_frames || last > total_frames {
                return Err("sample range exceeds the available audio frames".to_string());
            }
            (first - 1, last)
        }
        None => (0, total_frames),
    };
    let frame_count = end_frame_exclusive - start_frame;
    let data = &bytes[parsed.data_offset..parsed.data_offset + parsed.data_len];
    let shape = vec![frame_count, channels];
    let samples = if options.datatype == AudioReadDataType::Native {
        if let Some(storage) = native_pcm_storage(
            fmt,
            data,
            channels,
            start_frame,
            end_frame_exclusive,
            block_align,
            bytes_per_channel,
        )? {
            Tensor::new_integer(storage, shape)
                .map_err(|err| format!("decoded audio tensor shape is invalid: {err}"))?
        } else {
            let out = collect_wave_samples(
                data,
                channels,
                start_frame,
                end_frame_exclusive,
                block_align,
                bytes_per_channel,
                |sample_bytes| decode_wave_sample(sample_bytes, fmt, true),
            )?;
            Tensor::new_with_dtype(out, shape, native_wave_dtype(fmt))
                .map_err(|err| format!("decoded audio tensor shape is invalid: {err}"))?
        }
    } else {
        let out = collect_wave_samples(
            data,
            channels,
            start_frame,
            end_frame_exclusive,
            block_align,
            bytes_per_channel,
            |sample_bytes| decode_wave_sample(sample_bytes, fmt, false),
        )?;
        Tensor::new_with_dtype(out, shape, NumericDType::F64)
            .map_err(|err| format!("decoded audio tensor shape is invalid: {err}"))?
    };
    Ok(DecodedAudio {
        samples,
        sample_rate: fmt.sample_rate as f64,
    })
}

pub(super) fn collect_wave_samples<T>(
    data: &[u8],
    channels: usize,
    start_frame: usize,
    end_frame_exclusive: usize,
    block_align: usize,
    bytes_per_channel: usize,
    mut decode: impl FnMut(&[u8]) -> Result<T, String>,
) -> Result<Vec<T>, String> {
    let frame_count = end_frame_exclusive - start_frame;
    let sample_count = frame_count
        .checked_mul(channels)
        .ok_or_else(|| "decoded audio dimensions overflow platform limits".to_string())?;
    let mut out = Vec::new();
    out.try_reserve_exact(sample_count)
        .map_err(|err| format!("decoded audio allocation failed: {err}"))?;
    for channel in 0..channels {
        for frame in start_frame..end_frame_exclusive {
            let sample_offset = frame
                .checked_mul(block_align)
                .and_then(|base| base.checked_add(channel * bytes_per_channel))
                .ok_or_else(|| "sample offset overflows platform limits".to_string())?;
            out.push(decode(
                &data[sample_offset..sample_offset + bytes_per_channel],
            )?);
        }
    }
    Ok(out)
}

pub(super) fn native_pcm_storage(
    fmt: WaveFormat,
    data: &[u8],
    channels: usize,
    start_frame: usize,
    end_frame_exclusive: usize,
    block_align: usize,
    bytes_per_channel: usize,
) -> Result<Option<IntegerStorage>, String> {
    if fmt.effective_format_tag() != 0x0001 {
        return Ok(None);
    }
    let storage = match fmt.effective_bits_per_sample() {
        8 => IntegerStorage::U8(collect_wave_samples(
            data,
            channels,
            start_frame,
            end_frame_exclusive,
            block_align,
            bytes_per_channel,
            |sample| Ok(sample[0]),
        )?),
        16 => IntegerStorage::I16(collect_wave_samples(
            data,
            channels,
            start_frame,
            end_frame_exclusive,
            block_align,
            bytes_per_channel,
            |sample| Ok(i16::from_le_bytes([sample[0], sample[1]])),
        )?),
        // MATLAB returns 24-bit PCM as an int32 whose valid bits are left-aligned.
        24 => IntegerStorage::I32(collect_wave_samples(
            data,
            channels,
            start_frame,
            end_frame_exclusive,
            block_align,
            bytes_per_channel,
            |sample| Ok(sign_extend_24(sample) << 8),
        )?),
        32 => IntegerStorage::I32(collect_wave_samples(
            data,
            channels,
            start_frame,
            end_frame_exclusive,
            block_align,
            bytes_per_channel,
            |sample| {
                Ok(i32::from_le_bytes([
                    sample[0], sample[1], sample[2], sample[3],
                ]))
            },
        )?),
        _ => return Ok(None),
    };
    Ok(Some(storage))
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
