use super::*;

#[runtime_builtin(
    name = "audioread",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::audio::read"
)]
pub(super) async fn audioread_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    filename: Value,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let outputs = crate::output_count::current_output_count();
    if outputs.is_some_and(|count| count > 2) {
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
    let options = parse_audioread_options(args.view())?;
    let path = resolve_audioread_path(&filename)?;
    let input = FsInputLease::capture(path, &AUDIOREAD_ERROR_IO, AUDIOREAD_BUILTIN_NAME)?;
    let mut cooperation = AudioCooperation::new(runtime.as_ref(), AUDIOREAD_BUILTIN_NAME);
    cooperation.completed(0).await?;
    if needs_decode_stage(runtime.as_ref()) {
        let runtime = runtime.as_ref().expect("finite budget requires runtime");
        if let Some(stage) = admit_decode_stage(runtime, &input, options, outputs)? {
            return read_staged(runtime, &input, options, outputs, stage, &mut cooperation).await;
        }
    }
    let receipt = select_decode(runtime.as_ref(), &input, options, outputs)?;
    if let Some(receipt) = receipt.as_ref() {
        cooperation.install_resource_liveness_lease(receipt.resource_liveness_lease());
    }
    let result = async {
        cooperation.completed(0).await?;
        let decoded = decode_wave_stream(&input, options, &mut cooperation).await?;
        let value = present_decoded(decoded, outputs);
        cooperation.completed(0).await?;
        Ok(value)
    }
    .await;
    finish_receipt(receipt, &result, cooperation.was_cancelled());
    result
}

async fn read_staged(
    runtime: &crate::context::RuntimeContext,
    input: &FsInputLease,
    options: AudioreadOptions,
    outputs: Option<usize>,
    stage: crate::context::host_effect::HostEffectStagePermit,
    cooperation: &mut AudioCooperation,
) -> BuiltinResult<Value> {
    cooperation.install_stage_reservation(stage.reservation());
    let prepared = prepare_wave_stream(input, options, cooperation)
        .await?
        .ok_or_else(|| {
            build_runtime_error("audioread: staged route has no bounded opened source")
                .with_identifier("RunMat:Placement:InsufficientResourceEvidence")
                .build()
        })?;
    if !prepared.has_bounded_opened_source() {
        return Err(
            build_runtime_error("audioread: staged route has no direct opened source")
                .with_identifier("RunMat:Placement:InsufficientResourceEvidence")
                .build(),
        );
    }
    stage.checkpoint()?;
    let geometry = prepared.geometry;
    let receipt = select_decode_after_stage(runtime, input, options, outputs, &geometry, stage)?;
    cooperation.install_resource_liveness_lease(receipt.resource_liveness_lease());
    let result = async {
        cooperation.completed(0).await?;
        if outputs == Some(0) {
            validate_prepared_wave_stream_discarded(prepared, &input.path, cooperation).await?;
            return Ok(Value::output_list(Vec::new()));
        }
        let permit = receipt.output_allocation_permit().ok_or_else(|| {
            build_runtime_error("audioread: exact output allocation authority is unavailable")
                .with_identifier("RunMat:Placement:InsufficientResourceEvidence")
                .build()
        })?;
        let decoded =
            decode_prepared_wave_stream_exact(prepared, &input.path, cooperation, permit.as_ref())
                .await?;
        let value = present_decoded(decoded, outputs);
        cooperation.completed(0).await?;
        Ok(value)
    }
    .await;
    finish_receipt(Some(receipt), &result, cooperation.was_cancelled());
    result
}

fn present_decoded(decoded: DecodedAudio, outputs: Option<usize>) -> Value {
    match outputs {
        None => Value::Tensor(decoded.samples),
        Some(0) => Value::output_list(Vec::new()),
        Some(1) => Value::output_list(vec![Value::Tensor(decoded.samples)]),
        Some(2) => Value::output_list(vec![
            Value::Tensor(decoded.samples),
            Value::Num(decoded.sample_rate),
        ]),
        Some(_) => unreachable!("output arity was admitted before filesystem capture"),
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

pub(super) fn parse_audioread_options(
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<AudioreadOptions> {
    let mut range = None;
    let mut datatype = AudioReadDataType::Double;
    match args.len() {
        0 => {}
        1 => {
            let single = &args[0];
            if let Some(text) = scalar_text(single)? {
                datatype = parse_audioread_datatype(&text)?;
            } else {
                range = Some(parse_sample_range(single)?);
            }
        }
        2 => {
            let (samples, datatype_value) = (&args[0], &args[1]);
            range = Some(parse_sample_range(samples)?);
            let text = scalar_text(datatype_value)?.ok_or_else(|| {
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

pub(super) fn scalar_text(value: &Value) -> BuiltinResult<Option<String>> {
    match value {
        Value::String(runmat_value::StringElement::Text(s)) => Ok(Some(s.as_str().to_owned())),
        Value::CharArray(ca) if ca.rows == 1 => {
            crate::builtins::io::utf16::decode_utf16_with_error(ca.data(), || {
                audioread_error(&AUDIOREAD_ERROR_ARGUMENT)
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

pub(super) const COMPRESSED_AUDIO_ERROR: &str = "sample decoding is currently implemented for WAV/RF64 PCM and IEEE-float audio; compressed containers are metadata-only";
pub(super) const UNSUPPORTED_AUDIO_ERROR: &str = "unsupported audio format";

pub(super) fn decode_wave_sample(bytes: &[u8], coding: WaveSampleCoding, native: bool) -> f64 {
    match coding {
        WaveSampleCoding::Pcm8 => {
            let raw = bytes[0] as f64;
            if native {
                raw
            } else {
                (raw - 128.0) / 128.0
            }
        }
        WaveSampleCoding::Pcm16 => {
            let raw = i16::from_le_bytes([bytes[0], bytes[1]]);
            if native {
                raw as f64
            } else {
                raw as f64 / 32768.0
            }
        }
        WaveSampleCoding::Pcm24 => {
            let raw = sign_extend_24(bytes);
            if native {
                raw as f64
            } else {
                raw as f64 / 8_388_608.0
            }
        }
        WaveSampleCoding::Pcm32 => {
            let raw = i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            if native {
                raw as f64
            } else {
                raw as f64 / 2_147_483_648.0
            }
        }
        WaveSampleCoding::Float32 => {
            f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f64
        }
        WaveSampleCoding::Float64 => f64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]),
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
        Value::String(runmat_value::StringElement::Text(s)) => normalize_audioread_path(s),
        Value::CharArray(ca) if ca.rows == 1 => {
            let text = crate::builtins::io::utf16::decode_utf16_with_error(ca.data(), || {
                audioread_error(&AUDIOREAD_ERROR_ARGUMENT)
            })?;
            normalize_audioread_path(&text)
        }
        Value::StringArray(sa) if sa.data().len() == 1 && !sa.data()[0].is_missing() => {
            normalize_audioread_path(
                sa.data()[0]
                    .text_ref()
                    .expect("checked nonmissing scalar string"),
            )
        }
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
