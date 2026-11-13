use super::*;

#[runtime_builtin(
    name = "audioinfo",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::io::audio::info"
)]
pub(super) async fn audioinfo_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    filename: Value,
) -> BuiltinResult<Value> {
    let outputs = crate::output_count::current_output_count();
    if outputs.is_some_and(|count| count > 1) {
        return Err(audioinfo_error_with(
            &AUDIOINFO_ERROR_ARGUMENT,
            "audioinfo: too many output arguments",
        ));
    }
    let path = resolve_audioinfo_path(&filename)?;
    let input = FsInputLease::capture(path.clone(), &AUDIOINFO_ERROR_IO, AUDIOINFO_BUILTIN_NAME)?;
    let mut cooperation = AudioCooperation::new(runtime.as_ref(), AUDIOINFO_BUILTIN_NAME);
    cooperation.completed(0).await?;
    let receipt = select_metadata(runtime.as_ref(), &input, outputs)?;
    if let Some(receipt) = receipt.as_ref() {
        cooperation.install_resource_liveness_lease(receipt.resource_liveness_lease());
    }
    let result = async {
        cooperation.completed(0).await?;
        let scan = read_audioinfo_scan(&input, &mut cooperation).await?;
        cooperation.completed(0).await?;
        let metadata = AudioMetadata::parse(&scan).map_err(|message| {
            audioinfo_error_with(&AUDIOINFO_ERROR_FORMAT, format!("audioinfo: {message}"))
        })?;
        cooperation.completed(0).await?;
        let value = match outputs {
            Some(0) => Value::output_list(Vec::new()),
            None | Some(1) => Value::Struct(metadata.into_struct(&path, scan.file_size as f64)),
            Some(_) => unreachable!("output arity was admitted before filesystem capture"),
        };
        cooperation.completed(0).await?;
        Ok(value)
    }
    .await;
    finish_receipt(receipt, &result, cooperation.was_cancelled());
    result
}

pub(super) struct AudioInfoScan {
    pub(super) prefix: Vec<u8>,
    pub(super) tail: Option<Vec<u8>>,
    pub(super) file_size: u64,
}

pub(super) async fn read_audioinfo_scan(
    input: &FsInputLease,
    cooperation: &mut AudioCooperation,
) -> BuiltinResult<AudioInfoScan> {
    let path = input.path.as_path();
    let mut file = cooperation.await_io(input.open()).await?.map_err(|err| {
        audioinfo_error_with_source(
            &AUDIOINFO_ERROR_IO,
            format!("audioinfo: unable to read \"{}\" ({err})", path.display()),
            err,
        )
    })?;
    let metadata = match cooperation.await_io(file.metadata_async()).await? {
        Ok(metadata) => metadata,
        Err(err) if err.kind() == std::io::ErrorKind::Unsupported => cooperation
            .await_io(input.provider.metadata(path))
            .await?
            .map_err(|err| {
                audioinfo_error_with_source(
                    &AUDIOINFO_ERROR_IO,
                    format!(
                        "audioinfo: unable to inspect \"{}\" after opening ({err})",
                        path.display()
                    ),
                    err,
                )
            })?,
        Err(err) => {
            return Err(audioinfo_error_with_source(
                &AUDIOINFO_ERROR_IO,
                format!(
                    "audioinfo: unable to inspect opened file \"{}\" ({err})",
                    path.display()
                ),
                err,
            ));
        }
    };
    let file_size = metadata.len();
    let prefix_len = file_size.min(MAX_AUDIOINFO_PREFIX_BYTES);
    let mut bytes = Vec::new();
    file.by_ref()
        .take(prefix_len)
        .read_to_end(&mut bytes)
        .map_err(|err| {
            audioinfo_error_with_source(
                &AUDIOINFO_ERROR_IO,
                format!("audioinfo: unable to read \"{}\" ({err})", path.display()),
                err,
            )
        })?;
    let tail = if bytes.starts_with(b"OggS") && file_size > prefix_len {
        let tail_len = file_size.min(MAX_AUDIOINFO_TAIL_BYTES);
        let tail_start = file_size.saturating_sub(tail_len);
        match file.seek(SeekFrom::Start(tail_start)) {
            Ok(_) => {
                let mut tail = Vec::new();
                match file.take(tail_len).read_to_end(&mut tail) {
                    Ok(_) => Some(tail),
                    Err(_) => None,
                }
            }
            Err(_) => None,
        }
    } else {
        None
    };
    Ok(AudioInfoScan {
        prefix: bytes,
        tail,
        file_size,
    })
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct AudioMetadata {
    pub(super) format: &'static str,
    pub(super) compression_method: String,
    pub(super) num_channels: u16,
    pub(super) sample_rate: f64,
    pub(super) total_samples: Option<u64>,
    pub(super) bits_per_sample: Option<u16>,
    pub(super) bit_rate: Option<f64>,
}

impl AudioMetadata {
    pub(super) fn parse(scan: &AudioInfoScan) -> Result<Self, String> {
        let bytes = scan.prefix.as_slice();
        if bytes.len() < 4 {
            return Err("file is too small to contain audio metadata".to_string());
        }
        if bytes.starts_with(b"RIFF") || bytes.starts_with(b"RF64") {
            return parse_wave(bytes, scan.file_size);
        }
        if bytes.starts_with(b"fLaC") {
            return parse_flac(bytes);
        }
        if bytes.starts_with(b"FORM") {
            return parse_aiff(bytes);
        }
        if bytes.starts_with(b"OggS") {
            return parse_ogg_vorbis(scan);
        }
        if let Some(mp3) = parse_mp3(bytes, scan.file_size) {
            return Ok(mp3);
        }
        Err("unsupported audio format".to_string())
    }

    pub(super) fn into_struct(self, path: &Path, file_size: f64) -> StructValue {
        let total_samples = self.total_samples.map(|v| v as f64).unwrap_or(f64::NAN);
        let duration = self
            .total_samples
            .map(|samples| samples as f64 / self.sample_rate)
            .unwrap_or_else(|| {
                self.bit_rate
                    .filter(|rate| *rate > 0.0)
                    .map(|rate| file_size * 8.0 / rate)
                    .unwrap_or(f64::NAN)
            });
        let mut out = runmat_value::UnboundedStructValueBuilder::new();
        out.insert(
            "Filename",
            Value::string(path.to_string_lossy().into_owned()),
        );
        out.insert(
            "CompressionMethod",
            Value::string(self.compression_method.clone()),
        );
        out.insert("NumChannels", Value::Num(self.num_channels as f64));
        out.insert("SampleRate", Value::Num(self.sample_rate));
        out.insert("TotalSamples", Value::Num(total_samples));
        out.insert("Duration", Value::Num(duration));
        out.insert(
            "BitsPerSample",
            Value::Num(self.bits_per_sample.map(|v| v as f64).unwrap_or(f64::NAN)),
        );
        out.insert("BitRate", Value::Num(self.bit_rate.unwrap_or(f64::NAN)));
        out.insert("FileSize", Value::Num(file_size));
        out.insert("Format", Value::string(self.format.to_string()));
        out.finish()
    }
}

pub(super) fn parse_wave(bytes: &[u8], file_size: u64) -> Result<AudioMetadata, String> {
    let (format, data_bytes) = parse_wave_metadata(bytes, file_size)?;
    let total_samples = if format.block_align > 0 {
        Some(data_bytes / format.block_align as u64)
    } else {
        None
    };
    Ok(AudioMetadata {
        format: "WAV",
        compression_method: wave_compression_name(format.effective_format_tag()).to_string(),
        num_channels: format.channels,
        sample_rate: format.sample_rate as f64,
        total_samples,
        bits_per_sample: Some(
            format
                .valid_bits_per_sample
                .unwrap_or(format.bits_per_sample),
        ),
        bit_rate: Some(format.byte_rate as f64 * 8.0),
    })
}

pub(super) fn resolve_audioinfo_path(value: &Value) -> BuiltinResult<PathBuf> {
    match value {
        Value::String(runmat_value::StringElement::Text(s)) => normalize_audioinfo_path(s),
        Value::CharArray(ca) if ca.rows == 1 => {
            let text = crate::builtins::io::utf16::decode_utf16_with_error(ca.data(), || {
                audioinfo_error(&AUDIOINFO_ERROR_ARGUMENT)
            })?;
            normalize_audioinfo_path(&text)
        }
        Value::StringArray(sa) if sa.data().len() == 1 && !sa.data()[0].is_missing() => {
            normalize_audioinfo_path(
                sa.data()[0]
                    .text_ref()
                    .expect("checked nonmissing scalar string"),
            )
        }
        _ => Err(audioinfo_error(&AUDIOINFO_ERROR_ARGUMENT)),
    }
}

pub(super) fn normalize_audioinfo_path(raw: &str) -> BuiltinResult<PathBuf> {
    if raw.trim().is_empty() {
        return Err(audioinfo_error_with(
            &AUDIOINFO_ERROR_ARGUMENT,
            "audioinfo: filename must not be empty",
        ));
    }
    let expanded = expand_user_path(raw, AUDIOINFO_BUILTIN_NAME)
        .map_err(|msg| audioinfo_error_with(&AUDIOINFO_ERROR_ARGUMENT, msg))?;
    Ok(Path::new(&expanded).to_path_buf())
}
