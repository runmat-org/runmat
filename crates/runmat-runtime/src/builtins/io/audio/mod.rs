//! Audio file metadata builtins.

use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    StructValue, Value,
};
use runmat_filesystem as fs;
use runmat_macros::runtime_builtin;

use crate::builtins::common::fs::expand_user_path;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "audioinfo";
const MAX_AUDIOINFO_PREFIX_BYTES: u64 = 1024 * 1024;
const MAX_AUDIOINFO_TAIL_BYTES: u64 = 64 * 1024;

const AUDIOINFO_OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "info",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Audio metadata structure.",
}];
const AUDIOINFO_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "filename",
    ty: BuiltinParamType::StringScalar,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Audio file path.",
}];
const AUDIOINFO_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "info = audioinfo(filename)",
    inputs: &AUDIOINFO_INPUTS,
    outputs: &AUDIOINFO_OUTPUTS,
}];

const AUDIOINFO_ERROR_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.AUDIOINFO.ARGUMENT",
    identifier: Some("RunMat:audioinfo:InvalidArgument"),
    when: "Filename is missing or cannot be interpreted as a scalar path.",
    message: "audioinfo: invalid filename",
};
const AUDIOINFO_ERROR_IO: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.AUDIOINFO.IO",
    identifier: Some("RunMat:audioinfo:Io"),
    when: "The audio file cannot be read.",
    message: "audioinfo: unable to read file",
};
const AUDIOINFO_ERROR_FORMAT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.AUDIOINFO.FORMAT",
    identifier: Some("RunMat:audioinfo:UnsupportedFormat"),
    when: "The file is not a supported audio container or has malformed metadata.",
    message: "audioinfo: unsupported or invalid audio file",
};
const AUDIOINFO_ERRORS: [BuiltinErrorDescriptor; 3] = [
    AUDIOINFO_ERROR_ARGUMENT,
    AUDIOINFO_ERROR_IO,
    AUDIOINFO_ERROR_FORMAT,
];

pub const AUDIOINFO_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &AUDIOINFO_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &AUDIOINFO_ERRORS,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::io::audio")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "audioinfo",
    op_kind: GpuOpKind::Custom("io-audioinfo"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Runs on the host; file metadata inspection is not an acceleration operation.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::io::audio")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "audioinfo",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Not eligible for fusion; performs host-side file I/O.",
};

fn audioinfo_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    audioinfo_error_with(error, error.message)
}

fn audioinfo_error_with(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn audioinfo_error_with_source<E>(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
    source: E,
) -> RuntimeError
where
    E: std::error::Error + Send + Sync + 'static,
{
    let mut builder = build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .with_source(source);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn map_control_flow(err: RuntimeError) -> RuntimeError {
    let identifier = err.identifier().map(|value| value.to_string());
    let message = err.message().to_string();
    let mut builder = build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .with_source(err);
    if let Some(identifier) = identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "audioinfo",
    category = "io/audio",
    summary = "Read metadata from an audio file.",
    keywords = "audioinfo,audio,wav,flac,aiff,mp3,ogg,metadata,sample rate,channels",
    accel = "cpu",
    type_resolver(crate::builtins::io::type_resolvers::struct_type),
    descriptor(crate::builtins::io::audio::AUDIOINFO_DESCRIPTOR),
    builtin_path = "crate::builtins::io::audio"
)]
async fn audioinfo_builtin(filename: Value) -> BuiltinResult<Value> {
    let filename = gather_if_needed_async(&filename)
        .await
        .map_err(map_control_flow)?;
    let path = resolve_path(&filename)?;
    let scan = read_audioinfo_scan(&path).await?;
    let metadata = AudioMetadata::parse(&scan).map_err(|message| {
        audioinfo_error_with(&AUDIOINFO_ERROR_FORMAT, format!("audioinfo: {message}"))
    })?;
    Ok(Value::Struct(
        metadata.into_struct(&path, scan.file_size as f64),
    ))
}

struct AudioInfoScan {
    prefix: Vec<u8>,
    tail: Option<Vec<u8>>,
    file_size: u64,
}

async fn read_audioinfo_scan(path: &Path) -> BuiltinResult<AudioInfoScan> {
    let mut file = fs::File::open_async(path).await.map_err(|err| {
        audioinfo_error_with_source(
            &AUDIOINFO_ERROR_IO,
            format!("audioinfo: unable to read \"{}\" ({err})", path.display()),
            err,
        )
    })?;
    let metadata = match file.metadata_async().await {
        Ok(metadata) => metadata,
        Err(err) if err.kind() == std::io::ErrorKind::Unsupported => {
            fs::metadata_async(path).await.map_err(|err| {
                audioinfo_error_with_source(
                    &AUDIOINFO_ERROR_IO,
                    format!(
                        "audioinfo: unable to inspect \"{}\" after opening ({err})",
                        path.display()
                    ),
                    err,
                )
            })?
        }
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
struct AudioMetadata {
    format: &'static str,
    compression_method: String,
    num_channels: u16,
    sample_rate: f64,
    total_samples: Option<u64>,
    bits_per_sample: Option<u16>,
    bit_rate: Option<f64>,
}

impl AudioMetadata {
    fn parse(scan: &AudioInfoScan) -> Result<Self, String> {
        let bytes = scan.prefix.as_slice();
        if bytes.len() < 4 {
            return Err("file is too small to contain audio metadata".to_string());
        }
        if bytes.starts_with(b"RIFF") || bytes.starts_with(b"RF64") {
            return parse_wave(bytes);
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

    fn into_struct(self, path: &Path, file_size: f64) -> StructValue {
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
        let mut out = StructValue::new();
        out.insert(
            "Filename",
            Value::String(path.to_string_lossy().into_owned()),
        );
        out.insert(
            "CompressionMethod",
            Value::String(self.compression_method.clone()),
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
        out.insert("Format", Value::String(self.format.to_string()));
        out
    }
}

fn parse_wave(bytes: &[u8]) -> Result<AudioMetadata, String> {
    if bytes.len() < 12 || &bytes[8..12] != b"WAVE" {
        return Err("RIFF/RF64 file is not a WAVE container".to_string());
    }
    let is_rf64 = bytes.starts_with(b"RF64");
    let mut pos = 12usize;
    let mut fmt: Option<WaveFormat> = None;
    let mut data_bytes: Option<u64> = None;
    let mut rf64_data_size: Option<u64> = None;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let declared_size = read_u32_le(bytes, pos + 4)?;
        pos += 8;
        if is_rf64 && id == b"data" && declared_size == u32::MAX {
            let data_size = rf64_data_size.ok_or_else(|| {
                "RF64 data chunk uses sentinel size without ds64 metadata".to_string()
            })?;
            data_bytes = Some(data_size);
            break;
        }
        let size = declared_size as usize;
        match id {
            b"ds64" if is_rf64 => {
                let end = pos
                    .checked_add(size)
                    .ok_or_else(|| "WAVE chunk size overflows address space".to_string())?;
                if end > bytes.len() {
                    return Err("WAVE chunk extends past end of file".to_string());
                }
                if size < 24 {
                    return Err("RF64 ds64 chunk is too short".to_string());
                }
                rf64_data_size = Some(read_u64_le(bytes, pos + 8)?);
            }
            b"fmt " => {
                let end = pos
                    .checked_add(size)
                    .ok_or_else(|| "WAVE chunk size overflows address space".to_string())?;
                if end > bytes.len() {
                    return Err("WAVE chunk extends past end of file".to_string());
                }
                fmt = Some(parse_wave_fmt(&bytes[pos..end])?);
            }
            b"data" => {
                data_bytes = Some(size as u64);
                break;
            }
            _ => {
                let end = pos
                    .checked_add(size)
                    .ok_or_else(|| "WAVE chunk size overflows address space".to_string())?;
                if end > bytes.len() {
                    return Err("WAVE chunk extends past end of file".to_string());
                }
            }
        }
        let end = pos
            .checked_add(size)
            .ok_or_else(|| "WAVE chunk size overflows address space".to_string())?;
        pos = end
            .checked_add(size % 2)
            .ok_or_else(|| "WAVE chunk padding overflows address space".to_string())?;
    }

    let fmt = fmt.ok_or_else(|| "WAVE file is missing a fmt chunk".to_string())?;
    let total_samples = data_bytes.and_then(|len| {
        if fmt.block_align > 0 {
            Some(len / fmt.block_align as u64)
        } else {
            None
        }
    });
    Ok(AudioMetadata {
        format: "WAV",
        compression_method: wave_compression_name(fmt.format_tag).to_string(),
        num_channels: fmt.channels,
        sample_rate: fmt.sample_rate as f64,
        total_samples,
        bits_per_sample: Some(fmt.bits_per_sample),
        bit_rate: Some(fmt.byte_rate as f64 * 8.0),
    })
}

#[derive(Debug, Clone, Copy)]
struct WaveFormat {
    format_tag: u16,
    channels: u16,
    sample_rate: u32,
    byte_rate: u32,
    block_align: u16,
    bits_per_sample: u16,
}

fn parse_wave_fmt(chunk: &[u8]) -> Result<WaveFormat, String> {
    if chunk.len() < 16 {
        return Err("WAVE fmt chunk is too short".to_string());
    }
    let format_tag = u16::from_le_bytes([chunk[0], chunk[1]]);
    let channels = u16::from_le_bytes([chunk[2], chunk[3]]);
    let sample_rate = u32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
    let byte_rate = u32::from_le_bytes([chunk[8], chunk[9], chunk[10], chunk[11]]);
    let block_align = u16::from_le_bytes([chunk[12], chunk[13]]);
    let bits_per_sample = u16::from_le_bytes([chunk[14], chunk[15]]);
    if channels == 0 || sample_rate == 0 {
        return Err("WAVE fmt chunk has invalid channel count or sample rate".to_string());
    }
    Ok(WaveFormat {
        format_tag,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
    })
}

fn wave_compression_name(tag: u16) -> &'static str {
    match tag {
        0x0001 => "PCM",
        0x0003 => "IEEE Float",
        0x0006 => "A-law",
        0x0007 => "mu-law",
        0xFFFE => "Extensible",
        _ => "Unknown",
    }
}

fn parse_flac(bytes: &[u8]) -> Result<AudioMetadata, String> {
    let mut pos = 4usize;
    while pos + 4 <= bytes.len() {
        let header = bytes[pos];
        let block_type = header & 0x7F;
        let last = header & 0x80 != 0;
        let len = ((bytes[pos + 1] as usize) << 16)
            | ((bytes[pos + 2] as usize) << 8)
            | bytes[pos + 3] as usize;
        pos += 4;
        if pos + len > bytes.len() {
            return Err("FLAC metadata block extends past end of file".to_string());
        }
        if block_type == 0 {
            if len < 34 {
                return Err("FLAC STREAMINFO block is too short".to_string());
            }
            let stream = &bytes[pos..pos + len];
            let packed = u64::from_be_bytes([
                stream[10], stream[11], stream[12], stream[13], stream[14], stream[15], stream[16],
                stream[17],
            ]);
            let sample_rate = ((packed >> 44) & 0xFFFFF) as u32;
            let channels = (((packed >> 41) & 0x7) + 1) as u16;
            let bits_per_sample = (((packed >> 36) & 0x1F) + 1) as u16;
            let total_samples = packed & 0x000F_FFFF_FFFF;
            if sample_rate == 0 {
                return Err("FLAC STREAMINFO has zero sample rate".to_string());
            }
            return Ok(AudioMetadata {
                format: "FLAC",
                compression_method: "FLAC".to_string(),
                num_channels: channels,
                sample_rate: sample_rate as f64,
                total_samples: if total_samples == 0 {
                    None
                } else {
                    Some(total_samples)
                },
                bits_per_sample: Some(bits_per_sample),
                bit_rate: None,
            });
        }
        pos += len;
        if last {
            break;
        }
    }
    Err("FLAC file is missing STREAMINFO metadata".to_string())
}

fn parse_aiff(bytes: &[u8]) -> Result<AudioMetadata, String> {
    if bytes.len() < 12 {
        return Err("AIFF file is too short".to_string());
    }
    let form = &bytes[8..12];
    if form != b"AIFF" && form != b"AIFC" {
        return Err("FORM container is not AIFF or AIFC".to_string());
    }
    let mut pos = 12usize;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = read_u32_be(bytes, pos + 4)? as usize;
        pos += 8;
        let end = pos
            .checked_add(size)
            .ok_or_else(|| "AIFF chunk size overflows address space".to_string())?;
        if end > bytes.len() {
            return Err("AIFF chunk extends past end of file".to_string());
        }
        if id == b"COMM" {
            if size < 18 {
                return Err("AIFF COMM chunk is too short".to_string());
            }
            let channels = u16::from_be_bytes([bytes[pos], bytes[pos + 1]]);
            let total_samples = read_u32_be(bytes, pos + 2)? as u64;
            let bits_per_sample = u16::from_be_bytes([bytes[pos + 6], bytes[pos + 7]]);
            let sample_rate = read_ieee_extended_80(&bytes[pos + 8..pos + 18])?;
            let compression_method = if form == b"AIFC" && size >= 22 {
                match &bytes[pos + 18..pos + 22] {
                    b"NONE" => "PCM",
                    b"fl32" | b"FL32" => "IEEE Float",
                    b"fl64" | b"FL64" => "IEEE Float",
                    b"ulaw" | b"ULAW" => "mu-law",
                    b"alaw" | b"ALAW" => "A-law",
                    code => std::str::from_utf8(code).unwrap_or("Unknown"),
                }
                .to_string()
            } else {
                "PCM".to_string()
            };
            return Ok(AudioMetadata {
                format: if form == b"AIFC" { "AIFC" } else { "AIFF" },
                compression_method,
                num_channels: channels,
                sample_rate,
                total_samples: Some(total_samples),
                bits_per_sample: Some(bits_per_sample),
                bit_rate: Some(sample_rate * channels as f64 * bits_per_sample as f64),
            });
        }
        pos = end
            .checked_add(size % 2)
            .ok_or_else(|| "AIFF chunk padding overflows address space".to_string())?;
    }
    Err("AIFF file is missing COMM metadata".to_string())
}

fn parse_ogg_vorbis(scan: &AudioInfoScan) -> Result<AudioMetadata, String> {
    let prefix = scan.prefix.as_slice();
    let first = parse_ogg_page(prefix, 0)?;
    if first.body.len() < 30 || &first.body[1..7] != b"vorbis" || first.body[0] != 1 {
        return Err("Ogg container is not Vorbis audio".to_string());
    }
    let channels = first.body[11] as u16;
    let sample_rate = u32::from_le_bytes([
        first.body[12],
        first.body[13],
        first.body[14],
        first.body[15],
    ]);
    if channels == 0 || sample_rate == 0 {
        return Err(
            "Vorbis identification header has invalid channel count or sample rate".to_string(),
        );
    }
    let nominal_bitrate = i32::from_le_bytes([
        first.body[20],
        first.body[21],
        first.body[22],
        first.body[23],
    ]);
    let total_samples = match scan.tail.as_deref() {
        Some(bytes) => find_last_ogg_granule(bytes),
        None if scan.file_size <= prefix.len() as u64 => find_last_ogg_granule(prefix),
        None => None,
    };
    Ok(AudioMetadata {
        format: "OGG",
        compression_method: "Vorbis".to_string(),
        num_channels: channels,
        sample_rate: sample_rate as f64,
        total_samples,
        bits_per_sample: None,
        bit_rate: if nominal_bitrate > 0 {
            Some(nominal_bitrate as f64)
        } else {
            None
        },
    })
}

struct OggPage {
    granule_position: u64,
    body: Vec<u8>,
    next_pos: usize,
}

fn parse_ogg_page(bytes: &[u8], pos: usize) -> Result<OggPage, String> {
    if pos + 27 > bytes.len() || &bytes[pos..pos + 4] != b"OggS" {
        return Err("invalid Ogg page header".to_string());
    }
    let segments = bytes[pos + 26] as usize;
    if pos + 27 + segments > bytes.len() {
        return Err("Ogg segment table extends past end of file".to_string());
    }
    let mut body_len = 0usize;
    for len in &bytes[pos + 27..pos + 27 + segments] {
        body_len += *len as usize;
    }
    let body_start = pos + 27 + segments;
    if body_start + body_len > bytes.len() {
        return Err("Ogg page body extends past end of file".to_string());
    }
    let granule_position = u64::from_le_bytes([
        bytes[pos + 6],
        bytes[pos + 7],
        bytes[pos + 8],
        bytes[pos + 9],
        bytes[pos + 10],
        bytes[pos + 11],
        bytes[pos + 12],
        bytes[pos + 13],
    ]);
    Ok(OggPage {
        granule_position,
        body: bytes[body_start..body_start + body_len].to_vec(),
        next_pos: body_start + body_len,
    })
}

fn find_last_ogg_granule(bytes: &[u8]) -> Option<u64> {
    let mut pos = 0usize;
    let mut last = None;
    while pos + 27 <= bytes.len() {
        let Some(offset) = find_signature(&bytes[pos..], b"OggS") else {
            break;
        };
        pos += offset;
        match parse_ogg_page(bytes, pos) {
            Ok(page) => {
                if page.granule_position != u64::MAX {
                    last = Some(page.granule_position);
                }
                pos = page.next_pos;
            }
            Err(_) => break,
        }
    }
    last
}

fn parse_mp3(bytes: &[u8], file_size: u64) -> Option<AudioMetadata> {
    let mut pos = skip_id3v2(bytes);
    while pos + 4 <= bytes.len() {
        if bytes[pos] == 0xFF && (bytes[pos + 1] & 0xE0) == 0xE0 {
            if let Some(frame) = parse_mpeg_audio_header(&bytes[pos..pos + 4]) {
                let payload_bytes = file_size.saturating_sub(pos as u64) as f64;
                let duration = if frame.bit_rate > 0 {
                    Some(payload_bytes * 8.0 / frame.bit_rate as f64)
                } else {
                    None
                };
                let total_samples =
                    duration.map(|seconds| (seconds * frame.sample_rate as f64).round() as u64);
                return Some(AudioMetadata {
                    format: "MP3",
                    compression_method: frame.layer.to_string(),
                    num_channels: frame.channels,
                    sample_rate: frame.sample_rate as f64,
                    total_samples,
                    bits_per_sample: None,
                    bit_rate: Some(frame.bit_rate as f64),
                });
            }
        }
        pos += 1;
    }
    None
}

fn skip_id3v2(bytes: &[u8]) -> usize {
    if bytes.len() >= 10 && &bytes[0..3] == b"ID3" {
        let size = ((bytes[6] as usize & 0x7F) << 21)
            | ((bytes[7] as usize & 0x7F) << 14)
            | ((bytes[8] as usize & 0x7F) << 7)
            | (bytes[9] as usize & 0x7F);
        10 + size
    } else {
        0
    }
}

struct MpegFrame {
    layer: &'static str,
    sample_rate: u32,
    bit_rate: u32,
    channels: u16,
}

fn parse_mpeg_audio_header(header: &[u8]) -> Option<MpegFrame> {
    let version_id = (header[1] >> 3) & 0x03;
    let layer_id = (header[1] >> 1) & 0x03;
    let bitrate_index = (header[2] >> 4) & 0x0F;
    let sample_rate_index = (header[2] >> 2) & 0x03;
    let channel_mode = (header[3] >> 6) & 0x03;
    if version_id == 1
        || layer_id == 0
        || bitrate_index == 0
        || bitrate_index == 0x0F
        || sample_rate_index == 0x03
    {
        return None;
    }
    let version = match version_id {
        3 => MpegVersion::V1,
        2 => MpegVersion::V2,
        0 => MpegVersion::V25,
        _ => return None,
    };
    let layer = match layer_id {
        3 => MpegLayer::LayerI,
        2 => MpegLayer::LayerII,
        1 => MpegLayer::LayerIII,
        _ => return None,
    };
    let sample_rate = mpeg_sample_rate(version, sample_rate_index)?;
    let bit_rate = mpeg_bit_rate(version, layer, bitrate_index)? * 1000;
    Some(MpegFrame {
        layer: match layer {
            MpegLayer::LayerI => "MPEG Layer I",
            MpegLayer::LayerII => "MPEG Layer II",
            MpegLayer::LayerIII => "MPEG Layer III",
        },
        sample_rate,
        bit_rate,
        channels: if channel_mode == 3 { 1 } else { 2 },
    })
}

#[derive(Debug, Clone, Copy)]
enum MpegVersion {
    V1,
    V2,
    V25,
}

#[derive(Debug, Clone, Copy)]
enum MpegLayer {
    LayerI,
    LayerII,
    LayerIII,
}

fn mpeg_sample_rate(version: MpegVersion, index: u8) -> Option<u32> {
    let base = match index {
        0 => 44_100,
        1 => 48_000,
        2 => 32_000,
        _ => return None,
    };
    Some(match version {
        MpegVersion::V1 => base,
        MpegVersion::V2 => base / 2,
        MpegVersion::V25 => base / 4,
    })
}

fn mpeg_bit_rate(version: MpegVersion, layer: MpegLayer, index: u8) -> Option<u32> {
    const V1_L1: [u32; 16] = [
        0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 0,
    ];
    const V1_L2: [u32; 16] = [
        0, 32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 0,
    ];
    const V1_L3: [u32; 16] = [
        0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0,
    ];
    const V2_L1: [u32; 16] = [
        0, 32, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256, 0,
    ];
    const V2_L23: [u32; 16] = [
        0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0,
    ];
    let table = match (version, layer) {
        (MpegVersion::V1, MpegLayer::LayerI) => V1_L1,
        (MpegVersion::V1, MpegLayer::LayerII) => V1_L2,
        (MpegVersion::V1, MpegLayer::LayerIII) => V1_L3,
        (_, MpegLayer::LayerI) => V2_L1,
        (_, MpegLayer::LayerII | MpegLayer::LayerIII) => V2_L23,
    };
    table.get(index as usize).copied().filter(|rate| *rate > 0)
}

fn read_ieee_extended_80(bytes: &[u8]) -> Result<f64, String> {
    if bytes.len() != 10 {
        return Err("AIFF sample rate field must be 80 bits".to_string());
    }
    let sign = if bytes[0] & 0x80 != 0 { -1.0 } else { 1.0 };
    let exponent = (((bytes[0] & 0x7F) as u16) << 8) | bytes[1] as u16;
    let mantissa = u64::from_be_bytes([
        bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7], bytes[8], bytes[9],
    ]);
    if exponent == 0 && mantissa == 0 {
        return Ok(0.0);
    }
    let fraction = mantissa as f64 / (1u64 << 63) as f64;
    Ok(sign * fraction * 2f64.powi(exponent as i32 - 16383))
}

fn resolve_path(value: &Value) -> BuiltinResult<PathBuf> {
    match value {
        Value::String(s) => normalize_path(s),
        Value::CharArray(ca) if ca.rows == 1 => {
            let text: String = ca.data.iter().collect();
            normalize_path(&text)
        }
        Value::StringArray(sa) if sa.data.len() == 1 => normalize_path(&sa.data[0]),
        _ => Err(audioinfo_error(&AUDIOINFO_ERROR_ARGUMENT)),
    }
}

fn normalize_path(raw: &str) -> BuiltinResult<PathBuf> {
    if raw.trim().is_empty() {
        return Err(audioinfo_error_with(
            &AUDIOINFO_ERROR_ARGUMENT,
            "audioinfo: filename must not be empty",
        ));
    }
    let expanded = expand_user_path(raw, BUILTIN_NAME)
        .map_err(|msg| audioinfo_error_with(&AUDIOINFO_ERROR_ARGUMENT, msg))?;
    Ok(Path::new(&expanded).to_path_buf())
}

fn read_u32_le(bytes: &[u8], pos: usize) -> Result<u32, String> {
    if pos + 4 > bytes.len() {
        return Err("unexpected end of file".to_string());
    }
    Ok(u32::from_le_bytes([
        bytes[pos],
        bytes[pos + 1],
        bytes[pos + 2],
        bytes[pos + 3],
    ]))
}

fn read_u32_be(bytes: &[u8], pos: usize) -> Result<u32, String> {
    if pos + 4 > bytes.len() {
        return Err("unexpected end of file".to_string());
    }
    Ok(u32::from_be_bytes([
        bytes[pos],
        bytes[pos + 1],
        bytes[pos + 2],
        bytes[pos + 3],
    ]))
}

fn read_u64_le(bytes: &[u8], pos: usize) -> Result<u64, String> {
    if pos + 8 > bytes.len() {
        return Err("unexpected end of file".to_string());
    }
    Ok(u64::from_le_bytes([
        bytes[pos],
        bytes[pos + 1],
        bytes[pos + 2],
        bytes[pos + 3],
        bytes[pos + 4],
        bytes[pos + 5],
        bytes[pos + 6],
        bytes[pos + 7],
    ]))
}

fn find_signature(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use runmat_time::unix_timestamp_ms;
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_ID: AtomicU64 = AtomicU64::new(0);

    fn temp_path(ext: &str) -> PathBuf {
        let millis = unix_timestamp_ms();
        let unique = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        let mut path = std::env::temp_dir();
        path.push(format!(
            "runmat_audioinfo_{}_{}_{}.{}",
            std::process::id(),
            millis,
            unique,
            ext
        ));
        path
    }

    fn field<'a>(value: &'a Value, name: &str) -> &'a Value {
        let Value::Struct(st) = value else {
            panic!("expected struct");
        };
        st.fields
            .get(name)
            .unwrap_or_else(|| panic!("missing {name}"))
    }

    fn wav_fixture(sample_rate: u32, channels: u16, bits: u16, frames: u32) -> Vec<u8> {
        let block_align = channels * (bits / 8);
        let byte_rate = sample_rate * block_align as u32;
        let data_size = frames * block_align as u32;
        let riff_size = 36 + data_size;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&riff_size.to_le_bytes());
        bytes.extend_from_slice(b"WAVE");
        bytes.extend_from_slice(b"fmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&channels.to_le_bytes());
        bytes.extend_from_slice(&sample_rate.to_le_bytes());
        bytes.extend_from_slice(&byte_rate.to_le_bytes());
        bytes.extend_from_slice(&block_align.to_le_bytes());
        bytes.extend_from_slice(&bits.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&data_size.to_le_bytes());
        bytes.resize(bytes.len() + data_size as usize, 0);
        bytes
    }

    fn flac_fixture(sample_rate: u32, channels: u16, bits: u16, total_samples: u64) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"fLaC");
        bytes.push(0x80);
        bytes.extend_from_slice(&[0x00, 0x00, 0x22]);
        let mut streaminfo = vec![0u8; 34];
        streaminfo[0..2].copy_from_slice(&4096u16.to_be_bytes());
        streaminfo[2..4].copy_from_slice(&4096u16.to_be_bytes());
        let packed = ((sample_rate as u64) << 44)
            | (((channels as u64) - 1) << 41)
            | (((bits as u64) - 1) << 36)
            | (total_samples & 0x000F_FFFF_FFFF);
        streaminfo[10..18].copy_from_slice(&packed.to_be_bytes());
        bytes.extend_from_slice(&streaminfo);
        bytes
    }

    fn rf64_fixture(sample_rate: u32, channels: u16, bits: u16, frames: u64) -> Vec<u8> {
        let block_align = channels * (bits / 8);
        let byte_rate = sample_rate * block_align as u32;
        let data_size = frames * block_align as u64;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RF64");
        bytes.extend_from_slice(&u32::MAX.to_le_bytes());
        bytes.extend_from_slice(b"WAVE");
        bytes.extend_from_slice(b"ds64");
        bytes.extend_from_slice(&24u32.to_le_bytes());
        bytes.extend_from_slice(&(36u64 + data_size).to_le_bytes());
        bytes.extend_from_slice(&data_size.to_le_bytes());
        bytes.extend_from_slice(&frames.to_le_bytes());
        bytes.extend_from_slice(b"fmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&channels.to_le_bytes());
        bytes.extend_from_slice(&sample_rate.to_le_bytes());
        bytes.extend_from_slice(&byte_rate.to_le_bytes());
        bytes.extend_from_slice(&block_align.to_le_bytes());
        bytes.extend_from_slice(&bits.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&u32::MAX.to_le_bytes());
        bytes
    }

    fn ogg_vorbis_identification_fixture(granule: u64) -> Vec<u8> {
        let mut body = vec![0u8; 30];
        body[0] = 1;
        body[1..7].copy_from_slice(b"vorbis");
        body[11] = 2;
        body[12..16].copy_from_slice(&44_100u32.to_le_bytes());
        body[20..24].copy_from_slice(&128_000i32.to_le_bytes());

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"OggS");
        bytes.push(0);
        bytes.push(0);
        bytes.extend_from_slice(&granule.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.push(1);
        bytes.push(body.len() as u8);
        bytes.extend_from_slice(&body);
        bytes
    }

    fn audio_scan(prefix: Vec<u8>) -> AudioInfoScan {
        AudioInfoScan {
            file_size: prefix.len() as u64,
            prefix,
            tail: None,
        }
    }

    fn audio_scan_with_file_size(prefix: Vec<u8>, file_size: u64) -> AudioInfoScan {
        AudioInfoScan {
            prefix,
            tail: None,
            file_size,
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn audioinfo_descriptor_covers_core_form() {
        assert_eq!(
            AUDIOINFO_DESCRIPTOR.signatures[0].label,
            "info = audioinfo(filename)"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn audioinfo_reads_wav_metadata() {
        let path = temp_path("wav");
        fs::write(&path, wav_fixture(44_100, 2, 16, 4)).expect("write fixture");

        let info = block_on(audioinfo_builtin(Value::from(
            path.to_string_lossy().into_owned(),
        )))
        .expect("audioinfo");

        assert_eq!(field(&info, "Format"), &Value::String("WAV".to_string()));
        assert_eq!(
            field(&info, "CompressionMethod"),
            &Value::String("PCM".to_string())
        );
        assert_eq!(field(&info, "NumChannels"), &Value::Num(2.0));
        assert_eq!(field(&info, "SampleRate"), &Value::Num(44_100.0));
        assert_eq!(field(&info, "TotalSamples"), &Value::Num(4.0));
        assert_eq!(field(&info, "BitsPerSample"), &Value::Num(16.0));
        let _ = fs::remove_file(path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn audioinfo_reads_rf64_data_size_from_ds64() {
        let metadata =
            AudioMetadata::parse(&audio_scan(rf64_fixture(48_000, 2, 16, 5))).expect("rf64");
        assert_eq!(metadata.format, "WAV");
        assert_eq!(metadata.total_samples, Some(5));
        assert_eq!(metadata.num_channels, 2);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn audioinfo_keeps_rf64_sample_count_in_u64_space() {
        let block_align = 2u64 * (16u64 / 8);
        let frames = (u32::MAX as u64 / block_align) + 10;
        let metadata =
            AudioMetadata::parse(&audio_scan(rf64_fixture(48_000, 2, 16, frames))).expect("rf64");
        assert_eq!(metadata.total_samples, Some(frames));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn audioinfo_reads_flac_streaminfo() {
        let path = temp_path("flac");
        fs::write(&path, flac_fixture(48_000, 2, 24, 96_000)).expect("write fixture");

        let info = block_on(audioinfo_builtin(Value::from(
            path.to_string_lossy().into_owned(),
        )))
        .expect("audioinfo");

        assert_eq!(field(&info, "Format"), &Value::String("FLAC".to_string()));
        assert_eq!(field(&info, "NumChannels"), &Value::Num(2.0));
        assert_eq!(field(&info, "SampleRate"), &Value::Num(48_000.0));
        assert_eq!(field(&info, "TotalSamples"), &Value::Num(96_000.0));
        assert_eq!(field(&info, "Duration"), &Value::Num(2.0));
        assert_eq!(field(&info, "BitsPerSample"), &Value::Num(24.0));
        let _ = fs::remove_file(path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn audioinfo_rejects_unknown_data() {
        let path = temp_path("bin");
        fs::write(&path, b"not audio").expect("write fixture");

        let err = block_on(audioinfo_builtin(Value::from(
            path.to_string_lossy().into_owned(),
        )))
        .expect_err("format error");
        assert!(err.message().contains("unsupported audio format"));
        let _ = fs::remove_file(path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn audioinfo_parses_mp3_frame_header() {
        let mut bytes = vec![0xFF, 0xFB, 0x90, 0x64];
        bytes.resize(417, 0);
        let metadata = AudioMetadata::parse(&audio_scan(bytes)).expect("mp3 metadata");
        assert_eq!(metadata.format, "MP3");
        assert_eq!(metadata.compression_method, "MPEG Layer III");
        assert_eq!(metadata.num_channels, 2);
        assert_eq!(metadata.sample_rate, 44_100.0);
        assert_eq!(metadata.bit_rate, Some(128_000.0));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn audioinfo_mp3_duration_uses_actual_file_size() {
        let bytes = vec![0xFF, 0xFB, 0x90, 0x64];
        let metadata =
            AudioMetadata::parse(&audio_scan_with_file_size(bytes, 16_000)).expect("mp3 metadata");
        assert_eq!(metadata.total_samples, Some(44_100));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn audioinfo_large_non_ogg_scan_skips_tail_window() {
        let path = temp_path("wav");
        let bytes = wav_fixture(44_100, 1, 16, 600_000);
        fs::write(&path, &bytes).expect("write fixture");

        let scan = block_on(read_audioinfo_scan(&path)).expect("audio scan");
        assert_eq!(scan.file_size, bytes.len() as u64);
        assert!(scan.tail.is_none());

        let _ = fs::remove_file(path);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn audioinfo_ogg_large_file_without_tail_granule_reports_unknown_samples() {
        let scan = audio_scan_with_file_size(ogg_vorbis_identification_fixture(0), 1_000_000);
        let metadata = AudioMetadata::parse(&scan).expect("ogg metadata");
        assert_eq!(metadata.format, "OGG");
        assert_eq!(metadata.total_samples, None);
    }
}
