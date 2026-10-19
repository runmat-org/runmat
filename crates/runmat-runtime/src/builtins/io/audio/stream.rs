//! Bounded-input WAVE scan and range decoder. Output allocation is necessarily
//! proportional to requested frames; no whole-file byte copy is retained.

use super::*;

const INPUT_CHUNK_BYTES: usize = 64 * 1024;
const AUDIO_WORK_CHUNK: usize = 4096;

mod classify;
mod exact;
mod scatter;
pub(super) use exact::{
    decode_prepared_wave_stream_exact, validate_prepared_wave_stream_discarded,
};
use scatter::{collect_stream_samples, fill_stream_samples, validate_stream_samples};

#[cfg(test)]
#[path = "stream/tests.rs"]
mod tests;
#[cfg(test)]
#[path = "stream/fallback_tests.rs"]
mod fallback_tests;
#[cfg(test)]
#[path = "stream/native_float32_tests.rs"]
mod native_float32_tests;
#[cfg(test)]
#[path = "stream/one_pass_tests.rs"]
mod one_pass_tests;

#[derive(Clone, Copy)]
struct WaveStream {
    format: WaveFormat,
    opened_extent: u64,
    data_offset: u64,
    data_len: u64,
}

#[derive(Clone, Copy)]
struct StreamGeometry {
    data_offset: u64,
    channels: usize,
    start: usize,
    end: usize,
    block_align: usize,
    bytes_per_channel: usize,
}

fn format_error(message: impl Into<String>) -> RuntimeError {
    audioread_error_with(
        &AUDIOREAD_ERROR_FORMAT,
        format!("audioread: {}", message.into()),
    )
}

fn io_error(path: &Path, source: std::io::Error) -> RuntimeError {
    audioread_error_with_source(
        &AUDIOREAD_ERROR_IO,
        format!(
            "audioread: unable to read \"{}\" ({source})",
            path.display()
        ),
        source,
    )
}

pub(super) async fn decode_wave_stream(
    input: &FsInputLease,
    options: AudioreadOptions,
    cooperation: &mut AudioCooperation,
) -> BuiltinResult<DecodedAudio> {
    match prepare_wave_stream(input, options, cooperation).await? {
        Some(prepared) => decode_prepared_wave_stream(prepared, &input.path, cooperation).await,
        None => decode_whole_on_provider(input, options, cooperation).await,
    }
}

pub(super) struct PreparedWaveStream {
    file: OpenedAudioInput,
    parsed: WaveStream,
    pub(super) geometry: AudioDecodeGeometry,
}

impl PreparedWaveStream {
    pub(super) fn has_bounded_opened_source(&self) -> bool {
        self.file.opened_extent().is_some()
    }
}

pub(super) async fn prepare_wave_stream(
    input: &FsInputLease,
    options: AudioreadOptions,
    cooperation: &mut AudioCooperation,
) -> BuiltinResult<Option<PreparedWaveStream>> {
    cooperation.completed(0).await?;
    let mut file = match cooperation.await_io(input.open()).await? {
        Ok(file) => file,
        Err(source) if source.kind() == std::io::ErrorKind::Unsupported => {
            return Ok(None);
        }
        Err(source) => return Err(io_error(&input.path, source)),
    };
    let file_size = if let Some(extent) = file.opened_extent() {
        extent
    } else {
        match file.seek(SeekFrom::End(0)) {
            Ok(size) => size,
            Err(source) if source.kind() == std::io::ErrorKind::Unsupported => {
                return Ok(None);
            }
            Err(source) => return Err(io_error(&input.path, source)),
        }
    };
    if file_size < 4 {
        return Err(format_error("unsupported audio format"));
    }
    file.seek(SeekFrom::Start(0))
        .map_err(|source| io_error(&input.path, source))?;
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)
        .map_err(|source| io_error(&input.path, source))?;
    if &magic != b"RIFF" && &magic != b"RF64" {
        let classification =
            classify::non_wave(&mut file, &input.path, file_size, magic, cooperation).await?;
        return Err(format_error(classification));
    }
    let parsed = scan_wave(&mut file, &input.path, cooperation).await?;
    let geometry = AudioDecodeGeometry::for_wave(parsed, options)?;
    cooperation.completed(0).await?;
    Ok(Some(PreparedWaveStream {
        file,
        parsed,
        geometry,
    }))
}

pub(super) async fn decode_prepared_wave_stream(
    mut prepared: PreparedWaveStream,
    path: &Path,
    cooperation: &mut AudioCooperation,
) -> BuiltinResult<DecodedAudio> {
    decode_scanned_wave(
        &mut prepared.file,
        path,
        prepared.parsed,
        prepared.geometry,
        cooperation,
    )
    .await
}

async fn decode_seekable_wave<R: Read + Seek>(
    file: &mut R,
    path: &Path,
    options: AudioreadOptions,
    cooperation: &mut AudioCooperation,
) -> BuiltinResult<DecodedAudio> {
    let parsed = scan_wave(file, path, cooperation).await?;
    let geometry = AudioDecodeGeometry::for_wave(parsed, options)?;
    decode_scanned_wave(file, path, parsed, geometry, cooperation).await
}

impl AudioDecodeGeometry {
    fn for_wave(parsed: WaveStream, options: AudioreadOptions) -> BuiltinResult<Self> {
        let fmt = parsed.format;
        let coding = fmt.sample_coding().map_err(format_error)?;
        let channels = usize::from(fmt.channels);
        let block_align = usize::from(fmt.block_align);
        let bytes_per_channel = usize::from(fmt.bits_per_sample.div_ceil(8));
        let minimum_block_align = channels
            .checked_mul(bytes_per_channel)
            .ok_or_else(|| format_error("WAVE channel layout overflows platform limits"))?;
        if channels == 0
            || block_align == 0
            || bytes_per_channel == 0
            || minimum_block_align > block_align
        {
            return Err(format_error(
                "WAVE block alignment is smaller than the channel sample width",
            ));
        }
        let data_len = usize::try_from(parsed.data_len)
            .map_err(|_| format_error("RF64 data chunk is too large for this platform"))?;
        if data_len % block_align != 0 {
            return Err(format_error(
                "WAVE data chunk is not aligned to whole sample frames",
            ));
        }
        let total_frames = data_len / block_align;
        let (start, end) = match options.range {
            None => (0, total_frames),
            Some(AudioSampleRange { first, last }) => {
                let last = last.unwrap_or(total_frames);
                if first > total_frames || last > total_frames {
                    return Err(format_error(
                        "sample range exceeds the available audio frames",
                    ));
                }
                (first - 1, last)
            }
        };
        let sample_count = (end - start)
            .checked_mul(channels)
            .ok_or_else(|| format_error("decoded audio dimensions overflow platform limits"))?;
        let output_dtype = match (options.datatype, coding) {
            (AudioReadDataType::Double, _) => NumericDType::F64,
            (AudioReadDataType::Native, WaveSampleCoding::Pcm8) => NumericDType::U8,
            (AudioReadDataType::Native, WaveSampleCoding::Pcm16) => NumericDType::I16,
            (AudioReadDataType::Native, WaveSampleCoding::Pcm24 | WaveSampleCoding::Pcm32) => {
                NumericDType::I32
            }
            (AudioReadDataType::Native, WaveSampleCoding::Float32) => NumericDType::F32,
            (AudioReadDataType::Native, WaveSampleCoding::Float64) => NumericDType::F64,
        };
        let output_payload_bytes = sample_count
            .checked_mul(output_dtype.byte_size())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| format_error("decoded audio byte length overflows platform limits"))?;
        let frames_per_chunk =
            (INPUT_CHUNK_BYTES / block_align).clamp(1, (AUDIO_WORK_CHUNK / channels).max(1));
        let input_chunk_bytes = frames_per_chunk
            .checked_mul(block_align)
            .ok_or_else(|| format_error("audio input chunk overflows platform limits"))?;
        Ok(Self {
            opened_extent: parsed.opened_extent,
            data_offset: parsed.data_offset,
            data_len: parsed.data_len,
            coding,
            datatype: options.datatype,
            channels,
            block_align,
            bytes_per_channel,
            start,
            end,
            sample_count,
            output_dtype,
            output_payload_bytes,
            input_chunk_bytes,
        })
    }
}

async fn decode_scanned_wave<R: Read + Seek>(
    file: &mut R,
    path: &Path,
    parsed: WaveStream,
    facts: AudioDecodeGeometry,
    cooperation: &mut AudioCooperation,
) -> BuiltinResult<DecodedAudio> {
    let fmt = parsed.format;
    let coding = facts.coding;
    let shape = vec![facts.end - facts.start, facts.channels];
    let geometry = StreamGeometry {
        data_offset: facts.data_offset,
        channels: facts.channels,
        start: facts.start,
        end: facts.end,
        block_align: facts.block_align,
        bytes_per_channel: facts.bytes_per_channel,
    };
    let samples = match (facts.datatype, coding) {
        (AudioReadDataType::Native, WaveSampleCoding::Pcm8) => Tensor::new_integer(
            IntegerStorage::U8(
                collect_stream_samples(file, path, geometry, cooperation, |sample| sample[0])
                    .await?,
            ),
            shape,
        ),
        (AudioReadDataType::Native, WaveSampleCoding::Pcm16) => Tensor::new_integer(
            IntegerStorage::I16(
                collect_stream_samples(file, path, geometry, cooperation, |sample| {
                    i16::from_le_bytes([sample[0], sample[1]])
                })
                .await?,
            ),
            shape,
        ),
        (AudioReadDataType::Native, WaveSampleCoding::Pcm24) => Tensor::new_integer(
            IntegerStorage::I32(
                collect_stream_samples(file, path, geometry, cooperation, |sample| {
                    sign_extend_24(sample) << 8
                })
                .await?,
            ),
            shape,
        ),
        (AudioReadDataType::Native, WaveSampleCoding::Pcm32) => Tensor::new_integer(
            IntegerStorage::I32(
                collect_stream_samples(file, path, geometry, cooperation, |sample| {
                    i32::from_le_bytes([sample[0], sample[1], sample[2], sample[3]])
                })
                .await?,
            ),
            shape,
        ),
        (AudioReadDataType::Native, WaveSampleCoding::Float32) => Tensor::from_numeric_storage(
            runmat_value::NumericStorage::F32(
                collect_stream_samples(file, path, geometry, cooperation, |sample| {
                    f32::from_le_bytes([sample[0], sample[1], sample[2], sample[3]])
                })
                .await?,
            ),
            shape,
        ),
        (datatype, coding) => {
            let native = datatype == AudioReadDataType::Native;
            let out = collect_stream_samples(file, path, geometry, cooperation, |sample| {
                decode_wave_sample(sample, coding, native)
            })
            .await?;
            Tensor::new(out, shape)
        }
    }
    .map_err(|err| format_error(format!("decoded audio tensor shape is invalid: {err}")))?;
    cooperation.completed(0).await?;
    Ok(DecodedAudio {
        samples,
        sample_rate: f64::from(fmt.sample_rate),
    })
}

async fn decode_whole_on_provider(
    input: &FsInputLease,
    options: AudioreadOptions,
    cooperation: &mut AudioCooperation,
) -> BuiltinResult<DecodedAudio> {
    cooperation.completed(0).await?;
    let bytes = cooperation
        .await_io(input.provider.read(&input.path))
        .await?
        .map_err(|source| io_error(&input.path, source))?;
    cooperation.completed(0).await?;
    let mut cursor = std::io::Cursor::new(bytes.as_slice());
    let decoded = if bytes.starts_with(b"RIFF") || bytes.starts_with(b"RF64") {
        decode_seekable_wave(&mut cursor, &input.path, options, cooperation).await?
    } else {
        if bytes.len() < 4 {
            return Err(format_error(UNSUPPORTED_AUDIO_ERROR));
        }
        let magic: [u8; 4] = bytes[..4].try_into().expect("four-byte slice");
        let classification = classify::non_wave(
            &mut cursor,
            &input.path,
            bytes.len() as u64,
            magic,
            cooperation,
        )
        .await?;
        return Err(format_error(classification));
    };
    cooperation.completed(0).await?;
    Ok(decoded)
}

async fn scan_wave<R: Read + Seek>(
    file: &mut R,
    path: &Path,
    cooperation: &mut AudioCooperation,
) -> BuiltinResult<WaveStream> {
    let file_size = file
        .seek(SeekFrom::End(0))
        .map_err(|source| io_error(path, source))?;
    if file_size < 12 {
        return Err(format_error("RIFF/RF64 file is not a WAVE container"));
    }
    file.seek(SeekFrom::Start(0))
        .map_err(|source| io_error(path, source))?;
    let mut header = [0u8; 12];
    file.read_exact(&mut header)
        .map_err(|source| io_error(path, source))?;
    if (&header[..4] != b"RIFF" && &header[..4] != b"RF64") || &header[8..12] != b"WAVE" {
        return Err(format_error("RIFF/RF64 file is not a WAVE container"));
    }
    let is_rf64 = &header[..4] == b"RF64";
    let mut pos = 12u64;
    let mut fmt = None;
    let mut data = None;
    let mut rf64_data_size = None;
    while pos.checked_add(8).is_some_and(|end| end <= file_size) {
        cooperation.completed(8).await?;
        file.seek(SeekFrom::Start(pos))
            .map_err(|source| io_error(path, source))?;
        let mut chunk = [0u8; 8];
        file.read_exact(&mut chunk)
            .map_err(|source| io_error(path, source))?;
        let chunk_start = pos + 8;
        let declared = u32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
        let size = if is_rf64 && &chunk[..4] == b"data" && declared == u32::MAX {
            rf64_data_size.ok_or_else(|| {
                format_error("RF64 data chunk uses sentinel size without ds64 metadata")
            })?
        } else {
            u64::from(declared)
        };
        let chunk_end = chunk_start
            .checked_add(size)
            .ok_or_else(|| format_error("WAVE chunk size overflows platform limits"))?;
        if chunk_end > file_size {
            return Err(format_error("WAVE chunk extends past end of file"));
        }
        match &chunk[..4] {
            b"ds64" if is_rf64 => {
                if size < 24 {
                    return Err(format_error("RF64 ds64 chunk is too short"));
                }
                let mut ds64 = [0u8; 24];
                file.read_exact(&mut ds64)
                    .map_err(|source| io_error(path, source))?;
                rf64_data_size = Some(u64::from_le_bytes(
                    ds64[8..16].try_into().expect("fixed eight-byte slice"),
                ));
            }
            b"fmt " => {
                let mut bytes = [0u8; 40];
                let read_len = usize::try_from(size.min(40)).expect("at most forty");
                file.read_exact(&mut bytes[..read_len])
                    .map_err(|source| io_error(path, source))?;
                fmt = Some(parse_wave_fmt(&bytes[..read_len]).map_err(format_error)?);
            }
            b"data" => data = Some((chunk_start, size)),
            _ => {}
        }
        pos = chunk_end
            .checked_add(size % 2)
            .ok_or_else(|| format_error("WAVE chunk offset overflows platform limits"))?;
    }
    let format = fmt.ok_or_else(|| format_error("WAVE file is missing a fmt chunk"))?;
    let (data_offset, data_len) =
        data.ok_or_else(|| format_error("WAVE file is missing a data chunk"))?;
    Ok(WaveStream {
        format,
        opened_extent: file_size,
        data_offset,
        data_len,
    })
}
