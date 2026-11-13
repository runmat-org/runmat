//! Shared sample scatter for ordinary vectors and exact admitted storage.

use super::*;

pub(super) async fn collect_stream_samples<T: Copy + Default, R: Read + Seek>(
    file: &mut R,
    path: &Path,
    geometry: StreamGeometry,
    cooperation: &mut AudioCooperation,
    decode: impl Fn(&[u8]) -> T,
) -> BuiltinResult<Vec<T>> {
    if geometry.channels == 0 || geometry.block_align == 0 {
        return Err(format_error("invalid WAVE channel layout"));
    }
    let selected_frames = geometry
        .end
        .checked_sub(geometry.start)
        .ok_or_else(|| format_error("decoded audio dimensions overflow platform limits"))?;
    let sample_count = selected_frames
        .checked_mul(geometry.channels)
        .ok_or_else(|| format_error("decoded audio dimensions overflow platform limits"))?;
    cooperation.completed(0).await?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(sample_count)
        .map_err(|err| format_error(format!("decoded audio allocation failed: {err}")))?;
    while output.len() < sample_count {
        let initialized = (sample_count - output.len()).min(AUDIO_WORK_CHUNK);
        output.resize(output.len() + initialized, T::default());
        cooperation.completed(initialized).await?;
    }
    if selected_frames == 0 {
        return Ok(output);
    }
    let frames_per_chunk = (INPUT_CHUNK_BYTES / geometry.block_align)
        .clamp(1, (AUDIO_WORK_CHUNK / geometry.channels).max(1));
    let buffer_len = frames_per_chunk
        .checked_mul(geometry.block_align)
        .ok_or_else(|| format_error("audio input chunk overflows platform limits"))?;
    let mut buffer = Vec::new();
    buffer
        .try_reserve_exact(buffer_len)
        .map_err(|err| format_error(format!("audio input allocation failed: {err}")))?;
    buffer.resize(buffer_len, 0);
    fill_stream_samples(
        file,
        path,
        geometry,
        cooperation,
        &mut output,
        &mut buffer,
        decode,
    )
    .await?;
    Ok(output)
}

pub(super) async fn fill_stream_samples<T: Copy, R: Read + Seek>(
    file: &mut R,
    path: &Path,
    geometry: StreamGeometry,
    cooperation: &mut AudioCooperation,
    output: &mut [T],
    buffer: &mut [u8],
    decode: impl Fn(&[u8]) -> T,
) -> BuiltinResult<()> {
    let selected_frames = geometry
        .end
        .checked_sub(geometry.start)
        .ok_or_else(|| format_error("decoded audio dimensions overflow platform limits"))?;
    let sample_count = selected_frames
        .checked_mul(geometry.channels)
        .ok_or_else(|| format_error("decoded audio dimensions overflow platform limits"))?;
    if output.len() != sample_count {
        return Err(format_error("invalid decoded WAVE sample layout"));
    }
    visit_stream_samples(
        file,
        path,
        geometry,
        cooperation,
        buffer,
        |index, sample| {
            output[index] = decode(sample);
        },
    )
    .await
}

pub(super) async fn validate_stream_samples<R: Read + Seek>(
    file: &mut R,
    path: &Path,
    geometry: StreamGeometry,
    cooperation: &mut AudioCooperation,
    buffer: &mut [u8],
    coding: WaveSampleCoding,
) -> BuiltinResult<()> {
    visit_stream_samples(file, path, geometry, cooperation, buffer, |_, sample| {
        let _ = decode_wave_sample(sample, coding, false);
    })
    .await
}

async fn visit_stream_samples<R: Read + Seek>(
    file: &mut R,
    path: &Path,
    geometry: StreamGeometry,
    cooperation: &mut AudioCooperation,
    buffer: &mut [u8],
    mut visit: impl FnMut(usize, &[u8]),
) -> BuiltinResult<()> {
    let StreamGeometry {
        data_offset,
        channels,
        start,
        end,
        block_align,
        bytes_per_channel,
    } = geometry;
    if channels == 0 || block_align == 0 || bytes_per_channel == 0 {
        return Err(format_error("invalid WAVE channel layout"));
    }
    let minimum_block_align = channels
        .checked_mul(bytes_per_channel)
        .ok_or_else(|| format_error("WAVE channel layout overflows platform limits"))?;
    if minimum_block_align > block_align {
        return Err(format_error(
            "WAVE block alignment is smaller than the channel sample width",
        ));
    }
    let selected_frames = end
        .checked_sub(start)
        .ok_or_else(|| format_error("decoded audio dimensions overflow platform limits"))?;
    if start < end && bytes_per_channel > 8 {
        return Err(format_error("invalid decoded WAVE sample layout"));
    }
    if selected_frames == 0 {
        return Ok(());
    }
    let frames_per_chunk =
        (INPUT_CHUNK_BYTES / block_align).clamp(1, (AUDIO_WORK_CHUNK / channels).max(1));
    let buffer_len = frames_per_chunk
        .checked_mul(block_align)
        .ok_or_else(|| format_error("audio input chunk overflows platform limits"))?;
    if buffer.len() < buffer_len {
        return Err(format_error("audio input chunk capacity is insufficient"));
    }
    let mut frame = start;
    while frame < end {
        cooperation.completed(0).await?;
        let frames = (end - frame).min(frames_per_chunk);
        let frame_offset = frame
            .checked_mul(block_align)
            .ok_or_else(|| format_error("sample offset overflows platform limits"))?;
        let offset = data_offset
            .checked_add(
                u64::try_from(frame_offset)
                    .map_err(|_| format_error("sample offset overflows platform limits"))?,
            )
            .ok_or_else(|| format_error("sample offset overflows platform limits"))?;
        file.seek(SeekFrom::Start(offset))
            .map_err(|source| io_error(path, source))?;
        let size = frames
            .checked_mul(block_align)
            .ok_or_else(|| format_error("audio input chunk overflows platform limits"))?;
        file.read_exact(&mut buffer[..size])
            .map_err(|source| io_error(path, source))?;
        let mut scatter = |local_frame: usize, channel: usize| {
            let at = local_frame * block_align;
            let sample_at = at + channel * bytes_per_channel;
            let output_at = channel * selected_frames + (frame - start) + local_frame;
            visit(output_at, &buffer[sample_at..sample_at + bytes_per_channel]);
        };
        if channels <= AUDIO_WORK_CHUNK {
            for local_frame in 0..frames {
                for channel in 0..channels {
                    scatter(local_frame, channel);
                }
            }
            cooperation.completed(frames * channels).await?;
        } else {
            for local_frame in 0..frames {
                for first_channel in (0..channels).step_by(AUDIO_WORK_CHUNK) {
                    let next_channel = (first_channel + AUDIO_WORK_CHUNK).min(channels);
                    for channel in first_channel..next_channel {
                        scatter(local_frame, channel);
                    }
                    cooperation.completed(next_channel - first_channel).await?;
                }
            }
        }
        frame += frames;
    }
    Ok(())
}
