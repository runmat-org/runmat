use super::WaveSampleCoding;

pub(super) fn parse_wave_metadata(
    bytes: &[u8],
    file_size: u64,
) -> Result<(WaveFormat, u64), String> {
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
        let size = if is_rf64 && id == b"data" && declared_size == u32::MAX {
            rf64_data_size.ok_or_else(|| {
                "RF64 data chunk uses sentinel size without ds64 metadata".to_string()
            })?
        } else {
            u64::from(declared_size)
        };
        let chunk_end = (pos as u64)
            .checked_add(size)
            .ok_or_else(|| "WAVE chunk size overflows address space".to_string())?;
        if chunk_end > file_size {
            return Err("WAVE chunk extends past end of file".to_string());
        }
        if id == b"data" {
            data_bytes = Some(size);
            break;
        }
        let size = usize::try_from(size)
            .map_err(|_| "WAVE chunk size overflows address space".to_string())?;
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
    let data_bytes = data_bytes.ok_or_else(|| "WAVE file is missing a data chunk".to_string())?;
    Ok((fmt, data_bytes))
}

#[derive(Debug, Clone, Copy)]
pub(super) struct WaveFormat {
    pub(super) format_tag: u16,
    pub(super) subformat_tag: Option<u16>,
    pub(super) channels: u16,
    pub(super) sample_rate: u32,
    pub(super) byte_rate: u32,
    pub(super) block_align: u16,
    pub(super) bits_per_sample: u16,
    pub(super) valid_bits_per_sample: Option<u16>,
}

impl WaveFormat {
    pub(super) fn effective_format_tag(self) -> u16 {
        self.subformat_tag.unwrap_or(self.format_tag)
    }

    /// The encoded sample always occupies the container width. Extensible PCM
    /// precision is left-aligned inside that container, not a shorter stride.
    pub(super) fn storage_bits_per_sample(self) -> u16 {
        self.bits_per_sample
    }

    pub(super) fn sample_coding(self) -> Result<WaveSampleCoding, String> {
        let coding = match (self.effective_format_tag(), self.storage_bits_per_sample()) {
            (0x0001, 8) => WaveSampleCoding::Pcm8,
            (0x0001, 16) => WaveSampleCoding::Pcm16,
            (0x0001, 24) => WaveSampleCoding::Pcm24,
            (0x0001, 32) => WaveSampleCoding::Pcm32,
            (0x0003, 32) => WaveSampleCoding::Float32,
            (0x0003, 64) => WaveSampleCoding::Float64,
            (tag, bits) => {
                return Err(format!(
                    "unsupported WAVE sample encoding tag 0x{tag:04x} with {bits} bits per sample"
                ));
            }
        };
        Ok(coding)
    }
}

pub(super) fn parse_wave_fmt(chunk: &[u8]) -> Result<WaveFormat, String> {
    if chunk.len() < 16 {
        return Err("WAVE fmt chunk is too short".to_string());
    }
    let format_tag = u16::from_le_bytes([chunk[0], chunk[1]]);
    let channels = u16::from_le_bytes([chunk[2], chunk[3]]);
    let sample_rate = u32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
    let byte_rate = u32::from_le_bytes([chunk[8], chunk[9], chunk[10], chunk[11]]);
    let block_align = u16::from_le_bytes([chunk[12], chunk[13]]);
    let bits_per_sample = u16::from_le_bytes([chunk[14], chunk[15]]);
    if channels == 0 || sample_rate == 0 || block_align == 0 {
        return Err(
            "WAVE fmt chunk has invalid channel count, sample rate, or block alignment".to_string(),
        );
    }

    let mut subformat_tag = None;
    let mut valid_bits_per_sample = None;
    if format_tag == 0xFFFE {
        if chunk.len() < 40 {
            return Err("WAVE extensible fmt chunk is too short".to_string());
        }
        let cb_size = u16::from_le_bytes([chunk[16], chunk[17]]);
        if cb_size < 22 {
            return Err("WAVE extensible fmt chunk has invalid extension size".to_string());
        }
        let valid = u16::from_le_bytes([chunk[18], chunk[19]]);
        if valid > bits_per_sample {
            return Err("WAVE valid sample precision exceeds its container width".to_string());
        }
        valid_bits_per_sample = if valid == 0 { None } else { Some(valid) };
        subformat_tag = wave_extensible_subformat_tag(&chunk[24..40]);
    }
    Ok(WaveFormat {
        format_tag,
        subformat_tag,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        valid_bits_per_sample,
    })
}

pub(super) fn wave_compression_name(tag: u16) -> &'static str {
    match tag {
        0x0001 => "PCM",
        0x0003 => "IEEE Float",
        0x0006 => "A-law",
        0x0007 => "mu-law",
        0xFFFE => "Extensible",
        _ => "Unknown",
    }
}

pub(super) fn wave_extensible_subformat_tag(guid: &[u8]) -> Option<u16> {
    const BASE_TAIL: [u8; 12] = [
        0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71,
    ];
    if guid.len() == 16 && guid[2..4] == [0x00, 0x00] && guid[4..16] == BASE_TAIL {
        Some(u16::from_le_bytes([guid[0], guid[1]]))
    } else {
        None
    }
}

pub(super) fn read_u32_le(bytes: &[u8], pos: usize) -> Result<u32, String> {
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

pub(super) fn read_u32_be(bytes: &[u8], pos: usize) -> Result<u32, String> {
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

pub(super) fn read_u64_le(bytes: &[u8], pos: usize) -> Result<u64, String> {
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

pub(super) fn find_signature(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}
