use super::*;

pub(super) fn parse_flac(bytes: &[u8]) -> Result<AudioMetadata, String> {
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

pub(super) fn parse_aiff(bytes: &[u8]) -> Result<AudioMetadata, String> {
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
