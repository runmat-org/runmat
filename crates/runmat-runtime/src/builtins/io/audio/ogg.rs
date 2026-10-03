use super::*;

pub(super) fn parse_ogg_vorbis(scan: &AudioInfoScan) -> Result<AudioMetadata, String> {
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

pub(super) struct OggPage {
    pub(super) granule_position: u64,
    pub(super) body: Vec<u8>,
    pub(super) next_pos: usize,
}

pub(super) fn parse_ogg_page(bytes: &[u8], pos: usize) -> Result<OggPage, String> {
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

pub(super) fn find_last_ogg_granule(bytes: &[u8]) -> Option<u64> {
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
