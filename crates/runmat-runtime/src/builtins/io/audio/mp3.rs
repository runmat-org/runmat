use super::*;

pub(super) fn parse_mp3(bytes: &[u8], file_size: u64) -> Option<AudioMetadata> {
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

pub(super) fn skip_id3v2(bytes: &[u8]) -> usize {
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

pub(super) struct MpegFrame {
    pub(super) layer: &'static str,
    pub(super) sample_rate: u32,
    pub(super) bit_rate: u32,
    pub(super) channels: u16,
}

pub(super) fn parse_mpeg_audio_header(header: &[u8]) -> Option<MpegFrame> {
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
pub(super) enum MpegVersion {
    V1,
    V2,
    V25,
}

#[derive(Debug, Clone, Copy)]
pub(super) enum MpegLayer {
    LayerI,
    LayerII,
    LayerIII,
}

pub(super) fn mpeg_sample_rate(version: MpegVersion, index: u8) -> Option<u32> {
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

pub(super) fn mpeg_bit_rate(version: MpegVersion, layer: MpegLayer, index: u8) -> Option<u32> {
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
