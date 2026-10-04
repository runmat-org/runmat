//! Post-header WAVE facts owned by the audio decoder, never inferred from a
//! path extension or an opaque placement signature.

use super::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub(super) struct AudioDecodeGeometry {
    pub(super) opened_extent: u64,
    pub(super) data_offset: u64,
    pub(super) data_len: u64,
    pub(super) coding: WaveSampleCoding,
    pub(super) datatype: AudioReadDataType,
    pub(super) channels: usize,
    pub(super) block_align: usize,
    pub(super) bytes_per_channel: usize,
    pub(super) start: usize,
    pub(super) end: usize,
    pub(super) sample_count: usize,
    pub(super) output_dtype: NumericDType,
    pub(super) output_payload_bytes: u64,
    pub(super) input_chunk_bytes: usize,
}

impl AudioDecodeGeometry {
    pub(super) fn digest(self) -> Digest {
        Digest::sha256(serde_json::to_vec(&self).expect("typed WAVE geometry serializes"))
    }
}
