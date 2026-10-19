//! Post-header WAVE facts owned by the audio decoder, never inferred from a
//! path extension or an opaque placement signature.

use super::*;
use runmat_builtins::AudioDecodeResourceFacts;

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub(in crate::builtins::io::audio) struct AudioDecodeGeometry {
    pub(in crate::builtins::io::audio) opened_extent: u64,
    pub(in crate::builtins::io::audio) data_offset: u64,
    pub(in crate::builtins::io::audio) data_len: u64,
    pub(in crate::builtins::io::audio) coding: WaveSampleCoding,
    pub(in crate::builtins::io::audio) datatype: AudioReadDataType,
    pub(in crate::builtins::io::audio) channels: usize,
    pub(in crate::builtins::io::audio) block_align: usize,
    pub(in crate::builtins::io::audio) bytes_per_channel: usize,
    pub(in crate::builtins::io::audio) start: usize,
    pub(in crate::builtins::io::audio) end: usize,
    pub(in crate::builtins::io::audio) sample_count: usize,
    pub(in crate::builtins::io::audio) output_dtype: NumericDType,
    pub(in crate::builtins::io::audio) output_payload_bytes: u64,
    pub(in crate::builtins::io::audio) input_chunk_bytes: usize,
}

impl AudioDecodeGeometry {
    pub(super) fn resource_facts(self) -> AudioDecodeResourceFacts {
        AudioDecodeResourceFacts {
            opened_extent: self.opened_extent,
            data_offset: self.data_offset,
            data_len: self.data_len,
            coding: self.coding,
            datatype: self.datatype,
            channels: self.channels,
            block_align: self.block_align,
            bytes_per_channel: self.bytes_per_channel,
            start: self.start,
            end: self.end,
            sample_count: self.sample_count,
            output_dtype: self.output_dtype,
            output_payload_bytes: self.output_payload_bytes,
            input_chunk_bytes: self.input_chunk_bytes,
        }
    }

    pub(super) fn digest(self) -> Digest {
        Digest::sha256(
            serde_json::to_vec(&self.resource_facts()).expect("typed WAVE facts serialize"),
        )
    }
}
