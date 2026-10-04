//! Immutable, typed WAVE decode facts at the audio contract boundary.
//! The runtime audio decoder derives these only after validating the opened
//! source; a placement service does not parse WAVE headers.

use runmat_value::NumericDType;

use super::AudioReadDataType;

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub enum AudioSampleCoding {
    Pcm8,
    Pcm16,
    Pcm24,
    Pcm32,
    Float32,
    Float64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub struct AudioDecodeResourceFacts {
    pub opened_extent: u64,
    pub data_offset: u64,
    pub data_len: u64,
    pub coding: AudioSampleCoding,
    pub datatype: AudioReadDataType,
    pub channels: usize,
    pub block_align: usize,
    pub bytes_per_channel: usize,
    pub start: usize,
    pub end: usize,
    pub sample_count: usize,
    pub output_dtype: NumericDType,
    pub output_payload_bytes: u64,
    pub input_chunk_bytes: usize,
}
