use super::*;
use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy};
use futures::executor::block_on;
use runmat_execution::cancellation::CancellationSource;
use runmat_execution::CancellationReason;
use std::cell::Cell;
use std::io::{self, Cursor};
use std::num::NonZeroU64;
use std::rc::Rc;

struct CountedReader {
    source: Cursor<Vec<u8>>,
    byte_budget: usize,
    transferred: usize,
    seeks: usize,
}

impl CountedReader {
    fn new(bytes: Vec<u8>, byte_budget: usize) -> Self {
        Self {
            source: Cursor::new(bytes),
            byte_budget,
            transferred: 0,
            seeks: 0,
        }
    }
}

impl Read for CountedReader {
    fn read(&mut self, destination: &mut [u8]) -> io::Result<usize> {
        if self.transferred == self.byte_budget {
            return Err(io::Error::other("read budget exhausted"));
        }
        let allowed = destination.len().min(self.byte_budget - self.transferred);
        let read = self.source.read(&mut destination[..allowed])?;
        self.transferred += read;
        Ok(read)
    }
}

impl Seek for CountedReader {
    fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
        self.seeks += 1;
        self.source.seek(position)
    }
}

fn f32_sample(sample: &[u8]) -> f32 {
    f32::from_le_bytes([sample[0], sample[1], sample[2], sample[3]])
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn one_pass_reads_each_selected_frame_once_under_a_payload_sized_budget() {
    const FRAMES: usize = 4096;
    const CHANNELS: usize = 2;
    const BLOCK_ALIGN: usize = 8;
    let mut payload = Vec::new();
    for frame in 0..FRAMES {
        payload.extend_from_slice(&(frame as f32).to_le_bytes());
        payload.extend_from_slice(&(-(frame as f32)).to_le_bytes());
    }
    let payload_bytes = payload.len();
    assert_eq!(payload_bytes, FRAMES * BLOCK_ALIGN);
    let old_channel_major_bytes = CHANNELS * ((FRAMES - 1) * BLOCK_ALIGN + 4);
    assert_eq!(old_channel_major_bytes, 65_528);
    assert!(old_channel_major_bytes > payload_bytes);
    let geometry = StreamGeometry {
        data_offset: 0,
        channels: CHANNELS,
        start: 0,
        end: FRAMES,
        block_align: BLOCK_ALIGN,
        bytes_per_channel: 4,
    };
    let mut reader = CountedReader::new(payload.clone(), payload_bytes);
    let mut cooperation = AudioCooperation::new(None, AUDIOREAD_BUILTIN_NAME);
    let output = block_on(collect_stream_samples(
        &mut reader,
        Path::new("/audio/count.wav"),
        geometry,
        &mut cooperation,
        f32_sample,
    ))
    .expect("one pass fits the file-sized read budget");
    assert_eq!(reader.transferred, payload_bytes);
    assert_eq!(reader.seeks, 2, "4096 decoded samples per chunk");
    assert_eq!(output.len(), FRAMES * CHANNELS);
    assert_eq!(output[0], 0.0);
    assert_eq!(output[FRAMES - 1], (FRAMES - 1) as f32);
    assert_eq!(output[FRAMES].to_bits(), (-0.0f32).to_bits());
    assert_eq!(output[2 * FRAMES - 1], -((FRAMES - 1) as f32));

    let mut short_reader = CountedReader::new(payload, payload_bytes - 1);
    let error = block_on(collect_stream_samples(
        &mut short_reader,
        Path::new("/audio/count.wav"),
        geometry,
        &mut cooperation,
        f32_sample,
    ))
    .expect_err("a genuinely short read must not publish partial samples");
    assert_eq!(error.identifier(), AUDIOREAD_ERROR_IO.identifier);
    assert_eq!(short_reader.transferred, payload_bytes - 1);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn eight_channels_reduce_seeks_and_transfer_volume_together() {
    const FRAMES: usize = 4096;
    const CHANNELS: usize = 8;
    const BLOCK_ALIGN: usize = 32;
    let payload_bytes = FRAMES * BLOCK_ALIGN;
    let old_chunks_per_channel = FRAMES.div_ceil(INPUT_CHUNK_BYTES / BLOCK_ALIGN);
    let old_seeks = CHANNELS * old_chunks_per_channel;
    let old_bytes =
        CHANNELS * ((FRAMES - old_chunks_per_channel) * BLOCK_ALIGN + 4 * old_chunks_per_channel);
    assert_eq!(old_seeks, 16);
    assert_eq!(old_bytes, 1_048_128);
    let mut reader = CountedReader::new(vec![0; payload_bytes], payload_bytes);
    let mut cooperation = AudioCooperation::new(None, AUDIOREAD_BUILTIN_NAME);
    let output = block_on(collect_stream_samples(
        &mut reader,
        Path::new("/audio/eight.wav"),
        StreamGeometry {
            data_offset: 0,
            channels: CHANNELS,
            start: 0,
            end: FRAMES,
            block_align: BLOCK_ALIGN,
            bytes_per_channel: 4,
        },
        &mut cooperation,
        f32_sample,
    ))
    .expect("one pass over eight channels");
    assert_eq!(output.len(), FRAMES * CHANNELS);
    assert!(output.iter().all(|sample| sample.to_bits() == 0));
    assert_eq!(reader.transferred, payload_bytes);
    assert_eq!(
        reader.seeks, 8,
        "sample-work cap permits 512 frames per chunk"
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn partial_range_skips_unselected_frames_but_reads_frame_padding_once() {
    let mut payload = Vec::new();
    for bits in [
        [1.0f32.to_bits(), 2.0f32.to_bits()],
        [(-0.0f32).to_bits(), 0x7fc1_2345],
        [3.0f32.to_bits(), 4.0f32.to_bits()],
    ] {
        for sample in bits {
            payload.extend_from_slice(&sample.to_le_bytes());
        }
        payload.extend_from_slice(&[0xa5; 4]);
    }
    let geometry = StreamGeometry {
        data_offset: 0,
        channels: 2,
        start: 1,
        end: 3,
        block_align: 12,
        bytes_per_channel: 4,
    };
    let mut reader = CountedReader::new(payload, 24);
    let mut cooperation = AudioCooperation::new(None, AUDIOREAD_BUILTIN_NAME);
    let output = block_on(collect_stream_samples(
        &mut reader,
        Path::new("/audio/padded.wav"),
        geometry,
        &mut cooperation,
        f32_sample,
    ))
    .expect("selected padded frames");
    assert_eq!(reader.transferred, 24);
    assert_eq!(reader.seeks, 1);
    assert_eq!(
        output
            .iter()
            .map(|sample| sample.to_bits())
            .collect::<Vec<_>>(),
        [
            (-0.0f32).to_bits(),
            3.0f32.to_bits(),
            0x7fc1_2345,
            4.0f32.to_bits()
        ]
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn empty_selection_never_reads_or_seeks() {
    let mut reader = CountedReader::new(vec![0; 16], 0);
    let mut cooperation = AudioCooperation::new(None, AUDIOREAD_BUILTIN_NAME);
    let output = block_on(collect_stream_samples(
        &mut reader,
        Path::new("/audio/empty.wav"),
        StreamGeometry {
            data_offset: 0,
            channels: 2,
            start: 0,
            end: 0,
            block_align: 8,
            bytes_per_channel: 4,
        },
        &mut cooperation,
        f32_sample,
    ))
    .expect("empty selection");
    assert!(output.is_empty());
    assert_eq!(reader.transferred, 0);
    assert_eq!(reader.seeks, 0);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn maximum_valid_block_alignment_reads_each_complete_frame() {
    const BLOCK_ALIGN: usize = u16::MAX as usize;
    let mut payload = vec![0xa5; 2 * BLOCK_ALIGN];
    for (frame, first, second) in [(0, 1.25f32, -2.5f32), (1, 3.5f32, -4.75f32)] {
        let at = frame * BLOCK_ALIGN;
        payload[at..at + 4].copy_from_slice(&first.to_le_bytes());
        payload[at + 4..at + 8].copy_from_slice(&second.to_le_bytes());
    }
    let mut reader = CountedReader::new(payload, 2 * BLOCK_ALIGN);
    let mut cooperation = AudioCooperation::new(None, AUDIOREAD_BUILTIN_NAME);
    let output = block_on(collect_stream_samples(
        &mut reader,
        Path::new("/audio/large-block.wav"),
        StreamGeometry {
            data_offset: 0,
            channels: 2,
            start: 0,
            end: 2,
            block_align: BLOCK_ALIGN,
            bytes_per_channel: 4,
        },
        &mut cooperation,
        f32_sample,
    ))
    .expect("maximum valid WAVE block alignment");
    assert_eq!(output, [1.25, 3.5, -2.5, -4.75]);
    assert_eq!(reader.transferred, 2 * BLOCK_ALIGN);
    assert_eq!(reader.seeks, 2);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn cancellation_after_first_chunk_discards_initialized_output() {
    let source = Rc::new(CancellationSource::new());
    let cancel = source.clone();
    let turns = Rc::new(Cell::new(0));
    let counted = turns.clone();
    let budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
        source.token(),
        move || {
            counted.set(counted.get() + 1);
            if counted.get() == 5 {
                cancel.cancel(CancellationReason::User);
            }
            Box::pin(async { Ok(()) })
        },
    );
    let mut cooperation = AudioCooperation::with_budget(budget, AUDIOREAD_BUILTIN_NAME);
    let mut reader = CountedReader::new(vec![0; 2 * INPUT_CHUNK_BYTES], 2 * INPUT_CHUNK_BYTES);
    let result = block_on(collect_stream_samples(
        &mut reader,
        Path::new("/audio/cancel.wav"),
        StreamGeometry {
            data_offset: 0,
            channels: 2,
            start: 0,
            end: INPUT_CHUNK_BYTES / 8,
            block_align: 8,
            bytes_per_channel: 4,
        },
        &mut cooperation,
        f32_sample,
    ));
    let error = result.expect_err("cancelled collection cannot return initialized output");
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert_eq!(
        turns.get(),
        5,
        "four initialization turns and one decoded chunk"
    );
    assert_eq!(reader.transferred, AUDIO_WORK_CHUNK * 4);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn wide_pcm8_frame_checkpoints_within_channels_without_rereading() {
    const CHANNELS: usize = 5000;
    let payload: Vec<u8> = (0..CHANNELS).map(|channel| channel as u8).collect();
    let geometry = StreamGeometry {
        data_offset: 0,
        channels: CHANNELS,
        start: 0,
        end: 1,
        block_align: CHANNELS,
        bytes_per_channel: 1,
    };
    let mut complete_reader = CountedReader::new(payload.clone(), CHANNELS);
    let mut unmetered = AudioCooperation::new(None, AUDIOREAD_BUILTIN_NAME);
    let complete = block_on(collect_stream_samples(
        &mut complete_reader,
        Path::new("/audio/wide.wav"),
        geometry,
        &mut unmetered,
        |sample| sample[0],
    ))
    .expect("one complete wide frame");
    assert_eq!(complete, payload);
    assert_eq!(complete_reader.transferred, CHANNELS);
    assert_eq!(complete_reader.seeks, 1);

    let source = Rc::new(CancellationSource::new());
    let decoded = Rc::new(Cell::new(0));
    let counted = decoded.clone();
    let cancel = source.clone();
    let turns = Rc::new(Cell::new(0));
    let observed_turns = turns.clone();
    let budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
        source.token(),
        move || {
            observed_turns.set(observed_turns.get() + 1);
            if counted.get() == AUDIO_WORK_CHUNK {
                cancel.cancel(CancellationReason::User);
            }
            Box::pin(async { Ok(()) })
        },
    );
    let mut cooperation = AudioCooperation::with_budget(budget, AUDIOREAD_BUILTIN_NAME);
    let mut reader = CountedReader::new(payload, CHANNELS);
    let decoded_samples = decoded.clone();
    let error = block_on(collect_stream_samples(
        &mut reader,
        Path::new("/audio/wide.wav"),
        geometry,
        &mut cooperation,
        move |sample| {
            decoded_samples.set(decoded_samples.get() + 1);
            sample[0]
        },
    ))
    .expect_err("cancellation inside the wide frame cannot publish a partial vector");
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    assert_eq!(decoded.get(), AUDIO_WORK_CHUNK);
    assert_eq!(
        turns.get(),
        3,
        "two initialization turns and one channel turn"
    );
    assert_eq!(reader.transferred, CHANNELS);
    assert_eq!(reader.seeks, 1);
}
