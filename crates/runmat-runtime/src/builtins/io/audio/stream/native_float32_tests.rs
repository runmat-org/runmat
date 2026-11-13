use super::*;
use futures::executor::block_on;
use std::sync::Arc;

const INTERLEAVED_BITS: [u32; 6] = [
    0x8000_0000, // frame 1, channel 1: negative zero
    0x7f80_0000, // frame 1, channel 2: positive infinity
    0x7fc1_2345, // frame 2, channel 1: NaN with payload
    0x3fc0_0000, // frame 2, channel 2: 1.5
    0xff80_0000, // frame 3, channel 1: negative infinity
    0xc010_0000, // frame 3, channel 2: -2.25
];

fn float32_wave() -> Vec<u8> {
    let data_bytes = u32::try_from(INTERLEAVED_BITS.len() * 4).unwrap();
    let riff_bytes = 4 + 8 + 16 + 8 + data_bytes;
    let mut wave = Vec::new();
    wave.extend_from_slice(b"RIFF");
    wave.extend_from_slice(&riff_bytes.to_le_bytes());
    wave.extend_from_slice(b"WAVEfmt ");
    wave.extend_from_slice(&16u32.to_le_bytes());
    wave.extend_from_slice(&3u16.to_le_bytes()); // IEEE float
    wave.extend_from_slice(&2u16.to_le_bytes()); // channels
    wave.extend_from_slice(&8000u32.to_le_bytes());
    wave.extend_from_slice(&64000u32.to_le_bytes()); // bytes per second
    wave.extend_from_slice(&8u16.to_le_bytes()); // bytes per frame
    wave.extend_from_slice(&32u16.to_le_bytes());
    wave.extend_from_slice(b"data");
    wave.extend_from_slice(&data_bytes.to_le_bytes());
    for bits in INTERLEAVED_BITS {
        wave.extend_from_slice(&bits.to_le_bytes());
    }
    wave
}

fn with_input(test: impl FnOnce(&FsInputLease)) {
    let _lock = fs::provider_override_lock();
    let provider = Arc::new(fs::MemoryFsProvider::with_current_dir("/audio"));
    provider
        .write_project_path("/audio/native-float.wav", &float32_wave())
        .expect("Float32 WAVE fixture");
    let _owner = fs::replace_provider(provider);
    fs::set_current_dir("/audio").expect("fixture cwd");
    let input = FsInputLease::capture(
        PathBuf::from("native-float.wav"),
        &AUDIOREAD_ERROR_IO,
        AUDIOREAD_BUILTIN_NAME,
    )
    .expect("source owner");
    test(&input);
}

fn decode(
    input: &FsInputLease,
    range: Option<AudioSampleRange>,
    datatype: AudioReadDataType,
) -> DecodedAudio {
    let mut cooperation = AudioCooperation::new(None, AUDIOREAD_BUILTIN_NAME);
    block_on(decode_wave_stream(
        input,
        AudioreadOptions { range, datatype },
        &mut cooperation,
    ))
    .expect("Float32 WAVE decode")
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn native_float32_preserves_bits_and_column_major_channels() {
    with_input(|input| {
        let decoded = decode(input, None, AudioReadDataType::Native);
        assert_eq!(decoded.samples.shape(), &[3, 2]);
        assert_eq!(decoded.samples.numeric_dtype(), NumericDType::F32);
        assert_eq!(decoded.sample_rate, 8000.0);
        let actual: Vec<u32> = decoded
            .samples
            .as_f32_slice()
            .expect("single storage")
            .iter()
            .map(|sample| sample.to_bits())
            .collect();
        assert_eq!(
            actual,
            [
                INTERLEAVED_BITS[0],
                INTERLEAVED_BITS[2],
                INTERLEAVED_BITS[4],
                INTERLEAVED_BITS[1],
                INTERLEAVED_BITS[3],
                INTERLEAVED_BITS[5],
            ]
        );
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn native_float32_range_and_double_representation_remain_distinct() {
    with_input(|input| {
        let range = Some(AudioSampleRange {
            first: 2,
            last: Some(3),
        });
        let native = decode(input, range, AudioReadDataType::Native);
        assert_eq!(native.samples.shape(), &[2, 2]);
        let actual: Vec<u32> = native
            .samples
            .as_f32_slice()
            .expect("single storage")
            .iter()
            .map(|sample| sample.to_bits())
            .collect();
        assert_eq!(
            actual,
            [
                INTERLEAVED_BITS[2],
                INTERLEAVED_BITS[4],
                INTERLEAVED_BITS[3],
                INTERLEAVED_BITS[5],
            ]
        );

        let double = decode(input, range, AudioReadDataType::Double);
        assert_eq!(double.samples.shape(), &[2, 2]);
        assert_eq!(double.samples.numeric_dtype(), NumericDType::F64);
        assert!(double.samples.as_f64_slice().expect("double storage")[0].is_nan());
        assert_eq!(
            &double.samples.as_f64_slice().expect("double storage")[1..],
            &[f64::NEG_INFINITY, 1.5, -2.25]
        );
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn exact_capacity_scatter_matches_native_float32_column_major_bits() {
    let bytes = float32_wave();
    let mut source = std::io::Cursor::new(bytes);
    let mut cooperation = AudioCooperation::new(None, AUDIOREAD_BUILTIN_NAME);
    let mut output = [0.0_f32; 6];
    let mut input_chunk = [0_u8; INPUT_CHUNK_BYTES];
    let geometry = StreamGeometry {
        data_offset: 44,
        channels: 2,
        start: 0,
        end: 3,
        block_align: 8,
        bytes_per_channel: 4,
    };
    block_on(fill_stream_samples(
        &mut source,
        Path::new("native-float.wav"),
        geometry,
        &mut cooperation,
        &mut output,
        &mut input_chunk,
        |sample| f32::from_le_bytes([sample[0], sample[1], sample[2], sample[3]]),
    ))
    .expect("fixed-capacity scatter");
    assert_eq!(
        output.map(f32::to_bits),
        [
            INTERLEAVED_BITS[0],
            INTERLEAVED_BITS[2],
            INTERLEAVED_BITS[4],
            INTERLEAVED_BITS[1],
            INTERLEAVED_BITS[3],
            INTERLEAVED_BITS[5],
        ]
    );
}
