use super::*;

pub(super) struct AudioFileFixture {
    pub(super) path: PathBuf,
    _owner: runmat_filesystem::ProviderGuard,
    _lock: std::sync::MutexGuard<'static, ()>,
}

pub(super) fn audio_file(ext: &str, bytes: &[u8]) -> AudioFileFixture {
    let lock = runmat_filesystem::provider_override_lock();
    let path = PathBuf::from(format!("/audio/fixture.{ext}"));
    let provider = std::sync::Arc::new(runmat_filesystem::MemoryFsProvider::new());
    provider
        .write_project_path(&path, bytes)
        .expect("audio fixture bytes");
    let owner = runmat_filesystem::replace_provider(provider);
    AudioFileFixture {
        path,
        _owner: owner,
        _lock: lock,
    }
}

pub(super) fn field<'a>(value: &'a Value, name: &str) -> &'a Value {
    let Value::Struct(st) = value else {
        panic!("expected struct");
    };
    st.fields
        .get(name)
        .unwrap_or_else(|| panic!("missing {name}"))
}

pub(super) fn tensor(value: Value) -> Tensor {
    match value {
        Value::Tensor(t) => t,
        other => panic!("expected tensor, got {other:?}"),
    }
}

pub(super) fn output_list(value: Value) -> Vec<Value> {
    match value {
        Value::OutputList(values) => values
            .try_into_owned()
            .expect("unique test output list")
            .try_into_unbounded_vec()
            .expect("ordinary unbounded test output list"),
        other => panic!("expected output list, got {other:?}"),
    }
}

pub(super) fn wav_fixture(sample_rate: u32, channels: u16, bits: u16, frames: u32) -> Vec<u8> {
    let block_align = channels * (bits / 8);
    let byte_rate = sample_rate * block_align as u32;
    let data_size = frames * block_align as u32;
    let riff_size = 36 + data_size;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&riff_size.to_le_bytes());
    bytes.extend_from_slice(b"WAVE");
    bytes.extend_from_slice(b"fmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&channels.to_le_bytes());
    bytes.extend_from_slice(&sample_rate.to_le_bytes());
    bytes.extend_from_slice(&byte_rate.to_le_bytes());
    bytes.extend_from_slice(&block_align.to_le_bytes());
    bytes.extend_from_slice(&bits.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&data_size.to_le_bytes());
    bytes.resize(bytes.len() + data_size as usize, 0);
    bytes
}

pub(super) fn wav_with_payload(
    sample_rate: u32,
    channels: u16,
    bits: u16,
    format_tag: u16,
    payload: &[u8],
) -> Vec<u8> {
    let block_align = channels * (bits / 8);
    let byte_rate = sample_rate * block_align as u32;
    let riff_size = 36 + payload.len() as u32;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&riff_size.to_le_bytes());
    bytes.extend_from_slice(b"WAVE");
    bytes.extend_from_slice(b"fmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&format_tag.to_le_bytes());
    bytes.extend_from_slice(&channels.to_le_bytes());
    bytes.extend_from_slice(&sample_rate.to_le_bytes());
    bytes.extend_from_slice(&byte_rate.to_le_bytes());
    bytes.extend_from_slice(&block_align.to_le_bytes());
    bytes.extend_from_slice(&bits.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    bytes.extend_from_slice(payload);
    if !payload.len().is_multiple_of(2) {
        bytes.push(0);
    }
    bytes
}

pub(super) fn pcm16_wav(sample_rate: u32, channels: u16, samples_interleaved: &[i16]) -> Vec<u8> {
    let mut payload = Vec::new();
    for sample in samples_interleaved {
        payload.extend_from_slice(&sample.to_le_bytes());
    }
    wav_with_payload(sample_rate, channels, 16, 1, &payload)
}

pub(super) fn pcm8_wav(sample_rate: u32, channels: u16, samples_interleaved: &[u8]) -> Vec<u8> {
    wav_with_payload(sample_rate, channels, 8, 1, samples_interleaved)
}

pub(super) fn pcm24_wav(sample_rate: u32, channels: u16, samples_interleaved: &[i32]) -> Vec<u8> {
    let mut payload = Vec::new();
    for sample in samples_interleaved {
        assert!((-8_388_608..=8_388_607).contains(sample));
        let bytes = sample.to_le_bytes();
        payload.extend_from_slice(&bytes[..3]);
    }
    wav_with_payload(sample_rate, channels, 24, 1, &payload)
}

pub(super) fn pcm32_wav(sample_rate: u32, channels: u16, samples_interleaved: &[i32]) -> Vec<u8> {
    let mut payload = Vec::new();
    for sample in samples_interleaved {
        payload.extend_from_slice(&sample.to_le_bytes());
    }
    wav_with_payload(sample_rate, channels, 32, 1, &payload)
}

pub(super) fn extensible_pcm32_wav(
    sample_rate: u32,
    valid_bits: u16,
    samples_interleaved: &[i32],
) -> Vec<u8> {
    let mut bytes = Vec::new();
    let data_size = (samples_interleaved.len() * 4) as u32;
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(60u32 + data_size).to_le_bytes());
    bytes.extend_from_slice(b"WAVEfmt ");
    bytes.extend_from_slice(&40u32.to_le_bytes());
    bytes.extend_from_slice(&0xfffeu16.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&sample_rate.to_le_bytes());
    bytes.extend_from_slice(&(sample_rate * 4).to_le_bytes());
    bytes.extend_from_slice(&4u16.to_le_bytes());
    bytes.extend_from_slice(&32u16.to_le_bytes());
    bytes.extend_from_slice(&22u16.to_le_bytes());
    bytes.extend_from_slice(&valid_bits.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&[
        1, 0, 0, 0, 0, 0, 0x10, 0, 0x80, 0, 0, 0xaa, 0, 0x38, 0x9b, 0x71,
    ]);
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&data_size.to_le_bytes());
    for sample in samples_interleaved {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    bytes
}

pub(super) fn float32_wav(sample_rate: u32, channels: u16, samples_interleaved: &[f32]) -> Vec<u8> {
    let mut payload = Vec::new();
    for sample in samples_interleaved {
        payload.extend_from_slice(&sample.to_le_bytes());
    }
    wav_with_payload(sample_rate, channels, 32, 3, &payload)
}

pub(super) fn float64_wav(sample_rate: u32, channels: u16, samples_interleaved: &[f64]) -> Vec<u8> {
    let mut payload = Vec::new();
    for sample in samples_interleaved {
        payload.extend_from_slice(&sample.to_le_bytes());
    }
    wav_with_payload(sample_rate, channels, 64, 3, &payload)
}

pub(super) fn rf64_pcm16_wav(
    sample_rate: u32,
    channels: u16,
    samples_interleaved: &[i16],
) -> Vec<u8> {
    let mut payload = Vec::new();
    for sample in samples_interleaved {
        payload.extend_from_slice(&sample.to_le_bytes());
    }
    let bits = 16u16;
    let block_align = channels * (bits / 8);
    let byte_rate = sample_rate * block_align as u32;
    let data_size = payload.len() as u64;
    let riff_size = 36u64 + data_size;
    let sample_count = data_size / block_align as u64;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"RF64");
    bytes.extend_from_slice(&u32::MAX.to_le_bytes());
    bytes.extend_from_slice(b"WAVE");
    bytes.extend_from_slice(b"ds64");
    bytes.extend_from_slice(&28u32.to_le_bytes());
    bytes.extend_from_slice(&riff_size.to_le_bytes());
    bytes.extend_from_slice(&data_size.to_le_bytes());
    bytes.extend_from_slice(&sample_count.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(b"fmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&channels.to_le_bytes());
    bytes.extend_from_slice(&sample_rate.to_le_bytes());
    bytes.extend_from_slice(&byte_rate.to_le_bytes());
    bytes.extend_from_slice(&block_align.to_le_bytes());
    bytes.extend_from_slice(&bits.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&u32::MAX.to_le_bytes());
    bytes.extend_from_slice(&payload);
    bytes
}

pub(super) fn flac_fixture(
    sample_rate: u32,
    channels: u16,
    bits: u16,
    total_samples: u64,
) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"fLaC");
    bytes.push(0x80);
    bytes.extend_from_slice(&[0x00, 0x00, 0x22]);
    let mut streaminfo = vec![0u8; 34];
    streaminfo[0..2].copy_from_slice(&4096u16.to_be_bytes());
    streaminfo[2..4].copy_from_slice(&4096u16.to_be_bytes());
    let packed = ((sample_rate as u64) << 44)
        | (((channels as u64) - 1) << 41)
        | (((bits as u64) - 1) << 36)
        | (total_samples & 0x000F_FFFF_FFFF);
    streaminfo[10..18].copy_from_slice(&packed.to_be_bytes());
    bytes.extend_from_slice(&streaminfo);
    bytes
}

pub(super) fn rf64_fixture(sample_rate: u32, channels: u16, bits: u16, frames: u64) -> Vec<u8> {
    let block_align = channels * (bits / 8);
    let byte_rate = sample_rate * block_align as u32;
    let data_size = frames * block_align as u64;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"RF64");
    bytes.extend_from_slice(&u32::MAX.to_le_bytes());
    bytes.extend_from_slice(b"WAVE");
    bytes.extend_from_slice(b"ds64");
    bytes.extend_from_slice(&24u32.to_le_bytes());
    bytes.extend_from_slice(&(36u64 + data_size).to_le_bytes());
    bytes.extend_from_slice(&data_size.to_le_bytes());
    bytes.extend_from_slice(&frames.to_le_bytes());
    bytes.extend_from_slice(b"fmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&channels.to_le_bytes());
    bytes.extend_from_slice(&sample_rate.to_le_bytes());
    bytes.extend_from_slice(&byte_rate.to_le_bytes());
    bytes.extend_from_slice(&block_align.to_le_bytes());
    bytes.extend_from_slice(&bits.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&u32::MAX.to_le_bytes());
    bytes
}

pub(super) fn ogg_vorbis_identification_fixture(granule: u64) -> Vec<u8> {
    let mut body = vec![0u8; 30];
    body[0] = 1;
    body[1..7].copy_from_slice(b"vorbis");
    body[11] = 2;
    body[12..16].copy_from_slice(&44_100u32.to_le_bytes());
    body[20..24].copy_from_slice(&128_000i32.to_le_bytes());

    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"OggS");
    bytes.push(0);
    bytes.push(0);
    bytes.extend_from_slice(&granule.to_le_bytes());
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.push(1);
    bytes.push(body.len() as u8);
    bytes.extend_from_slice(&body);
    bytes
}

pub(super) fn audio_scan(prefix: Vec<u8>) -> AudioInfoScan {
    AudioInfoScan {
        file_size: prefix.len() as u64,
        prefix,
        tail: None,
    }
}

pub(super) fn audio_scan_with_file_size(prefix: Vec<u8>, file_size: u64) -> AudioInfoScan {
    AudioInfoScan {
        prefix,
        tail: None,
        file_size,
    }
}
