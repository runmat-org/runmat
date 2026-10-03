use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn audioinfo_descriptor_covers_core_form() {
    assert_eq!(
        AUDIOINFO_DESCRIPTOR.signatures[0].label,
        "info = audioinfo(filename)"
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn audioinfo_invalid_control_rejects_before_filesystem_read() {
    let err = block_on(audioinfo_builtin(Value::Num(42.0))).expect_err("numeric filename");
    assert_eq!(err.identifier(), Some("RunMat:audioinfo:InvalidArgument"));
    let _count = crate::output_count::push_output_count(Some(2));
    let err = block_on(audioinfo_builtin(Value::from("missing.wav"))).expect_err("output count");
    assert_eq!(err.identifier(), Some("RunMat:audioinfo:InvalidArgument"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn audioinfo_reads_wav_metadata() {
    let _lock = runmat_filesystem::provider_override_lock();
    let path = temp_path("wav");
    fs::write(&path, wav_fixture(44_100, 2, 16, 4)).expect("write fixture");

    let info = block_on(audioinfo_builtin(Value::from(
        path.to_string_lossy().into_owned(),
    )))
    .expect("audioinfo");

    assert_eq!(field(&info, "Format"), &Value::String("WAV".to_string()));
    assert_eq!(
        field(&info, "CompressionMethod"),
        &Value::String("PCM".to_string())
    );
    assert_eq!(field(&info, "NumChannels"), &Value::Num(2.0));
    assert_eq!(field(&info, "SampleRate"), &Value::Num(44_100.0));
    assert_eq!(field(&info, "TotalSamples"), &Value::Num(4.0));
    assert_eq!(field(&info, "BitsPerSample"), &Value::Num(16.0));
    let _ = fs::remove_file(path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn audioinfo_reads_rf64_data_size_from_ds64() {
    let metadata = AudioMetadata::parse(&audio_scan(rf64_fixture(48_000, 2, 16, 5))).expect("rf64");
    assert_eq!(metadata.format, "WAV");
    assert_eq!(metadata.total_samples, Some(5));
    assert_eq!(metadata.num_channels, 2);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn audioinfo_keeps_rf64_sample_count_in_u64_space() {
    let block_align = 2u64 * (16u64 / 8);
    let frames = (u32::MAX as u64 / block_align) + 10;
    let metadata =
        AudioMetadata::parse(&audio_scan(rf64_fixture(48_000, 2, 16, frames))).expect("rf64");
    assert_eq!(metadata.total_samples, Some(frames));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn audioinfo_reads_flac_streaminfo() {
    let _lock = runmat_filesystem::provider_override_lock();
    let path = temp_path("flac");
    fs::write(&path, flac_fixture(48_000, 2, 24, 96_000)).expect("write fixture");

    let info = block_on(audioinfo_builtin(Value::from(
        path.to_string_lossy().into_owned(),
    )))
    .expect("audioinfo");

    assert_eq!(field(&info, "Format"), &Value::String("FLAC".to_string()));
    assert_eq!(field(&info, "NumChannels"), &Value::Num(2.0));
    assert_eq!(field(&info, "SampleRate"), &Value::Num(48_000.0));
    assert_eq!(field(&info, "TotalSamples"), &Value::Num(96_000.0));
    assert_eq!(field(&info, "Duration"), &Value::Num(2.0));
    assert_eq!(field(&info, "BitsPerSample"), &Value::Num(24.0));
    let _ = fs::remove_file(path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn audioinfo_rejects_unknown_data() {
    let _lock = runmat_filesystem::provider_override_lock();
    let path = temp_path("bin");
    fs::write(&path, b"not audio").expect("write fixture");

    let err = block_on(audioinfo_builtin(Value::from(
        path.to_string_lossy().into_owned(),
    )))
    .expect_err("format error");
    assert!(err.message().contains("unsupported audio format"));
    let _ = fs::remove_file(path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn audioinfo_parses_mp3_frame_header() {
    let mut bytes = vec![0xFF, 0xFB, 0x90, 0x64];
    bytes.resize(417, 0);
    let metadata = AudioMetadata::parse(&audio_scan(bytes)).expect("mp3 metadata");
    assert_eq!(metadata.format, "MP3");
    assert_eq!(metadata.compression_method, "MPEG Layer III");
    assert_eq!(metadata.num_channels, 2);
    assert_eq!(metadata.sample_rate, 44_100.0);
    assert_eq!(metadata.bit_rate, Some(128_000.0));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn audioinfo_mp3_duration_uses_actual_file_size() {
    let bytes = vec![0xFF, 0xFB, 0x90, 0x64];
    let metadata =
        AudioMetadata::parse(&audio_scan_with_file_size(bytes, 16_000)).expect("mp3 metadata");
    assert_eq!(metadata.total_samples, Some(44_100));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn audioinfo_large_non_ogg_scan_skips_tail_window() {
    let path = temp_path("wav");
    let bytes = wav_fixture(44_100, 1, 16, 600_000);
    fs::write(&path, &bytes).expect("write fixture");

    let scan = block_on(read_audioinfo_scan(&path)).expect("audio scan");
    assert_eq!(scan.file_size, bytes.len() as u64);
    assert!(scan.tail.is_none());

    let _ = fs::remove_file(path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn audioinfo_ogg_large_file_without_tail_granule_reports_unknown_samples() {
    let scan = audio_scan_with_file_size(ogg_vorbis_identification_fixture(0), 1_000_000);
    let metadata = AudioMetadata::parse(&scan).expect("ogg metadata");
    assert_eq!(metadata.format, "OGG");
    assert_eq!(metadata.total_samples, None);
}
