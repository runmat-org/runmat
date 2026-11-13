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
    let err = block_on(audioinfo_builtin(None, Value::Num(42.0))).expect_err("numeric filename");
    assert_eq!(err.identifier(), Some("RunMat:audioinfo:InvalidArgument"));
    let _count = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let err =
        block_on(audioinfo_builtin(None, Value::from("missing.wav"))).expect_err("output count");
    assert_eq!(err.identifier(), Some("RunMat:audioinfo:InvalidArgument"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn audioinfo_relative_filename_keeps_public_spelling_with_captured_owner() {
    let _lock = runmat_filesystem::provider_override_lock();
    let provider = Arc::new(runmat_filesystem::MemoryFsProvider::with_current_dir(
        "/audio",
    ));
    provider
        .write_project_path("/audio/sample.wav", &wav_fixture(8000, 1, 16, 2))
        .expect("fixture");
    let _owner = runmat_filesystem::replace_provider(provider);
    runmat_filesystem::set_current_dir("/audio").expect("cwd");
    let info = block_on(audioinfo_builtin(None, Value::from("sample.wav"))).expect("metadata");
    assert_eq!(
        field(&info, "Filename"),
        &Value::String("sample.wav".into())
    );
    assert_eq!(field(&info, "SampleRate"), &Value::Num(8000.0));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn audioinfo_rejects_numeric_resident_path_without_gather() {
    let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![1, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .expect("genuine resident allocation");
    let original = allocation.handle.wire_descriptor();
    let err = block_on(audioinfo_builtin(
        None,
        Value::GpuTensor(allocation.handle.clone()),
    ))
    .expect_err("numeric resident filename rejects");
    assert_eq!(err.identifier(), Some("RunMat:audioinfo:InvalidArgument"));
    assert_eq!(allocation.handle.wire_descriptor(), original);
    assert_eq!(allocation.probe.retirements(), 0);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn audioinfo_reads_wav_metadata() {
    let file = audio_file("wav", &wav_fixture(44_100, 2, 16, 4));
    let path = &file.path;

    let info = block_on(audioinfo_builtin(
        None,
        Value::from(path.to_string_lossy().into_owned()),
    ))
    .expect("audioinfo");

    assert_eq!(
        field(&info, "Format"),
        &Value::String("WAV".to_string().into())
    );
    assert_eq!(
        field(&info, "CompressionMethod"),
        &Value::String("PCM".to_string().into())
    );
    assert_eq!(field(&info, "NumChannels"), &Value::Num(2.0));
    assert_eq!(field(&info, "SampleRate"), &Value::Num(44_100.0));
    assert_eq!(field(&info, "TotalSamples"), &Value::Num(4.0));
    assert_eq!(field(&info, "BitsPerSample"), &Value::Num(16.0));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn audioinfo_reads_rf64_data_size_from_ds64() {
    let metadata = AudioMetadata::parse(&audio_scan(rf64_pcm16_wav(
        48_000,
        2,
        &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    )))
    .expect("rf64");
    assert_eq!(metadata.format, "WAV");
    assert_eq!(metadata.total_samples, Some(5));
    assert_eq!(metadata.num_channels, 2);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn audioinfo_keeps_rf64_sample_count_in_u64_space() {
    let block_align = 2u64 * (16u64 / 8);
    let frames = (u32::MAX as u64 / block_align) + 10;
    let prefix = rf64_fixture(48_000, 2, 16, frames);
    // Model a sparse/virtual file: the bounded prefix is present, while the
    // filesystem metadata reports the genuine 64-bit payload extent.
    let scan = AudioInfoScan {
        file_size: prefix.len() as u64 + frames * block_align,
        prefix,
        tail: None,
    };
    let metadata = AudioMetadata::parse(&scan).expect("rf64");
    assert_eq!(metadata.total_samples, Some(frames));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn audioinfo_rejects_truncated_riff_and_rf64_data_extents() {
    let mut riff = wav_fixture(8_000, 1, 16, 2);
    riff.truncate(riff.len() - 2);
    let error = AudioMetadata::parse(&audio_scan(riff)).expect_err("truncated RIFF data");
    assert!(error.contains("extends past end of file"));

    let mut rf64 = rf64_pcm16_wav(8_000, 1, &[0, 1]);
    rf64.truncate(rf64.len() - 2);
    let error = AudioMetadata::parse(&audio_scan(rf64)).expect_err("truncated RF64 data");
    assert!(error.contains("extends past end of file"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn audioinfo_reads_flac_streaminfo() {
    let file = audio_file("flac", &flac_fixture(48_000, 2, 24, 96_000));
    let path = &file.path;

    let info = block_on(audioinfo_builtin(
        None,
        Value::from(path.to_string_lossy().into_owned()),
    ))
    .expect("audioinfo");

    assert_eq!(
        field(&info, "Format"),
        &Value::String("FLAC".to_string().into())
    );
    assert_eq!(field(&info, "NumChannels"), &Value::Num(2.0));
    assert_eq!(field(&info, "SampleRate"), &Value::Num(48_000.0));
    assert_eq!(field(&info, "TotalSamples"), &Value::Num(96_000.0));
    assert_eq!(field(&info, "Duration"), &Value::Num(2.0));
    assert_eq!(field(&info, "BitsPerSample"), &Value::Num(24.0));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn audioinfo_rejects_unknown_data() {
    let file = audio_file("bin", b"not audio");
    let path = &file.path;

    let err = block_on(audioinfo_builtin(
        None,
        Value::from(path.to_string_lossy().into_owned()),
    ))
    .expect_err("format error");
    assert!(err.message().contains("unsupported audio format"));
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
    let _lock = runmat_filesystem::provider_override_lock();
    let provider = std::sync::Arc::new(runmat_filesystem::MemoryFsProvider::with_current_dir(
        "/audio",
    ));
    let path = PathBuf::from("/audio/large.wav");
    let bytes = wav_fixture(44_100, 1, 16, 600_000);
    provider
        .write_project_path(&path, &bytes)
        .expect("write fixture");
    let _owner = runmat_filesystem::replace_provider(provider);

    let input = FsInputLease::capture(path, &AUDIOINFO_ERROR_IO, AUDIOINFO_BUILTIN_NAME)
        .expect("captured input");
    let mut cooperation = AudioCooperation::new(None, AUDIOINFO_BUILTIN_NAME);
    let scan = block_on(read_audioinfo_scan(&input, &mut cooperation)).expect("audio scan");
    assert_eq!(scan.file_size, bytes.len() as u64);
    assert_eq!(scan.prefix.len() as u64, MAX_AUDIOINFO_PREFIX_BYTES);
    assert!(scan.tail.is_none());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn audioinfo_ogg_large_file_without_tail_granule_reports_unknown_samples() {
    let scan = audio_scan_with_file_size(ogg_vorbis_identification_fixture(0), 1_000_000);
    let metadata = AudioMetadata::parse(&scan).expect("ogg metadata");
    assert_eq!(metadata.format, "OGG");
    assert_eq!(metadata.total_samples, None);
}
