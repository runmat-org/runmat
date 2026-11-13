use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn audioread_descriptor_covers_core_forms() {
    let labels: Vec<&str> = AUDIOREAD_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert_eq!(
        labels,
        vec![
            "y = audioread(filename)",
            "[y, Fs] = audioread(filename)",
            "y = audioread(filename, samples)",
            "[y, Fs] = audioread(filename, samples)",
            "y = audioread(filename, datatype)",
            "[y, Fs] = audioread(filename, samples, datatype)",
        ]
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn audioread_invalid_output_count_and_controls_precede_file_read() {
    let _count = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
    let err = block_on(audioread_builtin(
        None,
        Value::from("missing.wav"),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .expect_err("invalid output count");
    assert_eq!(err.identifier(), Some("RunMat:audioread:InvalidArgument"));
    drop(_count);
    let err = block_on(audioread_builtin(
        None,
        Value::Num(1.0),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .expect_err("numeric path");
    assert_eq!(err.identifier(), Some("RunMat:audioread:InvalidArgument"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn audioread_open_ended_range_after_last_frame_reports_format_error() {
    let file = audio_file("wav", &pcm16_wav(8_000, 1, &[0]));
    let path = &file.path;
    let range = Value::Tensor(Tensor::new(vec![2.0, f64::INFINITY], vec![1, 2]).expect("range"));
    let error = block_on(audioread_builtin(
        None,
        Value::from(path.to_string_lossy().into_owned()),
        crate::builtins::common::test_support::builtin_arguments(vec![range]),
    ))
    .expect_err("range beyond file");
    assert_eq!(
        error.identifier(),
        Some("RunMat:audioread:UnsupportedFormat")
    );
    assert!(error
        .message()
        .contains("exceeds the available audio frames"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn audioread_reads_mono_pcm16_as_normalized_double() {
    let file = audio_file("wav", &pcm16_wav(8_000, 1, &[-32768, 0, 32767]));
    let path = &file.path;

    let y = tensor(
        block_on(audioread_builtin(
            None,
            Value::from(path.to_string_lossy().into_owned()),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect("audioread"),
    );

    assert_eq!(y.shape(), &[3, 1]);
    assert_eq!(y.numeric_dtype(), NumericDType::F64);
    assert_eq!(y.materialize_f64()[0], -1.0);
    assert_eq!(y.materialize_f64()[1], 0.0);
    assert!((y.materialize_f64()[2] - (32767.0 / 32768.0)).abs() < 1e-12);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn audioread_stereo_output_is_column_major_by_channel() {
    let file = audio_file(
        "wav",
        &pcm16_wav(
            44_100,
            2,
            &[
                32767, -32768, // frame 1: left, right
                0, 16384, // frame 2: left, right
            ],
        ),
    );
    let path = &file.path;

    let y = tensor(
        block_on(audioread_builtin(
            None,
            Value::from(path.to_string_lossy().into_owned()),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect("audioread"),
    );

    assert_eq!(y.shape(), &[2, 2]);
    assert!((y.materialize_f64()[0] - (32767.0 / 32768.0)).abs() < 1e-12);
    assert_eq!(y.materialize_f64()[1], 0.0);
    assert_eq!(y.materialize_f64()[2], -1.0);
    assert_eq!(y.materialize_f64()[3], 0.5);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn audioread_sample_range_is_one_based_inclusive_and_returns_fs() {
    let file = audio_file("wav", &pcm16_wav(22_050, 1, &[-32768, -16384, 0, 16384]));
    let path = &file.path;
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let outputs = output_list(
        block_on(audioread_builtin(
            None,
            Value::from(path.to_string_lossy().into_owned()),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Tensor(
                Tensor::new(vec![2.0, 3.0], vec![1, 2]).expect("range"),
            )]),
        ))
        .expect("audioread"),
    );

    let y = tensor(outputs[0].clone());
    assert_eq!(outputs[1], Value::Num(22_050.0));
    assert_eq!(y.shape(), &[2, 1]);
    assert_eq!(y.materialize_f64(), vec![-0.5, 0.0]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn audioread_sample_range_accepts_documented_infinite_endpoint() {
    let range = Tensor::new(vec![2.0, f64::INFINITY], vec![1, 2]).expect("range");
    assert_eq!(
        parse_sample_range(&Value::Tensor(range)).expect("sample range"),
        AudioSampleRange {
            first: 2,
            last: None
        }
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn audioread_sample_range_rejects_all_typed_integer_classes() {
    let ranges = [
        IntegerStorage::I8(vec![1, 2]),
        IntegerStorage::I16(vec![1, 2]),
        IntegerStorage::I32(vec![1, 2]),
        IntegerStorage::I64(vec![1, 2]),
        IntegerStorage::U8(vec![1, 2]),
        IntegerStorage::U16(vec![1, 2]),
        IntegerStorage::U32(vec![1, 2]),
        IntegerStorage::U64(vec![1, 2]),
    ];
    for storage in ranges {
        let range = Tensor::new_integer(storage, vec![1, 2]).expect("range");
        let error =
            parse_sample_range(&Value::Tensor(range)).expect_err("integer range must reject");
        assert_eq!(error.identifier(), Some("RunMat:audioread:InvalidArgument"));
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn audioread_rejects_resident_inputs_before_provider_access() {
    let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![1, 2],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .expect("genuine unregistered allocation");
    let original = allocation.handle.wire_descriptor();
    assert!(
        crate::builtins::common::gpu_helpers::exact_provider_for_handle(&allocation.handle)
            .is_none()
    );
    let resident = Value::GpuTensor(allocation.handle.clone());
    let error = block_on(audioread_builtin(
        None,
        Value::from("unused.wav"),
        crate::builtins::common::test_support::builtin_arguments(vec![resident]),
    ))
    .expect_err("resident sample range must reject");
    assert_eq!(error.identifier(), Some("RunMat:audioread:InvalidArgument"));
    assert!(error
        .message()
        .contains("gpuArray inputs are not supported"));
    assert_eq!(allocation.handle.wire_descriptor(), original);
    assert_eq!(allocation.probe.retirements(), 0);
    drop(allocation.handle);
    assert_eq!(allocation.probe.retirements(), 1);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn audioread_infinite_endpoint_reads_through_last_frame() {
    let file = audio_file("wav", &pcm16_wav(8_000, 1, &[-4, 5, i16::MAX]));
    let path = &file.path;
    let range = Tensor::new(vec![2.0, f64::INFINITY], vec![1, 2]).expect("sample range tensor");

    let y = tensor(
        block_on(audioread_builtin(
            None,
            Value::from(path.to_string_lossy().into_owned()),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Tensor(range),
                Value::from("native"),
            ]),
        ))
        .expect("audioread"),
    );

    assert_eq!(y.shape(), &[2, 1]);
    assert_eq!(
        y.integer_storage(),
        Some(&IntegerStorage::I16(vec![5, i16::MAX]))
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn audioread_native_pcm8_uses_exact_uint8_storage() {
    let file = audio_file("wav", &pcm8_wav(11_025, 1, &[0, 128, 255]));
    let path = &file.path;

    let y = tensor(
        block_on(audioread_builtin(
            None,
            Value::from(path.to_string_lossy().into_owned()),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::from("native")]),
        ))
        .expect("audioread"),
    );

    assert_eq!(y.materialize_f64(), vec![0.0, 128.0, 255.0]);
    assert_eq!(
        y.integer_storage(),
        Some(&IntegerStorage::U8(vec![0, 128, 255]))
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn audioread_native_pcm16_preserves_exact_storage_and_channel_order() {
    let file = audio_file("wav", &pcm16_wav(44_100, 2, &[i16::MIN, i16::MAX, -2, 3]));
    let path = &file.path;

    let y = tensor(
        block_on(audioread_builtin(
            None,
            Value::from(path.to_string_lossy().into_owned()),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::from("native")]),
        ))
        .expect("audioread"),
    );

    assert_eq!(y.shape(), &[2, 2]);
    assert_eq!(
        y.integer_storage(),
        Some(&IntegerStorage::I16(vec![i16::MIN, -2, i16::MAX, 3]))
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn audioread_native_pcm16_preserves_exact_storage_for_sample_ranges() {
    let file = audio_file("wav", &pcm16_wav(8_000, 1, &[i16::MIN, -4, 5, i16::MAX]));
    let path = &file.path;

    let y = tensor(
        block_on(audioread_builtin(
            None,
            Value::from(path.to_string_lossy().into_owned()),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Tensor(Tensor::new(vec![2.0, 3.0], vec![1, 2]).expect("range")),
                Value::from("native"),
            ]),
        ))
        .expect("audioread"),
    );

    assert_eq!(y.shape(), &[2, 1]);
    assert_eq!(y.integer_storage(), Some(&IntegerStorage::I16(vec![-4, 5])));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn audioread_native_pcm24_left_aligns_into_exact_int32_storage() {
    let file = audio_file("wav", &pcm24_wav(8_000, 1, &[-8_388_608, 0, 8_388_607]));
    let path = &file.path;

    let y = tensor(
        block_on(audioread_builtin(
            None,
            Value::from(path.to_string_lossy().into_owned()),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::from("native")]),
        ))
        .expect("audioread"),
    );

    assert_eq!(
        y.integer_storage(),
        Some(&IntegerStorage::I32(vec![i32::MIN, 0, 2_147_483_392]))
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn audioread_native_pcm32_preserves_int32_extrema() {
    let file = audio_file("wav", &pcm32_wav(8_000, 1, &[i32::MIN, 0, i32::MAX]));
    let path = &file.path;

    let y = tensor(
        block_on(audioread_builtin(
            None,
            Value::from(path.to_string_lossy().into_owned()),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::from("native")]),
        ))
        .expect("audioread"),
    );

    assert_eq!(
        y.integer_storage(),
        Some(&IntegerStorage::I32(vec![i32::MIN, 0, i32::MAX]))
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn audioread_reads_float32_wave() {
    let file = audio_file("wav", &float32_wav(48_000, 1, &[-0.25, 0.0, 0.5]));
    let path = &file.path;

    let y = tensor(
        block_on(audioread_builtin(
            None,
            Value::from(path.to_string_lossy().into_owned()),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .expect("audioread"),
    );

    assert_eq!(y.shape(), &[3, 1]);
    assert_eq!(y.materialize_f64(), vec![-0.25, 0.0, 0.5]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn audioread_native_float32_wave_remains_single() {
    let file = audio_file("wav", &float32_wav(48_000, 1, &[-0.25, 0.0, 0.5]));
    let path = &file.path;

    let y = tensor(
        block_on(audioread_builtin(
            None,
            Value::from(path.to_string_lossy().into_owned()),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::from("native")]),
        ))
        .expect("audioread"),
    );

    assert_eq!(y.numeric_dtype(), NumericDType::F32);
    assert!(y.integer_storage().is_none());
    assert_eq!(y.materialize_f64(), vec![-0.25, 0.0, 0.5]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn audioread_native_float64_wave_remains_double() {
    let file = audio_file("wav", &float64_wav(48_000, 1, &[-0.25, 0.0, 0.5]));
    let path = &file.path;

    let y = tensor(
        block_on(audioread_builtin(
            None,
            Value::from(path.to_string_lossy().into_owned()),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::from("native")]),
        ))
        .expect("audioread"),
    );

    assert_eq!(y.numeric_dtype(), NumericDType::F64);
    assert!(y.integer_storage().is_none());
    assert_eq!(y.materialize_f64(), vec![-0.25, 0.0, 0.5]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn audioread_reads_rf64_data_size_from_ds64() {
    let file = audio_file("wav", &rf64_pcm16_wav(32_000, 1, &[0, 16384]));
    let path = &file.path;

    let outputs = {
        let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        output_list(
            block_on(audioread_builtin(
                None,
                Value::from(path.to_string_lossy().into_owned()),
                crate::builtins::common::test_support::builtin_arguments(Vec::new()),
            ))
            .expect("audioread"),
        )
    };

    let y = tensor(outputs[0].clone());
    assert_eq!(outputs[1], Value::Num(32_000.0));
    assert_eq!(y.shape(), &[2, 1]);
    assert_eq!(y.materialize_f64(), vec![0.0, 0.5]);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn audioread_reads_via_active_filesystem_provider() {
    let _lock = runmat_filesystem::provider_override_lock();
    let dir = tempfile::tempdir().expect("tempdir");
    let provider =
        runmat_filesystem::SandboxFsProvider::new(dir.path().to_path_buf()).expect("sandbox");
    let _guard = runmat_filesystem::replace_provider(Arc::new(provider));
    block_on(runmat_filesystem::write_async(
        "/audio.wav",
        pcm16_wav(16_000, 1, &[0, 16384]),
    ))
    .expect("provider write");

    let outputs = {
        let _out_guard =
            crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        output_list(
            block_on(audioread_builtin(
                None,
                Value::from("/audio.wav"),
                crate::builtins::common::test_support::builtin_arguments(Vec::new()),
            ))
            .expect("audioread"),
        )
    };

    let y = tensor(outputs[0].clone());
    assert_eq!(outputs[1], Value::Num(16_000.0));
    assert_eq!(y.shape(), &[2, 1]);
    assert_eq!(y.materialize_f64(), vec![0.0, 0.5]);
}
