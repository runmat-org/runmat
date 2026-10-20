use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn audioread_extensible_pcm24_in_32_uses_container_width_for_native_and_double() {
    let file = audio_file(
        "wav",
        &extensible_pcm32_wav(8_000, 24, &[i32::MIN, 0, 0x7fff_ff00]),
    );
    let path = &file.path;
    let name = Value::from(path.to_string_lossy().into_owned());
    let native = tensor(
        block_on(audioread_builtin(
            None,
            name.clone(),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::from("native")]),
        ))
        .expect("native extensible PCM"),
    );
    assert_eq!(
        native.integer_storage(),
        Some(&IntegerStorage::I32(vec![i32::MIN, 0, 0x7fff_ff00]))
    );
    let normalized = tensor(
        block_on(audioread_builtin(
            None,
            name,
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .expect("double PCM"),
    );
    assert_eq!(normalized.numeric_dtype(), NumericDType::F64);
    assert_eq!(
        normalized.materialize_f64(),
        vec![-1.0, 0.0, 0x7fff_ff00 as f64 / 2_147_483_648.0]
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn audioread_extensible_pcm16_in_32_preserves_left_aligned_samples() {
    let file = audio_file(
        "wav",
        &extensible_pcm32_wav(8_000, 16, &[i32::MIN, 0, 0x4000_0000]),
    );
    let path = &file.path;
    let name = Value::from(path.to_string_lossy().into_owned());
    let native = tensor(
        block_on(audioread_builtin(
            None,
            name.clone(),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::from("native")]),
        ))
        .expect("native extensible PCM"),
    );
    assert_eq!(
        native.integer_storage(),
        Some(&IntegerStorage::I32(vec![i32::MIN, 0, 0x4000_0000]))
    );
    let normalized = tensor(
        block_on(audioread_builtin(
            None,
            name,
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .expect("double PCM"),
    );
    assert_eq!(normalized.materialize_f64(), vec![-1.0, 0.0, 0.5]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn audioread_rejects_unsupported_encoding_even_for_empty_data() {
    let file = audio_file("wav", &wav_with_payload(8_000, 1, 8, 0x0006, &[]));
    let path = &file.path;
    let error = block_on(audioread_builtin(
        None,
        Value::from(path.to_string_lossy().into_owned()),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .expect_err("unsupported empty coding");
    assert_eq!(
        error.identifier(),
        Some("RunMat:audioread:UnsupportedFormat")
    );
    assert!(error.message().contains("unsupported WAVE sample encoding"));
}
