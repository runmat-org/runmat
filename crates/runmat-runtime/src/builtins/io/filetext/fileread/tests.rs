use super::*;
use crate::builtins::common::test_support;
#[cfg(not(target_arch = "wasm32"))]
use runmat_time::unix_timestamp_ms;
use std::io::Write;

fn unwrap_error_message(err: RuntimeError) -> String {
    err.message().to_string()
}

fn run_fileread(path: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    futures::executor::block_on(fileread_builtin(
        path,
        crate::builtins::common::test_support::builtin_arguments(rest),
    ))
}

fn unowned_resident_value() -> Value {
    Value::GpuTensor(
        runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 1],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .unwrap()
            .handle,
    )
}

#[test]
fn fileread_descriptor_is_integer_inapplicable() {
    assert_eq!(
        FILEREAD_INTEGER_AUDIT.kind,
        BuiltinIntegerAuditKind::NotApplicable
    );
    assert!(FILEREAD_INTEGER_AUDIT.canonical_builtin.is_none());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fileread_descriptor_signatures_cover_core_forms() {
    #[cfg(target_arch = "wasm32")]
    let _filesystem = test_support::filesystem::tempdir().expect("browser fixture");
    let labels: Vec<&str> = FILEREAD_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"text = fileread(filename)"));
    assert!(labels.contains(&"text = fileread(filename, encoding)"));
    assert!(labels.contains(&"text = fileread(filename, \"Encoding\", encoding)"));
}

#[test]
fn fileread_rejects_numeric_and_resident_paths_before_provider_access() {
    for invalid in [Value::Num(1.0), unowned_resident_value()] {
        let error = run_fileread(invalid, Vec::new()).expect_err("invalid filename");
        assert_eq!(error.identifier(), FILEREAD_ERROR_INVALID_INPUT.identifier);
        assert!(!error.message().to_ascii_lowercase().contains("provider"));
    }
}

#[test]
fn fileread_rejects_numeric_and_resident_encoding_inputs_before_provider_access() {
    let filename = Value::String("unused-fileread-path".to_string().into());
    for invalid in [Value::Num(1.0), unowned_resident_value()] {
        let error =
            run_fileread(filename.clone(), vec![invalid]).expect_err("invalid encoding input");
        assert_eq!(error.identifier(), FILEREAD_ERROR_INVALID_OPTION.identifier);
        assert!(!error.message().to_ascii_lowercase().contains("provider"));
    }

    let error = run_fileread(
        filename,
        vec![
            Value::String("Encoding".to_string().into()),
            unowned_resident_value(),
        ],
    )
    .expect_err("resident Encoding value");
    assert_eq!(error.identifier(), FILEREAD_ERROR_INVALID_OPTION.identifier);
    assert!(!error.message().to_ascii_lowercase().contains("provider"));
}

#[cfg(not(target_arch = "wasm32"))]
fn unique_path(prefix: &str) -> PathBuf {
    let millis = unix_timestamp_ms();
    let mut path = std::env::temp_dir();
    path.push(format!("runmat_{prefix}_{}_{}", std::process::id(), millis));
    path
}

#[cfg(target_arch = "wasm32")]
fn unique_path(prefix: &str) -> PathBuf {
    test_support::filesystem::unique_path(prefix)
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fileread_reads_text_file() {
    #[cfg(target_arch = "wasm32")]
    let _filesystem = test_support::filesystem::tempdir().expect("browser fixture");
    let path = unique_path("fileread_text");
    let contents = "RunMat fileread\nLine two\n";
    test_support::fs::write(&path, contents).expect("write sample file");

    let value = Value::from(path.to_string_lossy().to_string());
    let result = run_fileread(value, Vec::new()).expect("fileread result");

    match result {
        Value::CharArray(ca) => {
            let text: String =
                runmat_value::decode_utf16_code_units(ca.data()).expect("valid UTF-16 fixture");
            assert_eq!(ca.rows, 1);
            assert_eq!(ca.cols, contents.chars().count());
            assert_eq!(text, contents);
        }
        other => panic!("expected char array, got {other:?}"),
    }

    let _ = test_support::fs::remove_file(&path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fileread_accepts_char_array_input() {
    #[cfg(target_arch = "wasm32")]
    let _filesystem = test_support::filesystem::tempdir().expect("browser fixture");
    let path = unique_path("fileread_char_input");
    test_support::fs::write(&path, "abc").expect("write sample file");

    let path_str = path.to_string_lossy();
    let chars: Vec<char> = path_str.chars().collect();
    let len = path_str.encode_utf16().count();
    let char_array =
        CharArray::from_unicode_scalars(chars, 1, len).expect("char array from path string");
    let value = Value::CharArray(char_array);

    let result = run_fileread(value, Vec::new()).expect("fileread");
    match result {
        Value::CharArray(ca) => {
            let text: String =
                runmat_value::decode_utf16_code_units(ca.data()).expect("valid UTF-16 fixture");
            assert_eq!(text, "abc");
        }
        other => panic!("expected char array, got {other:?}"),
    }

    let _ = test_support::fs::remove_file(&path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fileread_accepts_string_array_scalar() {
    #[cfg(target_arch = "wasm32")]
    let _filesystem = test_support::filesystem::tempdir().expect("browser fixture");
    let path = unique_path("fileread_string_scalar");
    test_support::fs::write(&path, "xyz").expect("write sample file");

    let path_str = path.to_string_lossy().to_string();
    let string_array = runmat_value::StringArray::new(vec![path_str.clone()], vec![1, 1])
        .expect("string array scalar");
    let value = Value::StringArray(string_array);

    let result = run_fileread(value, Vec::new()).expect("fileread");
    match result {
        Value::CharArray(ca) => {
            let text: String =
                runmat_value::decode_utf16_code_units(ca.data()).expect("valid UTF-16 fixture");
            assert_eq!(text, "xyz");
        }
        other => panic!("expected char array, got {other:?}"),
    }

    let _ = test_support::fs::remove_file(&path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fileread_returns_empty_for_empty_file() {
    #[cfg(target_arch = "wasm32")]
    let _filesystem = test_support::filesystem::tempdir().expect("browser fixture");
    let path = unique_path("fileread_empty");
    fs::File::create(&path).expect("create empty file");

    let result = run_fileread(Value::from(path.to_string_lossy().to_string()), Vec::new())
        .expect("fileread");
    match result {
        Value::CharArray(ca) => {
            assert_eq!(ca.rows, 1);
            assert_eq!(ca.cols, 0);
            assert!(ca.data().is_empty());
        }
        other => panic!("expected char array, got {other:?}"),
    }

    let _ = test_support::fs::remove_file(&path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fileread_errors_when_file_missing() {
    #[cfg(target_arch = "wasm32")]
    let _filesystem = test_support::filesystem::tempdir().expect("browser fixture");
    let path = unique_path("fileread_missing");
    let err = unwrap_error_message(
        run_fileread(Value::from(path.to_string_lossy().to_string()), Vec::new()).unwrap_err(),
    );
    assert!(
        err.contains("unable to read"),
        "unexpected error message: {err}"
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fileread_preserves_non_utf8_bytes() {
    #[cfg(target_arch = "wasm32")]
    let _filesystem = test_support::filesystem::tempdir().expect("browser fixture");
    let path = unique_path("fileread_raw_bytes");
    let mut file = fs::File::create(&path).expect("create file");
    file.write_all(&[0xFF, 0x61, 0x00]).expect("write bytes");
    drop(file);

    let result = run_fileread(Value::from(path.to_string_lossy().to_string()), Vec::new())
        .expect("fileread");
    match result {
        Value::CharArray(ca) => {
            let collected: Vec<u32> = ca.data().iter().map(|ch| u32::from(*ch)).collect();
            assert_eq!(collected, vec![0x00FF, 0x0061, 0x0000]);
        }
        other => panic!("expected char array, got {other:?}"),
    }

    let _ = test_support::fs::remove_file(&path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fileread_supports_encoding_keyword() {
    #[cfg(target_arch = "wasm32")]
    let _filesystem = test_support::filesystem::tempdir().expect("browser fixture");
    let path = unique_path("fileread_utf8_keyword");
    let contents = "UTF-8 ✓";
    test_support::fs::write(&path, contents).expect("write sample file");

    let result = run_fileread(
        Value::from(path.to_string_lossy().to_string()),
        vec![Value::from("Encoding"), Value::from("utf-8")],
    )
    .expect("fileread");
    match result {
        Value::CharArray(ca) => {
            let text: String =
                runmat_value::decode_utf16_code_units(ca.data()).expect("valid UTF-16 fixture");
            assert_eq!(text, contents);
        }
        other => panic!("expected char array, got {other:?}"),
    }

    let _ = test_support::fs::remove_file(&path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fileread_supports_single_encoding_argument() {
    #[cfg(target_arch = "wasm32")]
    let _filesystem = test_support::filesystem::tempdir().expect("browser fixture");
    let path = unique_path("fileread_latin1");
    let bytes = [0xC0u8, 0x20, 0x41];
    test_support::fs::write(&path, bytes).expect("write latin1 data");

    let result = run_fileread(
        Value::from(path.to_string_lossy().to_string()),
        vec![Value::from("latin1")],
    )
    .expect("fileread");
    match result {
        Value::CharArray(ca) => {
            let codes: Vec<u32> = ca.data().iter().map(|ch| u32::from(*ch)).collect();
            assert_eq!(codes, vec![0x00C0, 0x0020, 0x0041]);
        }
        other => panic!("expected char array, got {other:?}"),
    }

    let _ = test_support::fs::remove_file(&path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fileread_raw_encoding_returns_bytes() {
    #[cfg(target_arch = "wasm32")]
    let _filesystem = test_support::filesystem::tempdir().expect("browser fixture");
    let path = unique_path("fileread_raw_encoding");
    let bytes = [0x01u8, 0xFF, 0x7F];
    test_support::fs::write(&path, bytes).expect("write raw bytes");

    let result = run_fileread(
        Value::from(path.to_string_lossy().to_string()),
        vec![Value::from("raw")],
    )
    .expect("fileread");
    match result {
        Value::CharArray(ca) => {
            let codes: Vec<u8> = ca
                .data()
                .iter()
                .map(|ch| u8::try_from(ch.raw()).unwrap())
                .collect();
            assert_eq!(codes, bytes);
        }
        other => panic!("expected char array, got {other:?}"),
    }

    let _ = test_support::fs::remove_file(&path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fileread_ascii_encoding_errors_on_invalid_bytes() {
    #[cfg(target_arch = "wasm32")]
    let _filesystem = test_support::filesystem::tempdir().expect("browser fixture");
    let path = unique_path("fileread_ascii_error");
    test_support::fs::write(&path, [0x41, 0x80]).expect("write bytes");

    let err = unwrap_error_message(
        run_fileread(
            Value::from(path.to_string_lossy().to_string()),
            vec![Value::from("Encoding"), Value::from("ascii")],
        )
        .unwrap_err(),
    );
    assert!(
        err.contains("not valid ASCII"),
        "unexpected error message: {err}"
    );

    let _ = test_support::fs::remove_file(&path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fileread_encoding_keyword_missing_value_errors() {
    #[cfg(target_arch = "wasm32")]
    let _filesystem = test_support::filesystem::tempdir().expect("browser fixture");
    let path = unique_path("fileread_encoding_missing");
    test_support::fs::write(&path, "abc").expect("write file");

    let err = unwrap_error_message(
        run_fileread(
            Value::from(path.to_string_lossy().to_string()),
            vec![Value::from("Encoding")],
        )
        .unwrap_err(),
    );
    assert!(
        err.contains("missing encoding value"),
        "unexpected error message: {err}"
    );

    let _ = test_support::fs::remove_file(&path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fileread_too_many_arguments_errors() {
    #[cfg(target_arch = "wasm32")]
    let _filesystem = test_support::filesystem::tempdir().expect("browser fixture");
    let path = unique_path("fileread_too_many");
    test_support::fs::write(&path, "abc").expect("write file");

    let err = unwrap_error_message(
        run_fileread(
            Value::from(path.to_string_lossy().to_string()),
            vec![
                Value::from("Encoding"),
                Value::from("utf-8"),
                Value::from("extra"),
            ],
        )
        .unwrap_err(),
    );
    assert!(
        err.contains("too many input arguments"),
        "unexpected error message: {err}"
    );

    let _ = test_support::fs::remove_file(&path);
}
