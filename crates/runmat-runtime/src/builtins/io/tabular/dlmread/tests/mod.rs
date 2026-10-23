use super::*;

#[test]
fn dlmread_path_requires_effective_scalar_text_geometry() {
    let paged = Value::CharArray(
        CharArray::from_unicode_scalars_with_shape("ab".chars().collect(), vec![1, 1, 2]).unwrap(),
    );
    assert!(resolve_path(&paged).is_err());
    let singleton = Value::CharArray(
        CharArray::from_unicode_scalars_with_shape("ab".chars().collect(), vec![1, 2, 1]).unwrap(),
    );
    assert_eq!(resolve_path(&singleton).unwrap(), PathBuf::from("ab"));
}
use crate::builtins::common::test_support;
use runmat_time::unix_timestamp_ns;
use std::fs;
use std::sync::atomic::{AtomicUsize, Ordering};

use runmat_value::{CharArray, IntegerStorage, Tensor as BuiltinTensor};

fn dlmread_builtin(path: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let _fs_lock = runmat_filesystem::provider_override_lock();
    futures::executor::block_on(super::dlmread_builtin(
        None,
        path,
        crate::builtins::common::test_support::builtin_arguments(rest),
    ))
}

static UNIQUE_COUNTER: AtomicUsize = AtomicUsize::new(0);

fn unique_path(prefix: &str) -> PathBuf {
    let nanos = unix_timestamp_ns();
    let seq = UNIQUE_COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut path = std::env::temp_dir();
    path.push(format!(
        "runmat_dlmread_{prefix}_{}_{}_{}",
        std::process::id(),
        nanos,
        seq
    ));
    path
}

#[test]
fn dlmread_descriptor_signatures_cover_core_forms() {
    let labels: Vec<&str> = DLMREAD_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(
        labels.contains(&"M = dlmread(filename)"),
        "missing base dlmread signature label"
    );
    assert!(
        labels.contains(&"M = dlmread(filename, delimiter)"),
        "missing delimiter dlmread signature label"
    );
    assert!(
        labels.contains(&"M = dlmread(filename, row, col)"),
        "missing offset dlmread signature label"
    );
    assert!(
        labels.contains(&"M = dlmread(filename, delimiter, range)"),
        "missing delimiter+range dlmread signature label"
    );
    assert!(
        labels.contains(&"M = dlmread(filename, delimiter, row, col)"),
        "missing delimiter+offset dlmread signature label"
    );
    assert!(
        labels.contains(&"M = dlmread(filename, delimiter, row, col, range)"),
        "missing delimiter+offset+range dlmread signature label"
    );
}

fn write_temp_file(lines: &[&str]) -> PathBuf {
    let path = unique_path("input").with_extension("txt");
    let contents = lines.join("\n");
    fs::write(&path, contents).expect("write temp file");
    path
}

fn write_temp_file_bytes(bytes: &[u8]) -> PathBuf {
    let path = unique_path("input_bytes").with_extension("txt");
    fs::write(&path, bytes).expect("write temp file bytes");
    path
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn dlmread_default_delimiter() {
    let path = write_temp_file(&["1,2,3", "4,5,6"]);
    let result = dlmread_builtin(Value::from(path.to_string_lossy().to_string()), Vec::new())
        .expect("dlmread");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), &[2, 3]);
            assert_eq!(t.materialize_f64(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
    fs::remove_file(path).ok();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn dlmread_semicolon_roundtrip() {
    let path = write_temp_file(&["1;2;3", "4;5;6"]);
    let args = vec![Value::from(";")];
    let result =
        dlmread_builtin(Value::from(path.to_string_lossy().to_string()), args).expect("dlmread");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), &[2, 3]);
            assert_eq!(t.materialize_f64(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
    fs::remove_file(path).ok();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn dlmread_ascii_code_delimiter() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let path = write_temp_file(&["5|6|7", "8|9|10"]);
    let args = vec![Value::Int(IntValue::I32('|' as i32))];
    let result =
        dlmread_builtin(Value::from(path.to_string_lossy().to_string()), args).expect("dlmread");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), &[2, 3]);
            assert_eq!(t.materialize_f64(), vec![5.0, 8.0, 6.0, 9.0, 7.0, 10.0]);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
    fs::remove_file(path).ok();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn dlmread_char_array_filename() {
    let path = write_temp_file(&["1,2", "3,4"]);
    let path_string = path.to_string_lossy().to_string();
    let chars: Vec<char> = path_string.chars().collect();
    let char_array = CharArray::from_unicode_scalars(chars, 1, path_string.encode_utf16().count())
        .expect("char array");
    let result = dlmread_builtin(Value::CharArray(char_array), Vec::new()).expect("dlmread");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), &[2, 2]);
            assert_eq!(t.materialize_f64(), vec![1.0, 3.0, 2.0, 4.0]);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
    fs::remove_file(path).ok();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn dlmread_handles_utf8_bom() {
    let bytes = b"\xEF\xBB\xBF1,2\n3,4\n";
    let path = write_temp_file_bytes(bytes);
    let result = dlmread_builtin(Value::from(path.to_string_lossy().to_string()), Vec::new())
        .expect("dlmread");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), &[2, 2]);
            assert_eq!(t.materialize_f64(), vec![1.0, 3.0, 2.0, 4.0]);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
    fs::remove_file(path).ok();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn dlmread_empty_file_returns_empty_tensor() {
    let path = write_temp_file(&[]);
    let result = dlmread_builtin(Value::from(path.to_string_lossy().to_string()), Vec::new())
        .expect("dlmread");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), &[0, 0]);
            assert!(t.materialize_f64().is_empty());
        }
        other => panic!("expected tensor, got {other:?}"),
    }
    fs::remove_file(path).ok();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn dlmread_with_offsets() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let path = write_temp_file(&["0,1,2", "3,4,5", "6,7,8"]);
    let args = vec![Value::Int(IntValue::I32(1)), Value::Int(IntValue::I32(1))];
    let result =
        dlmread_builtin(Value::from(path.to_string_lossy().to_string()), args).expect("dlmread");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), &[2, 2]);
            assert_eq!(t.materialize_f64(), vec![4.0, 7.0, 5.0, 8.0]);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
    fs::remove_file(path).ok();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn dlmread_empty_delimiter_placeholders_are_documented_strict_forms() {
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);

    let comma_path = write_temp_file(&["0,1,2", "3,4,5", "6,7,8"]);
    let comma = dlmread_builtin(
        Value::from(comma_path.to_string_lossy().to_string()),
        vec![
            Value::from(""),
            Value::Int(IntValue::U8(1)),
            Value::Int(IntValue::U8(1)),
        ],
    )
    .expect("empty string delimiter placeholder");
    let Value::Tensor(comma) = comma else {
        panic!("expected comma tensor")
    };
    assert_eq!(comma.shape(), &[2, 2]);
    assert_eq!(comma.materialize_f64(), vec![4.0, 7.0, 5.0, 8.0]);

    let whitespace_path = write_temp_file(&["0 1 2", "3 4 5", "6 7 8"]);
    let empty_char = CharArray::from_code_units(Vec::new(), 1, 0).expect("empty character vector");
    let whitespace = dlmread_builtin(
        Value::from(whitespace_path.to_string_lossy().to_string()),
        vec![
            Value::CharArray(empty_char),
            Value::Int(IntValue::U8(1)),
            Value::Int(IntValue::U8(1)),
        ],
    )
    .expect("empty char delimiter placeholder");
    let Value::Tensor(whitespace) = whitespace else {
        panic!("expected whitespace tensor")
    };
    assert_eq!(whitespace.shape(), &[2, 2]);
    assert_eq!(whitespace.materialize_f64(), vec![4.0, 7.0, 5.0, 8.0]);

    fs::remove_file(comma_path).ok();
    fs::remove_file(whitespace_path).ok();
}

#[test]
fn dlmread_strict_composed_range_rejects_before_file_access() {
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let path = unique_path("strict_composed_absent").with_extension("csv");
    assert!(!path.exists());
    let error = dlmread_builtin(
        Value::from(path.to_string_lossy().to_string()),
        vec![Value::Int(IntValue::U8(0)), Value::Int(IntValue::U8(0))],
    )
    .expect_err("omitted-delimiter offsets must reject before file access");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:DlmreadComposedRangeExtension")
    );
    assert!(!path.exists());
}

#[test]
fn dlmread_strict_resident_argument_rejects_before_provider_or_file_access() {
    test_support::with_test_provider(|provider| {
        let handle = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &[44.0],
                shape: &[1, 1],
            })
            .expect("upload resident delimiter");
        provider.reset_telemetry();
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let path = unique_path("strict_resident_absent").with_extension("csv");
        assert!(!path.exists());

        let error = dlmread_builtin(
            Value::from(path.to_string_lossy().to_string()),
            vec![Value::GpuTensor(handle.clone())],
        )
        .expect_err("resident argument must reject before provider access");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:DlmreadResidentArgumentExtension")
        );
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        assert!(!path.exists());
        handle.release().expect("free resident delimiter");
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn resident_numeric_filename_rejects_without_provider_readback() {
    test_support::with_test_provider(|provider| {
        let handle = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &[1.0],
                shape: &[1, 1],
            })
            .expect("resident numeric filename");
        provider.reset_telemetry();
        let _extension = crate::compatibility::push_runmat_extensions_enabled(true);
        let error = dlmread_builtin(Value::GpuTensor(handle.clone()), Vec::new())
            .expect_err("numeric storage cannot contain scalar text");
        assert!(error.message().contains("filename"));
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        handle.release().expect("release resident filename");
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn nested_resident_invalid_control_rejects_without_recursive_gather() {
    test_support::with_test_provider(|provider| {
        let handle = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &[44.0],
                shape: &[1, 1],
            })
            .expect("resident nested value");
        provider.reset_telemetry();
        let cell = runmat_value::CellArray::new(vec![Value::GpuTensor(handle.clone())], 1, 1)
            .expect("cell");
        let error = dlmread_builtin(Value::from("unused.csv"), vec![Value::Cell(cell)])
            .expect_err("cells are not delimiter controls");
        assert!(error.message().contains("controls"));
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        handle.release().expect("release nested value");
    });
}

#[test]
fn dlmread_integer_tensor_parsers_preserve_exact_bounds() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let row = Tensor::new_integer(IntegerStorage::U16(vec![7]), vec![1, 1]).expect("row");
    assert_eq!(value_to_start_index(&Value::Tensor(row), "row").unwrap(), 7);

    let negative =
        Tensor::new_integer(IntegerStorage::I16(vec![-1]), vec![1, 1]).expect("negative");
    assert!(value_to_start_index(&Value::Tensor(negative), "row").is_err());

    let comma = Tensor::new_integer(IntegerStorage::U16(vec![44]), vec![1, 1]).expect("delimiter");
    assert!(matches!(
        parse_delimiter(&Value::Tensor(comma)),
        Ok(DelimiterSpec::Char(','))
    ));
    let invalid_delimiter =
        Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1]).expect("delimiter");
    assert!(parse_delimiter(&Value::Tensor(invalid_delimiter)).is_err());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn dlmread_offsets_skip_nonnumeric_header_row_and_column() {
    let path = write_temp_file(&["Label,Jan,Feb", "alpha,1,2", "beta,3,4"]);
    let args = vec![
        Value::from(","),
        Value::Int(IntValue::I32(1)),
        Value::Int(IntValue::I32(1)),
    ];
    let result =
        dlmread_builtin(Value::from(path.to_string_lossy().to_string()), args).expect("dlmread");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), &[2, 2]);
            assert_eq!(t.materialize_f64(), vec![1.0, 3.0, 2.0, 4.0]);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
    fs::remove_file(path).ok();
}

mod ranges;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn dlmread_space_delimiter() {
    let path = write_temp_file(&["1  3", " 4 5 ", "6 7  "]);
    let args = vec![Value::from(" ")];
    let result =
        dlmread_builtin(Value::from(path.to_string_lossy().to_string()), args).expect("dlmread");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), &[3, 4]);
            assert_eq!(
                t.materialize_f64(),
                vec![
                    1.0, 0.0, 6.0, // column 0
                    0.0, 4.0, 7.0, // column 1
                    3.0, 5.0, 0.0, // column 2
                    0.0, 0.0, 0.0, // column 3 (trailing blanks)
                ]
            );
        }
        other => panic!("expected tensor, got {other:?}"),
    }
    fs::remove_file(path).ok();
}
