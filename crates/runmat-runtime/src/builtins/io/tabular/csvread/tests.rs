use super::*;
use runmat_time::unix_timestamp_ns;
use std::fs;
use std::sync::atomic::{AtomicUsize, Ordering};

use runmat_value::{CharArray, IntValue, IntegerStorage, Tensor as BuiltinTensor};

#[path = "tests/cooperation.rs"]
mod cooperation;

fn csvread_builtin(path: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let _provider_lock = runmat_filesystem::provider_override_lock();
    futures::executor::block_on(super::csvread_builtin(
        None,
        path,
        crate::builtins::common::test_support::builtin_arguments(rest),
    ))
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn csvread_descriptor_signatures_cover_core_forms() {
    let labels: Vec<&str> = CSVREAD_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"M = csvread(filename)"));
    assert!(labels.contains(&"M = csvread(filename, row, col)"));
    assert!(labels.contains(&"M = csvread(filename, row, col, range)"));
}

#[test]
fn csvread_path_requires_effective_scalar_text_geometry() {
    let paged = Value::CharArray(
        CharArray::from_unicode_scalars_with_shape("ab".chars().collect(), vec![1, 1, 2]).unwrap(),
    );
    assert!(resolve_path(&paged).is_err());
    let singleton = Value::CharArray(
        CharArray::from_unicode_scalars_with_shape("ab".chars().collect(), vec![1, 2, 1]).unwrap(),
    );
    assert_eq!(resolve_path(&singleton).unwrap(), PathBuf::from("ab"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn csvread_expands_home_directory_without_file_access() {
    let Some(home) = crate::builtins::common::fs::home_directory() else {
        return;
    };
    assert_eq!(
        normalize_path("~/runmat_csvread_home.csv").expect("home expansion"),
        home.join("runmat_csvread_home.csv")
    );
}

static UNIQUE_COUNTER: AtomicUsize = AtomicUsize::new(0);

fn unique_path(prefix: &str) -> PathBuf {
    let nanos = unix_timestamp_ns();
    let seq = UNIQUE_COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut path = std::env::temp_dir();
    path.push(format!(
        "runmat_csvread_{prefix}_{}_{}_{}",
        std::process::id(),
        nanos,
        seq
    ));
    path
}

fn write_temp_file(lines: &[&str]) -> PathBuf {
    let path = unique_path("input").with_extension("csv");
    let contents = lines.join("\n");
    fs::write(&path, contents).expect("write temp csv");
    path
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn csvread_basic_csv_roundtrip() {
    let path = write_temp_file(&["1,2,3", "4,5,6"]);
    let result = csvread_builtin(Value::from(path.to_string_lossy().to_string()), Vec::new())
        .expect("csvread");
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
fn csvread_with_offsets() {
    let path = write_temp_file(&["0,1,2", "3,4,5", "6,7,8"]);
    let args = vec![Value::Int(IntValue::I32(1)), Value::Int(IntValue::I32(1))];
    let result =
        csvread_builtin(Value::from(path.to_string_lossy().to_string()), args).expect("csv");
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
fn csvread_with_numeric_range() {
    let path = write_temp_file(&["1,2,3", "4,5,6", "7,8,9"]);
    let args = vec![
        Value::Int(IntValue::I32(0)),
        Value::Int(IntValue::I32(0)),
        Value::from(BuiltinTensor::new(vec![1.0, 1.0, 2.0, 2.0], vec![4, 1]).expect("tensor")),
    ];
    let result =
        csvread_builtin(Value::from(path.to_string_lossy().to_string()), args).expect("csv");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), &[2, 2]);
            assert_eq!(t.materialize_f64(), vec![5.0, 8.0, 6.0, 9.0]);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
    fs::remove_file(path).ok();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn csvread_with_string_range() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let path = write_temp_file(&["1,2,3", "4,5,6", "7,8,9"]);
    let args = vec![
        Value::Int(IntValue::I32(0)),
        Value::Int(IntValue::I32(0)),
        Value::from("B2:C3"),
    ];
    let result =
        csvread_builtin(Value::from(path.to_string_lossy().to_string()), args).expect("csv");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), &[2, 2]);
            assert_eq!(t.materialize_f64(), vec![5.0, 8.0, 6.0, 9.0]);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
    fs::remove_file(path).ok();
}

#[test]
fn csvread_numeric_range_reads_typed_integer_storage_exactly() {
    let range = BuiltinTensor::new_integer(IntegerStorage::U16(vec![1, 2, 3, 4]), vec![1, 4])
        .expect("range");

    let parsed = parse_range_numeric(&Value::Tensor(range)).expect("range");

    assert_eq!(parsed.start_row, 1);
    assert_eq!(parsed.start_col, 2);
    assert_eq!(parsed.end_row, Some(3));
    assert_eq!(parsed.end_col, Some(4));
}

#[test]
fn csvread_start_offsets_read_typed_integer_storage_exactly() {
    let row =
        BuiltinTensor::new_integer(IntegerStorage::U64(vec![7]), vec![1, 1]).expect("row offset");
    assert_eq!(value_to_start_index(&Value::Tensor(row), "row").unwrap(), 7);

    let negative = BuiltinTensor::new_integer(IntegerStorage::I16(vec![-1]), vec![1, 1])
        .expect("negative row offset");
    assert!(value_to_start_index(&Value::Tensor(negative), "row").is_err());

    assert_eq!(
        value_to_start_index(&Value::Int(IntValue::U64(u64::MAX)), "row").ok(),
        usize::try_from(u64::MAX).ok()
    );
    assert!(value_to_start_index(&Value::Num(1.0e300), "row").is_err());
}

#[test]
fn csvread_start_offsets_read_all_integer_storage_classes_with_poisoned_f64_mirrors() {
    let storages = [
        IntegerStorage::I8(vec![7]),
        IntegerStorage::I16(vec![7]),
        IntegerStorage::I32(vec![7]),
        IntegerStorage::I64(vec![7]),
        IntegerStorage::U8(vec![7]),
        IntegerStorage::U16(vec![7]),
        IntegerStorage::U32(vec![7]),
        IntegerStorage::U64(vec![7]),
    ];

    for storage in storages {
        let tensor = BuiltinTensor::new_integer(storage, vec![1, 1]).expect("index");
        assert_eq!(
            value_to_start_index(&Value::Tensor(tensor), "row").expect("index"),
            7
        );
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn csvread_empty_fields_become_zero() {
    let path = write_temp_file(&["1,,3", ",5,", "7,8,"]);
    let result =
        csvread_builtin(Value::from(path.to_string_lossy().to_string()), Vec::new()).expect("csv");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), &[3, 3]);
            assert_eq!(
                t.materialize_f64(),
                vec![1.0, 0.0, 7.0, 0.0, 5.0, 8.0, 3.0, 0.0, 0.0]
            );
        }
        other => panic!("expected tensor, got {other:?}"),
    }
    fs::remove_file(path).ok();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn csvread_errors_on_text() {
    let path = write_temp_file(&["1,2,3", "4,error,6"]);
    let err = csvread_builtin(Value::from(path.to_string_lossy().to_string()), Vec::new())
        .expect_err("should fail");
    let message = err.message().to_string();
    assert!(
        message.contains("nonnumeric token 'error'"),
        "unexpected error: {message}"
    );
    fs::remove_file(path).ok();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn csvread_accepts_char_array_filename() {
    let path = write_temp_file(&["1,2"]);
    let path_string = path.to_string_lossy().to_string();
    let data: Vec<char> = path_string.chars().collect();
    let cols = path_string.encode_utf16().count();
    let chars = CharArray::from_unicode_scalars(data, 1, cols).expect("char array");
    let result = csvread_builtin(Value::CharArray(chars), Vec::new()).expect("csv");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), &[1, 2]);
            assert_eq!(t.materialize_f64(), vec![1.0, 2.0]);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
    fs::remove_file(path).ok();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn csvread_with_header_and_row_labels_using_offsets() {
    let path = write_temp_file(&["Name,Jan,Feb", "alpha,1,2", "beta,3,4"]);
    let args = vec![Value::Int(IntValue::I32(1)), Value::Int(IntValue::I32(1))];
    let result =
        csvread_builtin(Value::from(path.to_string_lossy().to_string()), args).expect("csv");
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
fn csvread_skips_non_utf8_bytes_outside_requested_numeric_block() {
    let path = unique_path("latin1_offsets").with_extension("csv");
    fs::write(&path, b"label,\xFCtemp\n\xFFrow,21.5\n\xFErow,22.0\n").expect("write temp csv");
    let args = vec![Value::Int(IntValue::I32(1)), Value::Int(IntValue::I32(1))];
    let result =
        csvread_builtin(Value::from(path.to_string_lossy().to_string()), args).expect("csv");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), &[2, 1]);
            assert_eq!(t.materialize_f64(), vec![21.5, 22.0]);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
    fs::remove_file(path).ok();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn csvread_numeric_range_ignores_bytes_and_text_outside_rectangle() {
    let path = unique_path("latin1_numeric_range").with_extension("csv");
    fs::write(
            &path,
            b"header,\xFCbad,tail\n\xFFlabel,21.5,\xFEtail\n\xFFlabel,22.0,\xFEtail\nafter,\xFCbad,tail\n",
        )
        .expect("write temp csv");
    let range = BuiltinTensor::new(vec![1.0, 1.0, 2.0, 1.0], vec![4, 1]).expect("tensor");
    let args = vec![
        Value::Int(IntValue::I32(0)),
        Value::Int(IntValue::I32(0)),
        Value::from(range),
    ];
    let result =
        csvread_builtin(Value::from(path.to_string_lossy().to_string()), args).expect("csv");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), &[2, 1]);
            assert_eq!(t.materialize_f64(), vec![21.5, 22.0]);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
    fs::remove_file(path).ok();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn csvread_a1_range_ignores_bytes_and_text_outside_rectangle() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let path = unique_path("latin1_a1_range").with_extension("csv");
    fs::write(
            &path,
            b"header,\xFCbad,tail\n\xFFlabel,21.5,\xFEtail\n\xFFlabel,22.0,\xFEtail\nafter,\xFCbad,tail\n",
        )
        .expect("write temp csv");
    let args = vec![
        Value::Int(IntValue::I32(0)),
        Value::Int(IntValue::I32(0)),
        Value::from("B2:B3"),
    ];
    let result =
        csvread_builtin(Value::from(path.to_string_lossy().to_string()), args).expect("csv");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), &[2, 1]);
            assert_eq!(t.materialize_f64(), vec![21.5, 22.0]);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
    fs::remove_file(path).ok();
}

#[test]
fn csvread_public_dotdot_range_and_complex_tokens() {
    let path = write_temp_file(&["1+2i,3-4j,5i", "6,7,8"]);
    let args = vec![Value::Num(0.0), Value::Num(0.0), Value::from("A1..C2")];
    let Value::ComplexTensor(result) =
        csvread_builtin(Value::from(path.to_string_lossy().to_string()), args).unwrap()
    else {
        panic!("expected complex double result");
    };
    assert_eq!(result.shape(), &[2, 3]);
    assert_eq!(
        result.materialize_f64(),
        vec![
            (1.0, 2.0),
            (6.0, 0.0),
            (3.0, -4.0),
            (7.0, 0.0),
            (0.0, 5.0),
            (8.0, 0.0)
        ]
    );
    fs::remove_file(path).ok();
}

#[test]
fn csvread_four_vector_reads_all_integer_classes_authoritatively() {
    for storage in [
        IntegerStorage::I8(vec![0, 0, 1, 1]),
        IntegerStorage::I16(vec![0, 0, 1, 1]),
        IntegerStorage::I32(vec![0, 0, 1, 1]),
        IntegerStorage::I64(vec![0, 0, 1, 1]),
        IntegerStorage::U8(vec![0, 0, 1, 1]),
        IntegerStorage::U16(vec![0, 0, 1, 1]),
        IntegerStorage::U32(vec![0, 0, 1, 1]),
        IntegerStorage::U64(vec![0, 0, 1, 1]),
    ] {
        let tensor = BuiltinTensor::new_integer(storage, vec![1, 4]).unwrap();
        let range = parse_range_numeric(&Value::Tensor(tensor)).unwrap();
        assert_eq!((range.start_row, range.start_col), (0, 0));
        assert_eq!((range.end_row, range.end_col), (Some(1), Some(1)));
    }
}

#[test]
fn csvread_range_extensions_are_independently_gated() {
    let strict = crate::compatibility::push_runmat_extensions_enabled(false);
    assert_eq!(
        parse_range_string("A1:B2").unwrap_err().identifier(),
        CSVREAD_COLON_RANGE_EXTENSION.error_identifier
    );
    let two = BuiltinTensor::new(vec![0.0, 0.0], vec![1, 2]).unwrap();
    assert_eq!(
        parse_range_numeric(&Value::Tensor(two))
            .unwrap_err()
            .identifier(),
        CSVREAD_TWO_VECTOR_RANGE_EXTENSION.error_identifier
    );
    drop(strict);
}

#[test]
fn csvread_resident_control_gate_precedes_provider_and_file_access() {
    let strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let resident = Value::GpuTensor(
        runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 1],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .unwrap()
            .handle,
    );
    let error = csvread_builtin(Value::from("does-not-exist.csv"), vec![resident]).unwrap_err();
    assert_eq!(
        error.identifier(),
        CSVREAD_RESIDENT_CONTROL_EXTENSION.error_identifier
    );
    drop(strict);
    assert_eq!(CSVREAD_EXTENSIONS.len(), 3);
    assert_eq!(CSVREAD_INTEGER_CAPABILITIES[0].inputs.len(), 3);
}
