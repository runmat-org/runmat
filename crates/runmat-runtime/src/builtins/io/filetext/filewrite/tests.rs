use super::*;
use crate::builtins::common::test_support;
use crate::RuntimeError;
#[cfg(not(target_arch = "wasm32"))]
use runmat_time::unix_timestamp_ms;
use runmat_value::{CharArray, IntegerStorage, LogicalArray, StringArray, Tensor};
use std::io::Read;
use std::path::PathBuf;

fn unwrap_error_message(err: RuntimeError) -> String {
    err.message().to_string()
}

fn run_filewrite(path: Value, data: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    futures::executor::block_on(filewrite_builtin(
        path,
        data,
        crate::builtins::common::test_support::builtin_arguments(rest),
    ))
}

mod controls;

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
fn filewrite_writes_text_content() {
    #[cfg(target_arch = "wasm32")]
    let _filesystem = test_support::filesystem::tempdir().expect("browser fixture");
    let path = unique_path("filewrite_text");
    let contents = "RunMat filewrite\nLine two\n";

    let result = run_filewrite(
        Value::from(path.to_string_lossy().to_string()),
        Value::from(contents),
        Vec::new(),
    )
    .expect("filewrite");

    match result {
        Value::Num(n) => assert_eq!(n as usize, contents.len()),
        other => panic!("expected numeric byte count, got {other:?}"),
    }

    let written = test_support::fs::read_to_string(&path).expect("read filewrite output");
    assert_eq!(written, contents);

    let _ = test_support::fs::remove_file(&path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn filewrite_appends_when_requested() {
    #[cfg(target_arch = "wasm32")]
    let _filesystem = test_support::filesystem::tempdir().expect("browser fixture");
    let path = unique_path("filewrite_append");
    test_support::fs::write(&path, "first\n").expect("write baseline");

    run_filewrite(
        Value::from(path.to_string_lossy().to_string()),
        Value::from("second\n"),
        vec![Value::from("WriteMode"), Value::from("append")],
    )
    .expect("filewrite append");

    let written = test_support::fs::read_to_string(&path).expect("read appended file");
    assert_eq!(written, "first\nsecond\n");

    let _ = test_support::fs::remove_file(&path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn filewrite_errors_on_invalid_ascii() {
    #[cfg(target_arch = "wasm32")]
    let _filesystem = test_support::filesystem::tempdir().expect("browser fixture");
    let path = unique_path("filewrite_ascii_error");
    let err = unwrap_error_message(
        run_filewrite(
            Value::from(path.to_string_lossy().to_string()),
            Value::from("café"),
            vec![Value::from("Encoding"), Value::from("ascii")],
        )
        .unwrap_err(),
    );
    assert!(
        err.contains("cannot be encoded as ASCII"),
        "unexpected error message: {err}"
    );
    assert!(!path.exists());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn filewrite_writes_raw_bytes_from_tensor() {
    #[cfg(target_arch = "wasm32")]
    let _filesystem = test_support::filesystem::tempdir().expect("browser fixture");
    let path = unique_path("filewrite_raw_bytes");
    let tensor = Tensor::new(vec![0.0, 127.0, 255.0], vec![3, 1]).expect("tensor");
    run_filewrite(
        Value::from(path.to_string_lossy().to_string()),
        Value::Tensor(tensor),
        vec![Value::from("Encoding"), Value::from("raw")],
    )
    .expect("filewrite raw");

    let mut bytes = Vec::new();
    runmat_filesystem::File::open(&path)
        .expect("open raw file")
        .read_to_end(&mut bytes)
        .expect("read raw file");
    assert_eq!(bytes, vec![0u8, 127u8, 255u8]);

    let _ = test_support::fs::remove_file(&path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn filewrite_writes_raw_bytes_from_typed_integer_storage() {
    #[cfg(target_arch = "wasm32")]
    let _filesystem = test_support::filesystem::tempdir().expect("browser fixture");
    let path = unique_path("filewrite_typed_raw_bytes");
    let tensor = Tensor::new_integer(IntegerStorage::U8(vec![0, 127, 255]), vec![3, 1])
        .expect("typed byte tensor");

    run_filewrite(
        Value::from(path.to_string_lossy().to_string()),
        Value::Tensor(tensor),
        vec![Value::from("Encoding"), Value::from("raw")],
    )
    .expect("filewrite typed raw");

    let bytes = test_support::fs::read(&path).expect("read typed raw file");
    assert_eq!(bytes, vec![0u8, 127u8, 255u8]);

    let _ = test_support::fs::remove_file(&path);
}

#[test]
fn tensor_to_bytes_reads_each_integer_storage_class() {
    let cases = [
        (IntegerStorage::I8(vec![1]), 1),
        (IntegerStorage::I16(vec![2]), 2),
        (IntegerStorage::I32(vec![3]), 3),
        (IntegerStorage::I64(vec![4]), 4),
        (IntegerStorage::U8(vec![5]), 5),
        (IntegerStorage::U16(vec![6]), 6),
        (IntegerStorage::U32(vec![7]), 7),
        (IntegerStorage::U64(vec![8]), 8),
    ];

    for (storage, expected) in cases {
        let tensor = Tensor::new_integer(storage, vec![1, 1]).expect("typed tensor");
        assert_eq!(
            tensor_to_bytes(&tensor).expect("byte payload"),
            vec![expected]
        );
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn filewrite_typed_integer_tensor_range_errors_keep_exact_wide_values() {
    #[cfg(target_arch = "wasm32")]
    let _filesystem = test_support::filesystem::tempdir().expect("browser fixture");
    let path = unique_path("filewrite_typed_integer_range");
    let tensor = Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1])
        .expect("typed integer tensor");

    let err = unwrap_error_message(
        run_filewrite(
            Value::from(path.to_string_lossy().to_string()),
            Value::Tensor(tensor),
            vec![Value::from("Encoding"), Value::from("raw")],
        )
        .expect_err("wide uint64 must not fit in a byte"),
    );

    assert!(err.contains(&u64::MAX.to_string()), "{err}");
    assert!(!err.contains("18446744073709552000"), "{err}");
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn filewrite_numeric_scalar_writes_byte() {
    #[cfg(target_arch = "wasm32")]
    let _filesystem = test_support::filesystem::tempdir().expect("browser fixture");
    let path = unique_path("filewrite_numeric_scalar");
    run_filewrite(
        Value::from(path.to_string_lossy().to_string()),
        Value::Num(65.0),
        Vec::new(),
    )
    .expect("filewrite numeric scalar");

    let bytes = test_support::fs::read(&path).expect("read numeric scalar file");
    assert_eq!(bytes, vec![65u8]);

    let _ = test_support::fs::remove_file(&path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn filewrite_bool_scalar_writes_byte() {
    #[cfg(target_arch = "wasm32")]
    let _filesystem = test_support::filesystem::tempdir().expect("browser fixture");
    let path = unique_path("filewrite_bool_scalar");
    run_filewrite(
        Value::from(path.to_string_lossy().to_string()),
        Value::Bool(true),
        Vec::new(),
    )
    .expect("filewrite bool scalar");

    let bytes = test_support::fs::read(&path).expect("read bool scalar file");
    assert_eq!(bytes, vec![1u8]);

    let _ = test_support::fs::remove_file(&path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn filewrite_writes_logical_array_bytes() {
    #[cfg(target_arch = "wasm32")]
    let _filesystem = test_support::filesystem::tempdir().expect("browser fixture");
    let path = unique_path("filewrite_logical_array");
    let logical = LogicalArray::new(vec![0, 1, 2], vec![3]).expect("logical array");
    run_filewrite(
        Value::from(path.to_string_lossy().to_string()),
        Value::LogicalArray(logical),
        Vec::new(),
    )
    .expect("filewrite logical array");

    let bytes = test_support::fs::read(&path).expect("read logical array file");
    assert_eq!(bytes, vec![0u8, 1u8, 1u8]);

    let _ = test_support::fs::remove_file(&path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn filewrite_errors_on_numeric_out_of_range() {
    #[cfg(target_arch = "wasm32")]
    let _filesystem = test_support::filesystem::tempdir().expect("browser fixture");
    let path = unique_path("filewrite_out_of_range");
    let err = unwrap_error_message(
        run_filewrite(
            Value::from(path.to_string_lossy().to_string()),
            Value::Num(300.0),
            Vec::new(),
        )
        .unwrap_err(),
    );
    assert!(
        err.contains("range 0..255"),
        "unexpected error message: {err}"
    );
    assert!(!path.exists());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn filewrite_errors_on_non_integer_numeric() {
    #[cfg(target_arch = "wasm32")]
    let _filesystem = test_support::filesystem::tempdir().expect("browser fixture");
    let path = unique_path("filewrite_non_integer");
    let err = unwrap_error_message(
        run_filewrite(
            Value::from(path.to_string_lossy().to_string()),
            Value::Num(std::f64::consts::PI),
            Vec::new(),
        )
        .unwrap_err(),
    );
    assert!(
        err.contains("not an integer"),
        "unexpected error message: {err}"
    );
    assert!(!path.exists());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn filewrite_rejects_ascii_numeric_bytes_above_range() {
    #[cfg(target_arch = "wasm32")]
    let _filesystem = test_support::filesystem::tempdir().expect("browser fixture");
    let path = unique_path("filewrite_ascii_bytes");
    let tensor = Tensor::new(vec![255.0], vec![1, 1]).expect("tensor");
    let err = unwrap_error_message(
        run_filewrite(
            Value::from(path.to_string_lossy().to_string()),
            Value::Tensor(tensor),
            vec![Value::from("Encoding"), Value::from("ascii")],
        )
        .unwrap_err(),
    );
    assert!(
        err.contains("cannot be encoded as ASCII"),
        "unexpected error message: {err}"
    );
    assert!(!path.exists());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn filewrite_positional_encoding_argument() {
    #[cfg(target_arch = "wasm32")]
    let _filesystem = test_support::filesystem::tempdir().expect("browser fixture");
    let path = unique_path("filewrite_positional_encoding");
    run_filewrite(
        Value::from(path.to_string_lossy().to_string()),
        Value::from("Espa\u{00F1}a"),
        vec![Value::from("latin1")],
    )
    .expect("filewrite positional encoding");

    let bytes = test_support::fs::read(&path).expect("read latin1 file");
    assert_eq!(bytes, b"Espa\xF1a");

    let _ = test_support::fs::remove_file(&path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn filewrite_utf8_encoding_allows_arbitrary_bytes() {
    #[cfg(target_arch = "wasm32")]
    let _filesystem = test_support::filesystem::tempdir().expect("browser fixture");
    let path = unique_path("filewrite_utf8_numeric");
    let tensor = Tensor::new(vec![0.0, 255.0], vec![2, 1]).expect("tensor");
    run_filewrite(
        Value::from(path.to_string_lossy().to_string()),
        Value::Tensor(tensor),
        vec![Value::from("Encoding"), Value::from("utf-8")],
    )
    .expect("filewrite utf8 numeric");

    let bytes = test_support::fs::read(&path).expect("read utf8 numeric file");
    assert_eq!(bytes, vec![0u8, 255u8]);

    let _ = test_support::fs::remove_file(&path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn filewrite_rejects_unknown_option() {
    #[cfg(target_arch = "wasm32")]
    let _filesystem = test_support::filesystem::tempdir().expect("browser fixture");
    let path = unique_path("filewrite_unknown_option");
    let err = unwrap_error_message(
        run_filewrite(
            Value::from(path.to_string_lossy().to_string()),
            Value::from("data"),
            vec![Value::from("Mode"), Value::from("append")],
        )
        .unwrap_err(),
    );
    assert!(
        err.contains("unrecognised option"),
        "unexpected error message: {err}"
    );
    assert!(!path.exists());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn filewrite_rejects_duplicate_encoding() {
    #[cfg(target_arch = "wasm32")]
    let _filesystem = test_support::filesystem::tempdir().expect("browser fixture");
    let path = unique_path("filewrite_duplicate_encoding");
    let err = unwrap_error_message(
        run_filewrite(
            Value::from(path.to_string_lossy().to_string()),
            Value::from("data"),
            vec![
                Value::from("utf-8"),
                Value::from("Encoding"),
                Value::from("ascii"),
            ],
        )
        .unwrap_err(),
    );
    assert!(
        err.contains("duplicate 'Encoding'"),
        "unexpected error message: {err}"
    );
    assert!(!path.exists());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn filewrite_rejects_duplicate_writemode() {
    #[cfg(target_arch = "wasm32")]
    let _filesystem = test_support::filesystem::tempdir().expect("browser fixture");
    let path = unique_path("filewrite_duplicate_writemode");
    let err = unwrap_error_message(
        run_filewrite(
            Value::from(path.to_string_lossy().to_string()),
            Value::from("data"),
            vec![
                Value::from("WriteMode"),
                Value::from("append"),
                Value::from("WriteMode"),
                Value::from("overwrite"),
            ],
        )
        .unwrap_err(),
    );
    assert!(
        err.contains("duplicate 'WriteMode'"),
        "unexpected error message: {err}"
    );
    assert!(!path.exists());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn filewrite_rejects_invalid_writemode_value() {
    #[cfg(target_arch = "wasm32")]
    let _filesystem = test_support::filesystem::tempdir().expect("browser fixture");
    let path = unique_path("filewrite_invalid_writemode");
    let err = unwrap_error_message(
        run_filewrite(
            Value::from(path.to_string_lossy().to_string()),
            Value::from("data"),
            vec![Value::from("WriteMode"), Value::from("invalid")],
        )
        .unwrap_err(),
    );
    assert!(
        err.contains("unsupported write mode"),
        "unexpected error message: {err}"
    );
    assert!(!path.exists());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn filewrite_rejects_invalid_encoding_value() {
    #[cfg(target_arch = "wasm32")]
    let _filesystem = test_support::filesystem::tempdir().expect("browser fixture");
    let path = unique_path("filewrite_invalid_encoding");
    let err = unwrap_error_message(
        run_filewrite(
            Value::from(path.to_string_lossy().to_string()),
            Value::from("data"),
            vec![Value::from("Encoding"), Value::from("utf-32")],
        )
        .unwrap_err(),
    );
    assert!(
        err.contains("unsupported encoding"),
        "unexpected error message: {err}"
    );
    assert!(!path.exists());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn filewrite_accepts_char_array_filename() {
    #[cfg(target_arch = "wasm32")]
    let _filesystem = test_support::filesystem::tempdir().expect("browser fixture");
    let path = unique_path("filewrite_char_path");
    let path_str = path.to_string_lossy();
    let chars: Vec<char> = path_str.chars().collect();
    let units = path_str.encode_utf16().count();
    let char_array = CharArray::from_unicode_scalars(chars, 1, units).expect("char array path");

    run_filewrite(
        Value::CharArray(char_array),
        Value::from("hello"),
        Vec::new(),
    )
    .expect("filewrite char path");

    let written = test_support::fs::read_to_string(&path).expect("read char path file");
    assert_eq!(written, "hello");

    let _ = test_support::fs::remove_file(&path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn filewrite_string_array_stores_newlines() {
    #[cfg(target_arch = "wasm32")]
    let _filesystem = test_support::filesystem::tempdir().expect("browser fixture");
    let path = unique_path("filewrite_string_array");
    let array = StringArray::new(vec!["a".into(), "b".into(), "c".into()], vec![3, 1])
        .expect("string array");
    run_filewrite(
        Value::from(path.to_string_lossy().to_string()),
        Value::StringArray(array),
        Vec::new(),
    )
    .expect("filewrite string array");

    let written = test_support::fs::read_to_string(&path).expect("read string array file");
    assert_eq!(written, "a\nb\nc");

    let _ = test_support::fs::remove_file(&path);
}
