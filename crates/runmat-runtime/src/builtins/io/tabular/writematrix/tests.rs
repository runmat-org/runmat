use super::*;
use futures::executor::block_on;
use runmat_time::unix_timestamp_ms;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::builtins::common::test_support;
use runmat_accelerate_api::HostTensorView;
use runmat_value::{IntegerStorage, StringArray, Tensor};

#[path = "tests/admission.rs"]
mod admission;
#[path = "tests/owner.rs"]
mod owner;

async fn writematrix_builtin(data: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let rest = crate::call::BuiltinArguments::from_borrowed(&rest, None)?;
    super::writematrix_builtin(None, data, rest).await
}

static NEXT_ID: AtomicU64 = AtomicU64::new(0);

fn temp_path(ext: &str) -> PathBuf {
    let millis = unix_timestamp_ms();
    let unique = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    let mut path = std::env::temp_dir();
    path.push(format!(
        "runmat_writematrix_{}_{}_{}.{}",
        std::process::id(),
        millis,
        unique,
        ext
    ));
    path
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn writematrix_descriptor_signatures_cover_core_forms() {
    let labels: Vec<&str> = WRITEMATRIX_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"writematrix(data, filename)"));
    assert!(labels.contains(&"writematrix(data, filename, name, optionValue)"));
    assert!(labels.contains(&"writematrix(data, filename, nameValuePairs...)"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn writematrix_writes_space_delimited_txt() {
    let _fs_lock = runmat_filesystem::provider_override_lock();
    let path = temp_path("txt");
    let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
    let filename = path.to_string_lossy().into_owned();

    block_on(writematrix_builtin(
        Value::Tensor(tensor),
        vec![Value::from(filename)],
    ))
    .expect("writematrix");

    let contents = fs::read_to_string(&path).expect("read contents");
    assert_eq!(contents, "1 2 3\n4 5 6\n");
    let _ = fs::remove_file(path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn writematrix_defaults_to_comma_for_csv() {
    let _fs_lock = runmat_filesystem::provider_override_lock();
    let path = temp_path("csv");
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
    let filename = path.to_string_lossy().into_owned();

    block_on(writematrix_builtin(
        Value::Tensor(tensor),
        vec![Value::from(filename)],
    ))
    .expect("writematrix");

    let contents = fs::read_to_string(&path).expect("read contents");
    assert_eq!(contents, "1,2,3\n");
    let _ = fs::remove_file(path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn writematrix_preserves_typed_integer_matrix_values_exactly() {
    let _fs_lock = runmat_filesystem::provider_override_lock();
    let path = temp_path("csv");
    let tensor = Tensor::new_integer(
        IntegerStorage::U64(vec![u64::MAX, 17, (1_u64 << 53) + 1, 29]),
        vec![2, 2],
    )
    .expect("typed integer matrix");
    let filename = path.to_string_lossy().into_owned();

    block_on(writematrix_builtin(
        Value::Tensor(tensor),
        vec![Value::from(filename)],
    ))
    .expect("writematrix");

    let contents = fs::read_to_string(&path).expect("read contents");
    assert_eq!(contents, "18446744073709551615,9007199254740993\n17,29\n");
    let _ = fs::remove_file(path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn writematrix_preserves_public_low_rank_numeric_projection() {
    let _fs_lock = runmat_filesystem::provider_override_lock();
    let rank_zero_path = temp_path("csv");
    let rank_zero = Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), Vec::new())
        .expect("rank-zero integer");
    block_on(writematrix_builtin(
        Value::Tensor(rank_zero),
        vec![Value::from(rank_zero_path.to_string_lossy().into_owned())],
    ))
    .expect("write rank-zero tensor");
    assert_eq!(
        fs::read_to_string(&rank_zero_path).unwrap(),
        "18446744073709551615\n"
    );
    let _ = fs::remove_file(rank_zero_path);

    let rank_one_path = temp_path("csv");
    let rank_one = Tensor::new_integer(
        IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]),
        vec![2],
    )
    .expect("rank-one integer");
    block_on(writematrix_builtin(
        Value::Tensor(rank_one),
        vec![Value::from(rank_one_path.to_string_lossy().into_owned())],
    ))
    .expect("write rank-one tensor");
    assert_eq!(
        fs::read_to_string(&rank_one_path).unwrap(),
        "9007199254740993\n18446744073709551615\n"
    );
    let _ = fs::remove_file(rank_one_path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn writematrix_honours_write_mode_append() {
    let _fs_lock = runmat_filesystem::provider_override_lock();
    let path = temp_path("txt");
    let first = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
    let second = Tensor::new(vec![4.0, 5.0, 6.0], vec![1, 3]).unwrap();
    let filename = path.to_string_lossy().into_owned();

    block_on(writematrix_builtin(
        Value::Tensor(first),
        vec![Value::from(filename.clone())],
    ))
    .expect("initial write");

    block_on(writematrix_builtin(
        Value::Tensor(second),
        vec![
            Value::from(filename.clone()),
            Value::from("WriteMode"),
            Value::from("append"),
        ],
    ))
    .expect("append write");

    let contents = fs::read_to_string(&path).expect("read contents");
    assert_eq!(contents, "1 2 3\n4 5 6\n");
    let _ = fs::remove_file(path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn writematrix_uses_minimal_string_quoting_by_default() {
    let _fs_lock = runmat_filesystem::provider_override_lock();
    let path = temp_path("csv");
    let strings = StringArray::new(
        vec!["Alice".to_string(), "Bob, Jr.".to_string()],
        vec![1, 2],
    )
    .expect("string array");
    let filename = path.to_string_lossy().into_owned();

    block_on(writematrix_builtin(
        Value::StringArray(strings),
        vec![Value::from(filename)],
    ))
    .expect("writematrix");

    let contents = fs::read_to_string(&path).expect("read contents");
    assert_eq!(contents, "Alice,\"Bob, Jr.\"\n");
    let _ = fs::remove_file(path);
}

#[test]
fn writematrix_gates_integer_quote_alias_before_file_access() {
    let strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = block_on(writematrix_builtin(
        Value::Num(1.0),
        vec![
            Value::from("definitely/missing/out.csv"),
            Value::from("QuoteStrings"),
            Value::Int(runmat_value::IntValue::U8(1)),
        ],
    ))
    .expect_err("strict mode rejects integer quote aliases");
    assert_eq!(
        error.identifier(),
        WRITEMATRIX_QUOTE_ALIAS_EXTENSION.error_identifier
    );
    drop(strict);
}

#[test]
fn writematrix_gates_bytes_output_before_file_access() {
    let strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let outputs = crate::output_count::push_output_count(Some(1)).expect("output-count scope");
    let error = block_on(writematrix_builtin(
        Value::Num(1.0),
        vec![Value::from("definitely/missing/out.csv")],
    ))
    .expect_err("strict mode rejects the bytes-written output");
    assert_eq!(
        error.identifier(),
        WRITEMATRIX_BYTES_OUTPUT_EXTENSION.error_identifier
    );
    drop(outputs);
    drop(strict);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn writematrix_accepts_gpu_tensor_inputs() {
    test_support::with_test_provider(|provider| {
        let _fs_lock = runmat_filesystem::provider_override_lock();
        let path = temp_path("csv");
        let tensor = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let view = HostTensorView {
            data: &tensor.materialize_f64(),
            shape: tensor.shape(),
        };
        let handle = provider.upload(&view).expect("upload");
        let filename = path.to_string_lossy().into_owned();

        block_on(writematrix_builtin(
            Value::GpuTensor(handle),
            vec![Value::from(filename)],
        ))
        .expect("writematrix");

        let contents = fs::read_to_string(&path).expect("read contents");
        assert_eq!(contents, "1,2\n");
        let _ = fs::remove_file(path);
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn text_numeric_fields_use_original_single_and_double_storage() {
    let _lock = runmat_filesystem::provider_override_lock();
    for (data, expected) in [
        (
            Tensor::new(vec![1.693061224489796, 1e-20, -0.0], vec![1, 3]).unwrap(),
            "1.6930612244898,1e-20,0\n",
        ),
        (
            Tensor::from_f32(vec![1.2345679, 1e-8, -0.0], vec![1, 3]).unwrap(),
            "1.234568,1e-08,0\n",
        ),
    ] {
        let path = temp_path("csv");
        block_on(writematrix_builtin(
            Value::Tensor(data),
            vec![Value::from(path.to_string_lossy().as_ref())],
        ))
        .expect("write class-aware matrix text");
        assert_eq!(fs::read_to_string(&path).unwrap(), expected);
        let _ = fs::remove_file(path);
    }
    assert_eq!(
        format_numeric(runmat_value::NumericScalar::F32(1.2345679), ','),
        "1,234568",
    );
}
