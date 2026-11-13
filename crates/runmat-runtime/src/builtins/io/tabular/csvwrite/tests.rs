use super::*;

#[test]
fn csvwrite_path_requires_effective_scalar_text_geometry() {
    let paged = Value::CharArray(
        runmat_value::CharArray::from_unicode_scalars_with_shape(
            "ab".chars().collect(),
            vec![1, 1, 2],
        )
        .unwrap(),
    );
    assert!(resolve_path(&paged).is_err());
    let singleton = Value::CharArray(
        runmat_value::CharArray::from_unicode_scalars_with_shape(
            "ab".chars().collect(),
            vec![1, 2, 1],
        )
        .unwrap(),
    );
    assert_eq!(resolve_path(&singleton).unwrap(), PathBuf::from("ab"));
}
use runmat_time::unix_timestamp_ms;
use std::fs;
use std::sync::atomic::{AtomicU64, Ordering};

use runmat_accelerate_api::HostTensorView;
use runmat_value::{IntValue, IntegerStorage, LogicalArray};

use crate::builtins::common::fs as fs_helpers;
use crate::builtins::common::test_support;

#[path = "tests/cooperation.rs"]
mod cooperation;

fn csvwrite_builtin(filename: Value, data: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let _provider_lock = runmat_filesystem::provider_override_lock();
    futures::executor::block_on(super::csvwrite_builtin(
        None,
        filename,
        data,
        crate::builtins::common::test_support::builtin_arguments(rest),
    ))
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn csvwrite_descriptor_signatures_cover_core_forms() {
    let labels: Vec<&str> = CSVWRITE_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"csvwrite(filename, M)"));
    assert!(labels.contains(&"csvwrite(filename, M, row, col)"));
}

static NEXT_ID: AtomicU64 = AtomicU64::new(0);

fn temp_path(ext: &str) -> PathBuf {
    let millis = unix_timestamp_ms();
    let unique = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    let mut path = std::env::temp_dir();
    path.push(format!(
        "runmat_csvwrite_{}_{}_{}.{}",
        std::process::id(),
        millis,
        unique,
        ext
    ));
    path
}

fn line_ending() -> &'static str {
    "\n"
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn csvwrite_preserves_rank_zero_and_rank_one_column_layout() {
    let _outputs = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
    let cases = [
        (vec![7.0], Vec::new(), "7\n"),
        (vec![7.0, 8.0], vec![2], "7\n8\n"),
    ];
    for (values, shape, expected) in cases {
        let path = temp_path("csv");
        csvwrite_builtin(
            Value::from(path.to_string_lossy().into_owned()),
            Value::Tensor(Tensor::new(values, shape).unwrap()),
            Vec::new(),
        )
        .expect("write exact-rank tensor");
        let contents = fs::read_to_string(&path).expect("read output");
        fs::remove_file(&path).expect("remove test output");
        assert_eq!(contents, expected);
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn csvwrite_writes_basic_matrix() {
    let path = temp_path("csv");
    let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
    let filename = path.to_string_lossy().into_owned();

    csvwrite_builtin(Value::from(filename), Value::Tensor(tensor), Vec::new()).expect("csvwrite");

    let contents = fs::read_to_string(&path).expect("read contents");
    assert_eq!(contents, format!("1,2,3{le}4,5,6{le}", le = line_ending()));
    let _ = fs::remove_file(path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn csvwrite_preserves_typed_integer_matrix_values_exactly() {
    let path = temp_path("csv");
    let tensor = Tensor::new_integer(
        IntegerStorage::U64(vec![u64::MAX, 17, (1_u64 << 53) + 1, 29]),
        vec![2, 2],
    )
    .expect("typed integer matrix");
    let filename = path.to_string_lossy().into_owned();

    csvwrite_builtin(Value::from(filename), Value::Tensor(tensor), Vec::new()).expect("csvwrite");

    let contents = fs::read_to_string(&path).expect("read contents");
    assert_eq!(
        contents,
        format!(
            "18446744073709551615,9007199254740993{le}17,29{le}",
            le = line_ending()
        )
    );
    let _ = fs::remove_file(path);
}

#[test]
fn csvwrite_serializes_all_eight_integer_classes() {
    let cases = [
        (IntegerStorage::I8(vec![-8]), "-8\n"),
        (IntegerStorage::I16(vec![-16]), "-16\n"),
        (IntegerStorage::I32(vec![-32]), "-32\n"),
        (
            IntegerStorage::I64(vec![i64::MIN]),
            "-9223372036854775808\n",
        ),
        (IntegerStorage::U8(vec![8]), "8\n"),
        (IntegerStorage::U16(vec![16]), "16\n"),
        (IntegerStorage::U32(vec![32]), "32\n"),
        (
            IntegerStorage::U64(vec![u64::MAX]),
            "18446744073709551615\n",
        ),
    ];
    for (storage, expected) in cases {
        let path = temp_path("csv");
        let tensor = Tensor::new_integer(storage, vec![1, 1]).unwrap();
        csvwrite_builtin(
            Value::from(path.to_string_lossy().into_owned()),
            Value::Tensor(tensor),
            Vec::new(),
        )
        .unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), expected);
        let _ = fs::remove_file(path);
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn csvwrite_honours_offsets() {
    let path = temp_path("csv");
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let filename = path.to_string_lossy().into_owned();

    csvwrite_builtin(
        Value::from(filename),
        Value::Tensor(tensor),
        vec![Value::Int(IntValue::I32(1)), Value::Int(IntValue::I32(2))],
    )
    .expect("csvwrite");

    let contents = fs::read_to_string(&path).expect("read contents");
    assert_eq!(
        contents,
        format!("{le},,1,3{le},,2,4{le}", le = line_ending())
    );
    let _ = fs::remove_file(path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn csvwrite_handles_gpu_tensors() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    test_support::with_test_provider(|provider| {
        let path = temp_path("csv");
        let tensor = Tensor::new(vec![0.5, 1.5], vec![1, 2]).unwrap();
        let view = HostTensorView {
            data: &tensor.materialize_f64(),
            shape: tensor.shape(),
        };
        let handle = provider.upload(&view).expect("upload");
        let filename = path.to_string_lossy().into_owned();

        csvwrite_builtin(Value::from(filename), Value::GpuTensor(handle), Vec::new())
            .expect("csvwrite");

        let contents = fs::read_to_string(&path).expect("read contents");
        assert_eq!(contents, format!("0.5,1.5{le}", le = line_ending()));
        let _ = fs::remove_file(path);
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn csvwrite_formats_with_short_g_precision() {
    let path = temp_path("csv");
    let values = Tensor::new(vec![12.3456, 1_234_567.0, 0.000123456, -0.0], vec![1, 4]).unwrap();
    let filename = path.to_string_lossy().into_owned();

    csvwrite_builtin(Value::from(filename), Value::Tensor(values), Vec::new()).expect("csvwrite");

    let contents = fs::read_to_string(&path).expect("read contents");
    assert_eq!(
        contents,
        format!("12.346,1.2346e+06,0.00012346,0{le}", le = line_ending())
    );
    let _ = fs::remove_file(path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn csvwrite_rejects_negative_offsets() {
    let path = temp_path("csv");
    let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
    let filename = path.to_string_lossy().into_owned();
    let err = csvwrite_builtin(
        Value::from(filename),
        Value::Tensor(tensor),
        vec![Value::Num(-1.0), Value::Num(0.0)],
    )
    .expect_err("negative offsets should be rejected");
    let message = err.message().to_string();
    assert!(
        message.contains("row offset"),
        "unexpected error message: {message}"
    );
}

#[test]
fn csvwrite_offset_parser_preserves_typed_integer_tensor_bounds() {
    let offset =
        Tensor::new_integer(IntegerStorage::U16(vec![7]), vec![1, 1]).expect("typed offset");
    assert_eq!(
        parse_offset(&Value::Tensor(offset), "row offset").unwrap(),
        7
    );

    let negative =
        Tensor::new_integer(IntegerStorage::I16(vec![-1]), vec![1, 1]).expect("negative");
    assert!(parse_offset(&Value::Tensor(negative), "row offset").is_err());

    let too_large =
        Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1]).expect("too large");
    let parsed = parse_offset(&Value::Tensor(too_large), "row offset");
    if usize::try_from(u64::MAX).is_ok() {
        assert_eq!(parsed.unwrap(), usize::MAX);
    } else {
        assert!(parsed.is_err());
    }

    assert!(parse_offset(&Value::Num(usize::MAX as f64), "row offset").is_err());
    assert!(parse_offset(&Value::Num((usize::MAX as f64) + 1.0), "row offset").is_err());
}

#[cfg(feature = "wgpu")]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn csvwrite_handles_wgpu_provider_gather() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
        runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
    );
    let Some(provider) = runmat_accelerate_api::provider() else {
        panic!("wgpu provider not registered");
    };

    let path = temp_path("csv");
    let tensor = Tensor::new(vec![2.0, 4.0], vec![1, 2]).unwrap();
    let view = HostTensorView {
        data: &tensor.materialize_f64(),
        shape: tensor.shape(),
    };
    let handle = provider.upload(&view).expect("upload");
    let filename = path.to_string_lossy().into_owned();

    csvwrite_builtin(Value::from(filename), Value::GpuTensor(handle), Vec::new())
        .expect("csvwrite");

    let contents = fs::read_to_string(&path).expect("read contents");
    assert_eq!(contents, format!("2,4{le}", le = line_ending()));
    let _ = fs::remove_file(path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn csvwrite_expands_home_directory() {
    let Some(mut home) = fs_helpers::home_directory() else {
        // Skip when home directory cannot be determined.
        return;
    };
    let filename = format!(
        "runmat_csvwrite_home_{}_{}.csv",
        std::process::id(),
        NEXT_ID.fetch_add(1, Ordering::Relaxed)
    );
    home.push(&filename);

    let tilde_path = format!("~/{}", filename);
    let tensor = Tensor::new(vec![42.0], vec![1, 1]).unwrap();

    if let Err(error) = csvwrite_builtin(Value::from(tilde_path), Value::Tensor(tensor), Vec::new())
    {
        if error.message().contains("Operation not permitted")
            || error.message().contains("Permission denied")
        {
            return;
        }
        panic!("csvwrite: {error}");
    }

    let contents = fs::read_to_string(&home).expect("read contents");
    assert_eq!(contents, format!("42{le}", le = line_ending()));
    let _ = fs::remove_file(home);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn csvwrite_rejects_non_numeric_inputs() {
    let path = temp_path("csv");
    let filename = path.to_string_lossy().into_owned();
    let err = csvwrite_builtin(
        Value::from(filename),
        Value::String("abc".into()),
        Vec::new(),
    )
    .expect_err("csvwrite should fail");
    let message = err.message().to_string();
    assert!(
        message.contains("csvwrite"),
        "unexpected error message: {message}"
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn csvwrite_accepts_logical_arrays() {
    let path = temp_path("csv");
    let logical = LogicalArray::new(vec![1, 0, 1, 0], vec![2, 2]).unwrap();
    let filename = path.to_string_lossy().into_owned();

    csvwrite_builtin(
        Value::from(filename),
        Value::LogicalArray(logical),
        Vec::new(),
    )
    .expect("csvwrite");

    let contents = fs::read_to_string(&path).expect("read contents");
    assert_eq!(contents, format!("1,1{le}0,0{le}", le = line_ending()));
    let _ = fs::remove_file(path);
}

#[test]
fn csvwrite_writes_complex_double_and_single_with_lf() {
    for complex in [
        ComplexTensor::new(vec![(1.0, 2.0), (3.0, -4.0)], vec![1, 2]).unwrap(),
        ComplexTensor::from_complex_storage(
            runmat_value::ComplexStorage::F32(vec![(1.0, 2.0), (3.0, -4.0)].into()),
            vec![1, 2],
        )
        .unwrap(),
    ] {
        let path = temp_path("csv");
        csvwrite_builtin(
            Value::from(path.to_string_lossy().into_owned()),
            Value::ComplexTensor(complex),
            Vec::new(),
        )
        .expect("complex csvwrite");
        assert_eq!(fs::read(&path).unwrap(), b"1+2i,3-4i\n");
        let _ = fs::remove_file(path);
    }
}

#[test]
fn csvwrite_declares_independent_output_and_resident_extensions() {
    assert_eq!(CSVWRITE_EXTENSIONS[0].id, "csvwrite-bytes-written-output");
    assert_eq!(CSVWRITE_EXTENSIONS[1].id, "csvwrite-resident-input");
    assert_eq!(CSVWRITE_INTEGER_CAPABILITIES[0].inputs.len(), 3);
    assert!(CSVWRITE_INTEGER_CAPABILITIES[0]
        .inputs
        .iter()
        .all(|input| input.classes.len() == 8));
}

#[test]
fn csvwrite_output_and_resident_gates_run_before_side_effects_or_provider_access() {
    let path = temp_path("csv");
    let filename = Value::from(path.to_string_lossy().into_owned());
    let strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let outputs = crate::output_count::push_output_count(Some(1)).expect("output-count scope");
    let error = csvwrite_builtin(
        filename.clone(),
        Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap()),
        Vec::new(),
    )
    .unwrap_err();
    assert_eq!(
        error.identifier(),
        CSVWRITE_BYTES_OUTPUT_EXTENSION.error_identifier
    );
    assert!(!path.exists(), "output gate must precede file creation");
    drop(outputs);

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
    let error = csvwrite_builtin(filename.clone(), resident, Vec::new()).unwrap_err();
    assert_eq!(
        error.identifier(),
        CSVWRITE_RESIDENT_INPUT_EXTENSION.error_identifier
    );
    assert!(!path.exists(), "resident gate must precede file creation");
    drop(strict);

    let no_outputs = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
    let value = csvwrite_builtin(
        filename,
        Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap()),
        Vec::new(),
    )
    .unwrap();
    assert_eq!(value, Value::output_list(Vec::new()));
    drop(no_outputs);
    let _ = fs::remove_file(path);

    let path = temp_path("csv");
    let filename = Value::from(path.to_string_lossy().into_owned());
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let too_many_outputs =
        crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let error = csvwrite_builtin(
        filename,
        Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap()),
        Vec::new(),
    )
    .unwrap_err();
    assert!(error.message().contains("too many output arguments"));
    assert!(!path.exists(), "output arity must precede file creation");
    drop(too_many_outputs);
}
