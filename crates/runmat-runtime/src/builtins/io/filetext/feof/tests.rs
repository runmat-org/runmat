use super::*;
use crate::builtins::common::test_support;
use crate::builtins::io::filetext::{fclose, fgets, fopen, fread, registry};
use crate::RuntimeError;
use runmat_accelerate_api::HostTensorView;
use runmat_filesystem::File;
#[cfg(not(target_arch = "wasm32"))]
use runmat_time::system_time_now;
use runmat_value::{IntValue, IntegerStorage, Tensor, Value};
use std::io::Write;
use std::path::PathBuf;
#[cfg(not(target_arch = "wasm32"))]
use std::time::UNIX_EPOCH;

fn unwrap_error_message(err: RuntimeError) -> String {
    err.message().to_string()
}

fn run_evaluate(fid_value: &Value) -> BuiltinResult<bool> {
    futures::executor::block_on(evaluate(fid_value))
}

fn run_fopen(args: &[Value]) -> BuiltinResult<fopen::FopenEval> {
    futures::executor::block_on(fopen::evaluate(args))
}

fn run_fread(fid_value: &Value, args: &[Value]) -> BuiltinResult<fread::FreadEval> {
    futures::executor::block_on(fread::evaluate(fid_value, args))
}

fn run_fclose(args: &[Value]) -> BuiltinResult<fclose::FcloseEval> {
    futures::executor::block_on(fclose::evaluate(args))
}

fn registry_guard() -> registry::RegistryTestGuard {
    registry::test_guard()
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn feof_descriptor_signature_covers_core_form() {
    let labels: Vec<&str> = FEOF_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"tf = feof(fid)"));
}

#[test]
fn typed_file_identifier_parser_rejects_unrepresentable_uint64() {
    assert_eq!(parse_fid(&Value::Int(IntValue::U16(7))).unwrap(), 7);
    assert!(parse_fid(&Value::Int(IntValue::U64(u64::MAX))).is_err());

    let fid_tensor =
        Tensor::new_integer(IntegerStorage::U16(vec![7]), vec![1, 1]).expect("typed fid tensor");
    assert_eq!(parse_fid(&Value::Tensor(fid_tensor)).unwrap(), 7);
    let fid_too_large = Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1])
        .expect("typed fid tensor");
    assert!(parse_fid(&Value::Tensor(fid_too_large)).is_err());

    let single = Tensor::new_with_dtype(vec![7.0], vec![1, 1], NumericDType::F32)
        .expect("single file identifier");
    assert!(preflight_fid(&Value::Tensor(single.clone())).is_err());
    assert!(parse_fid(&Value::Tensor(single)).is_err());

    assert!(parse_scalar_fid(f64::from_bits(1.0_f64.to_bits() + 1)).is_err());
}

#[test]
fn feof_typed_identifier_tensors_ignore_poisoned_f64_mirrors() {
    let classes = [
        IntegerStorage::I8(vec![7]),
        IntegerStorage::I16(vec![7]),
        IntegerStorage::I32(vec![7]),
        IntegerStorage::I64(vec![7]),
        IntegerStorage::U8(vec![7]),
        IntegerStorage::U16(vec![7]),
        IntegerStorage::U32(vec![7]),
        IntegerStorage::U64(vec![7]),
    ];
    for storage in classes {
        let tensor = Tensor::new_integer(storage, vec![1, 1]).expect("typed fid");
        assert_eq!(parse_fid(&Value::Tensor(tensor)).unwrap(), 7);
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn feof_returns_false_before_reading() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("feof_false_before_read");
    {
        let mut file = File::create(&path).expect("create");
        file.write_all(b"abc").expect("write");
    }

    let open = run_fopen(&[
        Value::from(path.to_string_lossy().to_string()),
        Value::from("rb"),
    ])
    .expect("fopen");
    let fid = open.as_open().unwrap().fid as i32;

    let at_end = run_evaluate(&Value::Num(fid as f64)).expect("feof");
    assert!(!at_end);

    run_fclose(&[Value::Num(fid as f64)]).unwrap();
    test_support::fs::remove_file(path).unwrap();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn feof_returns_true_after_a_read_encounters_end_of_file() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("feof_true_after_read");
    {
        let mut file = File::create(&path).expect("create");
        file.write_all(&[1u8, 2, 3]).expect("write");
    }

    let open = run_fopen(&[
        Value::from(path.to_string_lossy().to_string()),
        Value::from("rb"),
    ])
    .expect("fopen");
    let fid = open.as_open().unwrap().fid as i32;

    // An unbounded read probes once beyond the available bytes.
    let _ = run_fread(&Value::Num(fid as f64), &Vec::new()).expect("fread");

    let at_end = run_evaluate(&Value::Num(fid as f64)).expect("feof");
    assert!(at_end);

    run_fclose(&[Value::Num(fid as f64)]).unwrap();
    test_support::fs::remove_file(path).unwrap();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn feof_newly_opened_empty_file_is_false_until_a_read_encounters_eof() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("feof_empty_file");
    File::create(&path).expect("create empty");

    let open = run_fopen(&[
        Value::from(path.to_string_lossy().to_string()),
        Value::from("rb"),
    ])
    .expect("fopen");
    let fid = open.as_open().unwrap().fid as i32;

    let at_end = run_evaluate(&Value::Num(fid as f64)).expect("feof");
    assert!(!at_end);

    let _ = run_fread(&Value::Num(fid as f64), &Vec::new()).expect("fread");
    assert!(run_evaluate(&Value::Num(fid as f64)).expect("feof"));

    run_fclose(&[Value::Num(fid as f64)]).unwrap();
    test_support::fs::remove_file(path).unwrap();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn feof_invalid_identifier_errors() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let err = unwrap_error_message(run_evaluate(&Value::Num(42.0)).unwrap_err());
    assert_eq!(err, FEOF_ERROR_INVALID_IDENTIFIER.message);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn feof_rejects_non_integer_identifier() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let err = unwrap_error_message(run_evaluate(&Value::Num(1.5)).unwrap_err());
    assert_eq!(
        err,
        format!(
            "{}: file identifier must be an integer",
            FEOF_ERROR_INVALID_INPUT.message
        )
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn feof_rejects_nan_identifier() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let err = unwrap_error_message(run_evaluate(&Value::Num(f64::NAN)).unwrap_err());
    assert_eq!(
        err,
        format!(
            "{}: file identifier must be finite",
            FEOF_ERROR_INVALID_INPUT.message
        )
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn feof_rejects_negative_identifier() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let err = unwrap_error_message(run_evaluate(&Value::Num(-1.0)).unwrap_err());
    assert_eq!(
        err,
        format!(
            "{}: file identifier must be non-negative",
            FEOF_ERROR_INVALID_INPUT.message
        )
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn feof_rejects_non_numeric_inputs() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let err = unwrap_error_message(run_evaluate(&Value::from("abc")).unwrap_err());
    assert_eq!(
        err,
        format!(
            "{}: {}",
            FEOF_ERROR_INVALID_INPUT.message, IDENTIFIER_TYPE_ERROR_DETAIL
        )
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn feof_accepts_scalar_tensor_identifier() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("feof_tensor_identifier");
    {
        let mut file = File::create(&path).expect("create");
        file.write_all(b"data").expect("write");
    }

    let open = run_fopen(&[
        Value::from(path.to_string_lossy().to_string()),
        Value::from("rb"),
    ])
    .expect("fopen");
    let fid = open.as_open().unwrap().fid as f64;

    let tensor = Tensor::new(vec![fid], vec![1]).unwrap();
    let at_end = run_evaluate(&Value::Tensor(tensor)).expect("feof");
    assert!(!at_end);

    run_fclose(&[Value::Num(fid)]).unwrap();
    test_support::fs::remove_file(path).unwrap();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn feof_errors_on_closed_identifier() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("feof_closed_identifier");
    {
        let mut file = File::create(&path).expect("create");
        file.write_all(b"x").expect("write");
    }

    let open = run_fopen(&[
        Value::from(path.to_string_lossy().to_string()),
        Value::from("rb"),
    ])
    .expect("fopen");
    let fid = open.as_open().unwrap().fid as f64;

    run_fclose(&[Value::Num(fid)]).unwrap();

    let err = unwrap_error_message(run_evaluate(&Value::Num(fid)).unwrap_err());
    assert_eq!(err, FEOF_ERROR_INVALID_IDENTIFIER.message);

    test_support::fs::remove_file(path).unwrap();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn feof_accepts_gpu_identifier_via_gather() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("feof_gpu_identifier");
    {
        let mut file = File::create(&path).expect("create");
        file.write_all(b"xyz").expect("write");
    }

    let open = run_fopen(&[
        Value::from(path.to_string_lossy().to_string()),
        Value::from("rb"),
    ])
    .expect("fopen");
    let fid = open.as_open().unwrap().fid as f64;

    crate::builtins::common::test_support::with_test_provider(|provider| {
        let data = [fid];
        let shape = [1usize];
        let view = HostTensorView {
            data: &data,
            shape: &shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let value = Value::GpuTensor(handle.clone());

        {
            let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
            let err = run_evaluate(&value).expect_err("resident ID must be gated");
            assert_eq!(
                err.identifier(),
                Some("RunMat:compatibility:FeofResidentIdExtension")
            );
        }
        {
            let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
            let at_end = run_evaluate(&value).expect("feof");
            assert!(!at_end);
        }

        handle.release().expect("free");
    });

    run_fclose(&[Value::Num(fid)]).unwrap();
    test_support::fs::remove_file(path).unwrap();
}

#[test]
fn feof_rejects_known_invalid_resident_class_and_shape_before_download() {
    let owner = runmat_accelerate_api::test_support::TestAllocationOwner::new();
    let single = owner
        .allocate_zeroed(
            vec![1, 1],
            runmat_accelerate_api::NumericElementType::F32,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .expect("single allocation");
    let vector = owner
        .allocate_zeroed(
            vec![2, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .expect("vector allocation");
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    for allocation in [&single, &vector] {
        assert!(
            crate::builtins::common::gpu_helpers::exact_provider_for_handle(&allocation.handle)
                .is_none()
        );
        let error = run_evaluate(&Value::GpuTensor(allocation.handle.clone()))
            .expect_err("metadata-invalid identifier");
        assert_eq!(error.identifier(), FEOF_ERROR_INVALID_INPUT.identifier);
        assert_eq!(allocation.probe.retirements(), 0);
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn feof_standard_identifier_returns_false() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let result = run_evaluate(&Value::Num(0.0)).expect("feof");
    assert!(!result);
}

#[test]
fn feof_exact_end_requires_a_subsequent_read_and_frewind_clears_state() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("feof_exact_end_then_rewind");
    {
        let mut file = File::create(&path).expect("create");
        file.write_all(b"abc").expect("write");
    }
    let open = run_fopen(&[
        Value::from(path.to_string_lossy().to_string()),
        Value::from("rb"),
    ])
    .expect("fopen");
    let fid = open.as_open().unwrap().fid;

    let _ = run_fread(&Value::Num(fid), &[Value::Num(3.0)]).expect("exact read");
    assert!(!run_evaluate(&Value::Num(fid)).expect("feof at exact end"));
    let _ = run_fread(&Value::Num(fid), &[Value::Num(1.0)]).expect("eof read");
    assert!(run_evaluate(&Value::Num(fid)).expect("feof after eof read"));

    futures::executor::block_on(crate::builtins::io::filetext::frewind::evaluate(
        &Value::Num(fid),
    ))
    .expect("frewind");
    assert!(!run_evaluate(&Value::Num(fid)).expect("feof after rewind"));

    run_fclose(&[Value::Num(fid)]).unwrap();
    test_support::fs::remove_file(path).unwrap();
}

#[test]
fn fgets_marks_eof_when_final_newline_lookahead_reaches_end() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("feof_fgets_state");
    {
        let mut file = File::create(&path).expect("create");
        file.write_all(b"line\n").expect("write");
    }
    let open = run_fopen(&[
        Value::from(path.to_string_lossy().to_string()),
        Value::from("r"),
    ])
    .expect("fopen");
    let fid = open.as_open().unwrap().fid;

    futures::executor::block_on(fgets::evaluate(&Value::Num(fid), &[]))
        .expect("newline-terminated line");
    assert!(run_evaluate(&Value::Num(fid)).expect("final newline lookahead reached eof"));

    run_fclose(&[Value::Num(fid)]).unwrap();
    test_support::fs::remove_file(path).unwrap();
}

#[test]
fn feof_typed_and_logical_identifiers_are_independently_gated() {
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let integer = run_evaluate(&Value::Int(IntValue::I32(3))).unwrap_err();
    assert_eq!(
        integer.identifier(),
        Some("RunMat:compatibility:FeofIntegerIdExtension")
    );
    let logical = run_evaluate(&Value::Bool(true)).unwrap_err();
    assert_eq!(
        logical.identifier(),
        Some("RunMat:compatibility:FeofLogicalIdExtension")
    );
}

#[cfg(not(target_arch = "wasm32"))]
fn unique_path(prefix: &str) -> PathBuf {
    let now = system_time_now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let filename = format!("{prefix}_{now}.tmp");
    std::env::temp_dir().join(filename)
}

#[cfg(target_arch = "wasm32")]
fn unique_path(prefix: &str) -> PathBuf {
    test_support::filesystem::unique_path(prefix)
}
