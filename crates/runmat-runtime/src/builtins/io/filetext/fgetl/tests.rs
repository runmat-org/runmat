use super::*;
use crate::builtins::common::test_support;
use crate::builtins::io::filetext::{fopen, registry};
use crate::RuntimeError;
use runmat_accelerate_api::HostTensorView;
#[cfg(not(target_arch = "wasm32"))]
use runmat_time::system_time_now;
use runmat_value::{CellArray, IntValue, IntegerStorage, Tensor};
use std::path::{Path, PathBuf};
#[cfg(not(target_arch = "wasm32"))]
use std::time::UNIX_EPOCH;

fn unwrap_error_message(err: RuntimeError) -> String {
    err.message().to_string()
}

fn run_evaluate(fid_value: &Value, rest: &[Value]) -> BuiltinResult<FgetlEval> {
    futures::executor::block_on(evaluate(fid_value, rest))
}

fn run_fopen(args: &[Value]) -> BuiltinResult<fopen::FopenEval> {
    futures::executor::block_on(fopen::evaluate(args))
}

fn registry_guard() -> registry::RegistryTestGuard {
    registry::test_guard()
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fgetl_descriptor_signature_covers_core_form() {
    let labels: Vec<&str> = FGETL_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"tline = fgetl(fid)"));
}

#[test]
fn fgetl_integer_capability_covers_all_typed_file_identifiers() {
    assert_eq!(INTEGER_CAPABILITIES.len(), 1);
    assert_eq!(INTEGER_CAPABILITIES[0].inputs[0].classes.len(), 8);
    assert_eq!(
        INTEGER_CAPABILITIES[0].inputs[0].availability,
        BuiltinIntegerInputAvailability::RunMatOnly
    );
}

#[test]
fn fgetl_typed_single_and_resident_identifiers_are_independently_gated() {
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let integer = preflight_fid(&Value::Int(IntValue::I32(3))).unwrap_err();
    assert_eq!(
        integer.identifier(),
        Some("RunMat:compatibility:FgetlIntegerIdExtension")
    );

    let single = Tensor::new_with_dtype(vec![3.0], vec![1, 1], NumericDType::F32)
        .expect("single identifier");
    let single = preflight_fid(&Value::Tensor(single)).unwrap_err();
    assert_eq!(
        single.identifier(),
        Some("RunMat:compatibility:FgetlSingleIdExtension")
    );

    test_support::with_test_provider(|provider| {
        let host = [3.0f64];
        let view = HostTensorView {
            data: &host,
            shape: &[1, 1],
        };
        let resident = Value::GpuTensor(provider.upload(&view).expect("upload identifier"));
        let resident = preflight_fid(&resident).unwrap_err();
        assert_eq!(
            resident.identifier(),
            Some("RunMat:compatibility:FgetlResidentIdExtension")
        );
    });
}

#[test]
fn fgetl_rejects_known_nonscalar_resident_identifier_before_download() {
    let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![2, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .expect("resident vector");
    assert!(
        crate::builtins::common::gpu_helpers::exact_provider_for_handle(&allocation.handle)
            .is_none()
    );
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let error = run_evaluate(&Value::GpuTensor(allocation.handle.clone()), &[])
        .expect_err("nonscalar identifier");
    assert_eq!(error.identifier(), FGETL_ERROR_INVALID_INPUT.identifier);
    assert_eq!(allocation.probe.retirements(), 0);
}

#[test]
fn fgetl_rejects_cell_containing_resident_identifier_without_gather() {
    let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![1, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .expect("resident scalar");
    assert!(
        crate::builtins::common::gpu_helpers::exact_provider_for_handle(&allocation.handle)
            .is_none()
    );
    let cell = CellArray::new(vec![Value::GpuTensor(allocation.handle.clone())], 1, 1)
        .expect("cell fixture");
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let error = run_evaluate(&Value::Cell(cell), &[]).expect_err("cell is not a file identifier");
    assert_eq!(error.identifier(), FGETL_ERROR_INVALID_INPUT.identifier);
    assert_eq!(allocation.probe.retirements(), 0);
}

#[cfg(not(target_arch = "wasm32"))]
fn unique_path(prefix: &str) -> PathBuf {
    let now = system_time_now()
        .duration_since(UNIX_EPOCH)
        .expect("time went backwards");
    let filename = format!("{}_{}_{}.tmp", prefix, now.as_secs(), now.subsec_nanos());
    std::env::temp_dir().join(filename)
}

#[cfg(target_arch = "wasm32")]
fn unique_path(prefix: &str) -> PathBuf {
    test_support::filesystem::unique_path(prefix)
}

fn fopen_path(path: &Path) -> FopenHandle {
    let eval = run_fopen(&[Value::from(path.to_string_lossy().to_string())]).expect("fopen");
    let open = eval.as_open().expect("open outputs");
    assert!(open.fid >= 3.0);
    FopenHandle {
        fid: open.fid as i32,
    }
}

struct FopenHandle {
    fid: i32,
}

impl Drop for FopenHandle {
    fn drop(&mut self) {
        let _ = registry::close(self.fid);
    }
}

fn char_text(value: Value) -> String {
    match value {
        Value::CharArray(ca) => {
            runmat_value::decode_utf16_code_units(ca.data()).expect("valid UTF-16 fixture")
        }
        other => panic!("expected char array, got {other:?}"),
    }
}

fn assert_empty_char(value: Value) {
    match value {
        Value::CharArray(ca) => {
            assert_eq!(ca.rows, 1);
            assert_eq!(ca.cols, 0);
            assert!(ca.data().is_empty());
        }
        other => panic!("expected empty char array, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn parse_fid_rejects_values_outside_i32_range() {
    let too_large =
        unwrap_error_message(parse_fid(&Value::Num(i32::MAX as f64 + 1.0)).unwrap_err());
    assert_eq!(
        too_large,
        format!(
            "{}: file identifier is out of range",
            FGETL_ERROR_INVALID_INPUT.message
        )
    );

    let too_small =
        unwrap_error_message(parse_fid(&Value::Num(i32::MIN as f64 - 1.0)).unwrap_err());
    assert_eq!(
        too_small,
        format!(
            "{}: file identifier is out of range",
            FGETL_ERROR_INVALID_INPUT.message
        )
    );

    let int_too_large = unwrap_error_message(
        parse_fid(&Value::Int(IntValue::I64(i32::MAX as i64 + 1))).unwrap_err(),
    );
    assert_eq!(
        int_too_large,
        format!(
            "{}: file identifier is out of range",
            FGETL_ERROR_INVALID_INPUT.message
        )
    );

    let uint64_too_large =
        unwrap_error_message(parse_fid(&Value::Int(IntValue::U64(u64::MAX))).unwrap_err());
    assert_eq!(
        uint64_too_large,
        format!(
            "{}: file identifier is out of range",
            FGETL_ERROR_INVALID_INPUT.message
        )
    );

    let typed_tensor =
        Tensor::new_integer(IntegerStorage::U16(vec![7]), vec![1, 1]).expect("typed fid");
    assert_eq!(parse_fid(&Value::Tensor(typed_tensor)).unwrap(), 7);
    let typed_tensor_too_large =
        Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1]).expect("typed fid");
    let typed_tensor_too_large =
        unwrap_error_message(parse_fid(&Value::Tensor(typed_tensor_too_large)).unwrap_err());
    assert_eq!(
        typed_tensor_too_large,
        format!(
            "{}: file identifier is out of range",
            FGETL_ERROR_INVALID_INPUT.message
        )
    );

    let tensor = Tensor::new(vec![i32::MAX as f64 + 1.0], vec![1, 1]).expect("tensor");
    let tensor_too_large = unwrap_error_message(parse_fid(&Value::Tensor(tensor)).unwrap_err());
    assert_eq!(
        tensor_too_large,
        format!(
            "{}: file identifier is out of range",
            FGETL_ERROR_INVALID_INPUT.message
        )
    );
}

#[test]
fn fgetl_typed_identifier_tensors_ignore_poisoned_f64_mirrors() {
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
fn fgetl_reads_line_without_newline() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("fgetl_line");
    test_support::fs::write(&path, "Hello world\nSecond line\n").unwrap();

    let handle = fopen_path(&path);
    let eval = run_evaluate(&Value::Num(handle.fid as f64), &[]).expect("fgetl");
    assert_eq!(char_text(eval.first_output()), "Hello world");

    test_support::fs::remove_file(&path).unwrap();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fgetl_returns_minus_one_at_eof() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("fgetl_eof");
    test_support::fs::write(&path, "line\n").unwrap();
    let handle = fopen_path(&path);

    let _ = run_evaluate(&Value::Num(handle.fid as f64), &[]).expect("first read");
    let eval = run_evaluate(&Value::Num(handle.fid as f64), &[]).expect("second read");
    assert_eq!(eval.first_output(), Value::Num(-1.0));

    test_support::fs::remove_file(&path).unwrap();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fgetl_distinguishes_empty_line_from_eof() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("fgetl_empty");
    test_support::fs::write(&path, "\nlast").unwrap();
    let handle = fopen_path(&path);

    let first = run_evaluate(&Value::Num(handle.fid as f64), &[]).expect("first");
    assert_empty_char(first.first_output());
    let second = run_evaluate(&Value::Num(handle.fid as f64), &[]).expect("second");
    assert_eq!(char_text(second.first_output()), "last");
    let third = run_evaluate(&Value::Num(handle.fid as f64), &[]).expect("third");
    assert_eq!(third.first_output(), Value::Num(-1.0));

    test_support::fs::remove_file(&path).unwrap();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fgetl_handles_crlf_lf_and_cr_newlines() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("fgetl_newlines");
    test_support::fs::write(&path, b"first\r\nsecond\nthird\rfourth").unwrap();
    let handle = fopen_path(&path);

    let first = run_evaluate(&Value::Num(handle.fid as f64), &[]).expect("first");
    assert_eq!(char_text(first.first_output()), "first");
    let second = run_evaluate(&Value::Num(handle.fid as f64), &[]).expect("second");
    assert_eq!(char_text(second.first_output()), "second");
    let third = run_evaluate(&Value::Num(handle.fid as f64), &[]).expect("third");
    assert_eq!(char_text(third.first_output()), "third");
    let fourth = run_evaluate(&Value::Num(handle.fid as f64), &[]).expect("fourth");
    assert_eq!(char_text(fourth.first_output()), "fourth");

    test_support::fs::remove_file(&path).unwrap();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fgetl_consumes_lfcr_as_one_newline_sequence() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("fgetl_lfcr");
    test_support::fs::write(&path, b"first\n\rsecond").unwrap();
    let handle = fopen_path(&path);

    let first = run_evaluate(&Value::Num(handle.fid as f64), &[]).expect("first");
    assert_eq!(char_text(first.first_output()), "first");
    let second = run_evaluate(&Value::Num(handle.fid as f64), &[]).expect("second");
    assert_eq!(char_text(second.first_output()), "second");

    test_support::fs::remove_file(&path).unwrap();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fgetl_preserves_position_across_repeated_reads() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("fgetl_repeated");
    test_support::fs::write(&path, "one\ntwo\nthree\n").unwrap();
    let handle = fopen_path(&path);

    let one = run_evaluate(&Value::Num(handle.fid as f64), &[]).expect("one");
    let two = run_evaluate(&Value::Num(handle.fid as f64), &[]).expect("two");
    let three = run_evaluate(&Value::Num(handle.fid as f64), &[]).expect("three");
    assert_eq!(char_text(one.first_output()), "one");
    assert_eq!(char_text(two.first_output()), "two");
    assert_eq!(char_text(three.first_output()), "three");

    test_support::fs::remove_file(&path).unwrap();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fgetl_errors_for_write_only_identifier() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("fgetl_write_only");
    test_support::fs::write(&path, "payload").unwrap();
    let eval = run_fopen(&[
        Value::from(path.to_string_lossy().to_string()),
        Value::from("w"),
    ])
    .expect("fopen");
    let open = eval.as_open().expect("open outputs");
    assert!(open.fid >= 3.0);

    let err = unwrap_error_message(run_evaluate(&Value::Num(open.fid), &[]).unwrap_err());
    assert_eq!(
        err,
        format!(
            "{}: file identifier is not open for reading",
            FGETL_ERROR_INVALID_IDENTIFIER.message
        )
    );
    let _ = registry::close(open.fid as i32);
    test_support::fs::remove_file(&path).unwrap();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fgetl_errors_for_closed_identifier() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("fgetl_closed");
    test_support::fs::write(&path, "payload").unwrap();
    let handle = fopen_path(&path);
    let fid = handle.fid;
    let _ = registry::close(fid);
    std::mem::forget(handle);

    let err = unwrap_error_message(run_evaluate(&Value::Num(fid as f64), &[]).unwrap_err());
    assert_eq!(err, FGETL_ERROR_INVALID_IDENTIFIER.message);
    test_support::fs::remove_file(&path).unwrap();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fgetl_gathers_gpu_scalar_argument() {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("fgetl_gpu_args");
    test_support::fs::write(&path, b"abcdef\nextra").unwrap();
    let handle = fopen_path(&path);

    test_support::with_test_provider(|provider| {
        let fid_host = [handle.fid as f64];
        let fid_view = HostTensorView {
            data: &fid_host,
            shape: &[1, 1],
        };
        let fid_gpu = Value::GpuTensor(provider.upload(&fid_view).expect("upload fid"));

        let eval = run_evaluate(&fid_gpu, &[]).expect("fgetl");
        assert_eq!(char_text(eval.first_output()), "abcdef");
    });

    test_support::fs::remove_file(&path).unwrap();
}
