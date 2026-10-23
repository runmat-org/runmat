use super::admission::{coerce_to_format_string, preflight_fid, preflight_format};
use super::prepared::preflight_special_roles;
use super::target::parse_fid;
use super::*;
use crate::builtins::common::test_support;
use crate::builtins::io::filetext::{fclose, fopen, registry};
use crate::RuntimeError;
use runmat_accelerate_api::HostTensorView;
use runmat_builtins::FPRINTF_ERROR_INVALID_IDENTIFIER;
use runmat_filesystem::File;
#[cfg(not(target_arch = "wasm32"))]
use runmat_time::system_time_now;
use runmat_value::{CharArray, IntValue, IntegerStorage, Tensor};
use std::io::Read;
use std::path::PathBuf;
#[cfg(not(target_arch = "wasm32"))]
use std::time::UNIX_EPOCH;

mod admission;

fn unwrap_error_message(err: RuntimeError) -> String {
    err.message().to_string()
}

fn run_evaluate(args: &[Value]) -> BuiltinResult<FprintfEval> {
    futures::executor::block_on(evaluate(args))
}

fn run_fopen(args: &[Value]) -> BuiltinResult<fopen::FopenEval> {
    futures::executor::block_on(fopen::evaluate(args))
}

fn run_fclose(args: &[Value]) -> BuiltinResult<fclose::FcloseEval> {
    futures::executor::block_on(fclose::evaluate(args))
}

fn registry_guard() -> registry::RegistryTestGuard {
    registry::test_guard()
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fprintf_descriptor_signatures_cover_core_forms() {
    let labels: Vec<&str> = FPRINTF_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"count = fprintf(formatSpec, A...)"));
    assert!(labels.contains(&"count = fprintf(fid_or_stream, formatSpec, A...)"));
}

#[test]
fn fprintf_integer_capabilities_and_special_roles_are_independently_gated() {
    assert_eq!(INTEGER_CAPABILITIES.len(), 3);
    assert_eq!(INTEGER_CAPABILITIES[0].inputs[0].classes.len(), 8);
    let _matlab = crate::compatibility::push_runmat_extensions_enabled(false);
    let fid = preflight_fid(&Value::Int(IntValue::I32(1))).unwrap_err();
    assert_eq!(
        fid.identifier(),
        Some("RunMat:compatibility:FprintfIntegerIdExtension")
    );
    let format = Tensor::new_integer(
        IntegerStorage::U16(vec![b'%' as u16, b'd' as u16]),
        vec![1, 2],
    )
    .expect("format");
    let format = preflight_format(&Value::Tensor(format)).unwrap_err();
    assert_eq!(
        format.identifier(),
        Some("RunMat:compatibility:FprintfIntegerFormatExtension")
    );
}

#[test]
fn fprintf_resident_integer_format_is_gated_before_gathering() {
    let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_numeric(runmat_accelerate_api::HostNumericTensorOwned {
            data: runmat_accelerate_api::HostNumericDataOwned::U16(vec![b'%' as u16, b'd' as u16]),
            shape: vec![1, 2],
            storage: runmat_accelerate_api::GpuTensorStorage::Real,
        })
        .expect("native uint16 format allocation");
    let original = allocation.handle.wire_descriptor();
    assert!(
        crate::builtins::common::gpu_helpers::exact_provider_for_handle(&allocation.handle)
            .is_none()
    );
    let args = [Value::Num(1.0), Value::GpuTensor(allocation.handle.clone())];
    let _matlab = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = preflight_special_roles(&args).unwrap_err();
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:FprintfResidentFormatExtension")
    );
    drop(args);
    assert_eq!(allocation.handle.wire_descriptor(), original);
    assert_eq!(allocation.probe.retirements(), 0);
    drop(allocation.handle);
    assert_eq!(allocation.probe.retirements(), 1);
}

#[test]
fn fprintf_all_host_numeric_format_tensors_are_gated_before_gathering() {
    let double = Tensor::new(vec![b'%' as f64, b'd' as f64], vec![1, 2]).expect("double");
    let single = Tensor::from_numeric_storage(
        runmat_value::NumericStorage::F32(vec![b'%' as f32, b'd' as f32]),
        vec![1, 2],
    )
    .expect("single");
    let _matlab = crate::compatibility::push_runmat_extensions_enabled(false);
    for format in [double, single] {
        let error = preflight_format(&Value::Tensor(format)).unwrap_err();
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:FprintfNumericFormatExtension")
        );
    }
}

#[test]
fn fprintf_textual_stream_labels_have_an_independent_gate() {
    let _matlab = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = preflight_special_roles(&[
        Value::from("stderr"),
        Value::from("failure: %d"),
        Value::Num(1.0),
    ])
    .unwrap_err();
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:FprintfStreamLabelExtension")
    );
}

#[test]
fn fprintf_integer_string_conversion_preserves_character_codes() {
    let tensor = Tensor::new_integer(IntegerStorage::U64(vec![65, 66, 67]), vec![1, 3])
        .expect("character codes");
    let format = crate::builtins::common::format::validate_format_syntax("[%s]").unwrap();
    let args = coerce_single_integer_string_argument(format, vec![Value::Tensor(tensor)])
        .expect("integer string conversion");
    assert_eq!(args, vec![Value::String("ABC".to_string().into())]);
    assert_eq!(format_with_repetition("[%s]", &args).unwrap(), "[ABC]");
}

#[test]
fn numeric_format_accepts_trailing_singletons_but_not_extra_pages() {
    let valid = Value::Tensor(Tensor::new(vec![b'%' as f64, b'd' as f64], vec![1, 2, 1]).unwrap());
    assert!(matches!(
        coerce_to_format_string(&valid),
        Ok(Some(Value::String(_)))
    ));
    let invalid = Value::Tensor(
        Tensor::new(
            vec![b'%' as f64, b'd' as f64, b'x' as f64, b'y' as f64],
            vec![1, 2, 2],
        )
        .unwrap(),
    );
    assert!(coerce_to_format_string(&invalid).unwrap().is_none());
    let char_row = Value::CharArray(
        CharArray::from_unicode_scalars_with_shape(vec!['%', 'd'], vec![1, 2, 1]).unwrap(),
    );
    assert!(matches!(
        coerce_to_format_string(&char_row),
        Ok(Some(Value::String(_)))
    ));
    let char_pages = Value::CharArray(
        CharArray::from_unicode_scalars_with_shape(vec!['%', 'd', '%', 'd'], vec![1, 2, 2])
            .unwrap(),
    );
    assert!(coerce_to_format_string(&char_pages).is_err());
}

#[test]
fn fprintf_formats_wide_uint64_without_binary64_rounding() {
    let value = Value::Int(IntValue::U64(9_007_199_254_740_993));
    assert_eq!(
        format_with_repetition("%u", &[value]).unwrap(),
        "9007199254740993"
    );
}

#[test]
fn fprintf_format_tensor_reads_typed_integer_storage_exactly() {
    let format = Tensor::new_integer(
        IntegerStorage::U16(vec![b'%' as u16, b'd' as u16]),
        vec![1, 2],
    )
    .expect("format tensor");

    assert_eq!(
        coerce_to_format_string(&Value::Tensor(format)).unwrap(),
        Some(Value::String("%d".to_string().into()))
    );
}

#[test]
fn fprintf_fid_parser_reads_typed_integer_storage_exactly() {
    let fid = Tensor::new_integer(IntegerStorage::U16(vec![7]), vec![1, 1]).expect("fid tensor");
    assert_eq!(parse_fid(&Value::Tensor(fid)).unwrap(), 7);

    let too_large =
        Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1]).expect("fid");
    assert!(parse_fid(&Value::Tensor(too_large)).is_err());
}

#[test]
fn fprintf_fid_parser_accepts_canonical_singleton_geometry() {
    for shape in [vec![1], vec![1, 1], vec![1, 1, 1]] {
        let fid = Tensor::new(vec![1.0], shape).expect("singleton tensor");
        assert_eq!(parse_fid(&Value::Tensor(fid.clone())).unwrap(), 1);
        assert!(super::admission::is_fid_candidate(&Value::Tensor(fid)));
    }
}

#[test]
fn fprintf_fid_parser_ignores_poisoned_mirrors_for_every_integer_class() {
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
        let fid = Tensor::new_integer(storage, vec![1, 1]).expect("typed fid");
        assert_eq!(parse_fid(&Value::Tensor(fid)).unwrap(), 7);
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fprintf_matrix_column_major() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("fprintf_matrix");
    let open = run_fopen(&[
        Value::from(path.to_string_lossy().to_string()),
        Value::from("w"),
    ])
    .expect("fopen");
    let fid = open.as_open().unwrap().fid as i32;

    let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
    let args = vec![
        Value::Num(fid as f64),
        Value::String("%d %d\n".to_string().into()),
        Value::Tensor(tensor),
    ];
    let eval = run_evaluate(&args).expect("fprintf");
    assert_eq!(eval.bytes_written(), 12);

    run_fclose(&[Value::Num(fid as f64)]).unwrap();

    let contents = test_support::fs::read_to_string(&path).expect("read");
    assert_eq!(contents, "1 4\n2 5\n3 6\n");
    test_support::fs::remove_file(path).unwrap();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fprintf_uses_shared_scientific_and_general_conversions() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("fprintf_floating");
    let open = run_fopen(&[
        Value::from(path.to_string_lossy().to_string()),
        Value::from("w"),
    ])
    .expect("fopen");
    let fid = open.as_open().unwrap().fid as i32;
    let eval = run_evaluate(&[
        Value::Num(fid as f64),
        Value::String("%E|%G|%d".to_string().into()),
        Value::Num(1234.5),
        Value::Num(1.0e-10),
        Value::Num(2.5),
    ])
    .expect("fprintf");
    let expected = "1.234500E+03|1E-10|2.500000e+00";
    assert_eq!(eval.bytes_written(), expected.len());
    run_fclose(&[Value::Num(fid as f64)]).unwrap();
    assert_eq!(
        test_support::fs::read_to_string(&path).expect("read"),
        expected
    );
    test_support::fs::remove_file(path).unwrap();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fprintf_ascii_encoding_errors() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("fprintf_ascii");
    let open = run_fopen(&[
        Value::from(path.to_string_lossy().to_string()),
        Value::from("w"),
        Value::from("native"),
        Value::from("ascii"),
    ])
    .expect("fopen");
    let fid = open.as_open().unwrap().fid as i32;

    let args = vec![
        Value::Num(fid as f64),
        Value::String("%s".to_string().into()),
        Value::String("café".to_string().into()),
    ];
    let err = unwrap_error_message(run_evaluate(&args).unwrap_err());
    assert!(err.contains("cannot be encoded as ASCII"), "{err}");

    run_fclose(&[Value::Num(fid as f64)]).unwrap();
    test_support::fs::remove_file(path).unwrap();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fprintf_gpu_gathers_values() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("fprintf_gpu");

    test_support::with_test_provider(|provider| {
        registry::reset_for_tests();
        let open = run_fopen(&[
            Value::from(path.to_string_lossy().to_string()),
            Value::from("w"),
        ])
        .expect("fopen");
        let fid = open.as_open().unwrap().fid as i32;

        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let view = HostTensorView {
            data: &tensor.materialize_f64(),
            shape: tensor.shape(),
        };
        let handle = provider.upload(&view).expect("upload");
        let args = vec![
            Value::Num(fid as f64),
            Value::String("%.1f,".to_string().into()),
            Value::GpuTensor(handle),
        ];
        let eval = run_evaluate(&args).expect("fprintf");
        assert_eq!(eval.bytes_written(), 12);

        run_fclose(&[Value::Num(fid as f64)]).unwrap();
    });

    let mut file = File::open(&path).expect("open");
    let mut contents = String::new();
    file.read_to_string(&mut contents).expect("read");
    assert_eq!(contents, "1.0,2.0,3.0,");
    test_support::fs::remove_file(path).unwrap();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fprintf_missing_format_errors() {
    let err = unwrap_error_message(run_evaluate(&[Value::Num(1.0)]).unwrap_err());
    assert!(err.contains("missing format string"), "{err}");
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fprintf_literal_with_extra_args_errors() {
    let err = unwrap_error_message(
        run_evaluate(&[
            Value::String("literal text".to_string().into()),
            Value::Int(IntValue::I32(1)),
        ])
        .unwrap_err(),
    );
    assert!(err.contains("contains no conversion specifiers"), "{err}");
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fprintf_invalid_identifier_errors() {
    let err = unwrap_error_message(
        run_evaluate(&[Value::Num(99.0), Value::String("value".to_string().into())]).unwrap_err(),
    );
    assert_eq!(err, FPRINTF_ERROR_INVALID_IDENTIFIER.message);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fprintf_read_only_error() {
    let _guard = registry_guard();
    registry::reset_for_tests();
    let path = unique_path("fprintf_read_only");
    test_support::fs::write(&path, b"readonly").unwrap();
    let open = run_fopen(&[
        Value::from(path.to_string_lossy().to_string()),
        Value::from("r"),
    ])
    .expect("fopen");
    let fid = open.as_open().unwrap().fid as i32;
    let err = unwrap_error_message(
        run_evaluate(&[
            Value::Num(fid as f64),
            Value::String("text".to_string().into()),
        ])
        .unwrap_err(),
    );
    assert!(err.contains("not open for writing"), "{err}");

    run_fclose(&[Value::Num(fid as f64)]).unwrap();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fprintf_encoding_aliases_encode_expected_bytes() {
    let utf = encode_output("é", Some("utf_8")).expect("utf_8 alias");
    assert_eq!(utf, "é".as_bytes());

    let latin = encode_output("é", Some("cp819")).expect("cp819 alias");
    assert_eq!(latin, vec![0xE9]);

    let win = encode_output("€’", Some("windows-1252")).expect("windows-1252 alias");
    assert_eq!(win, vec![0x80, 0x92]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fprintf_windows1252_reports_unencodable_characters() {
    let err = encode_output("Ā", Some("cp1252")).expect_err("cp1252 should reject U+0100");
    assert!(err.contains("cannot be encoded"), "{err}");
}

#[cfg(not(target_arch = "wasm32"))]
fn unique_path(prefix: &str) -> PathBuf {
    let nanos = system_time_now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let filename = format!("runmat_{prefix}_{nanos}.txt");
    std::env::temp_dir().join(filename)
}

#[cfg(target_arch = "wasm32")]
fn unique_path(prefix: &str) -> PathBuf {
    test_support::filesystem::unique_path(prefix)
}
