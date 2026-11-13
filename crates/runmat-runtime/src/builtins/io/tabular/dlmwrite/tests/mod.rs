use super::*;

#[test]
fn dlmwrite_path_requires_effective_scalar_text_geometry() {
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
use crate::builtins::common::test_support;
use runmat_time::unix_timestamp_ms;
use std::fs;
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(feature = "wgpu")]
use runmat_accelerate::backend::wgpu::provider as wgpu_provider;
use runmat_accelerate_api::HostTensorView;
use runmat_value::{IntValue, IntegerStorage};

use crate::builtins::common::fs as fs_helpers;

fn dlmwrite_builtin(filename: Value, data: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let _fs_lock = runmat_filesystem::provider_override_lock();
    futures::executor::block_on(super::dlmwrite_builtin(
        None,
        filename,
        data,
        crate::builtins::common::test_support::builtin_arguments(rest),
    ))
}

static NEXT_ID: AtomicU64 = AtomicU64::new(0);

fn error_message(err: RuntimeError) -> String {
    err.message().to_string()
}

fn temp_path(ext: &str) -> PathBuf {
    let millis = unix_timestamp_ms();
    let unique = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    let mut path = std::env::temp_dir();
    path.push(format!(
        "runmat_dlmwrite_{}_{}_{}.{}",
        std::process::id(),
        millis,
        unique,
        ext
    ));
    path
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn dlmwrite_descriptor_signatures_cover_core_forms() {
    let labels: Vec<&str> = DLMWRITE_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"bytesWritten = dlmwrite(filename, M)"));
    assert!(labels.contains(&"bytesWritten = dlmwrite(filename, M, delimiter, row, col)"));
    assert!(labels.contains(&"bytesWritten = dlmwrite(filename, M, name, optionValue)"));
    assert!(labels.contains(&"bytesWritten = dlmwrite(filename, M, args...)"));
    assert!(labels.contains(&"dlmwrite(filename, M)"));
    assert!(labels.contains(&"dlmwrite(filename, M, args...)"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn wasm_precision_parser_handles_common_specs() {
    fn fmt(value: f64, spec: &str) -> String {
        super::format_float(value, spec).expect("formatting failed")
    }

    assert_eq!(fmt(12.3456, "%.2f"), "12.35");
    assert_eq!(fmt(-12.3456, "%+08.1f"), "-00012.3");
    assert_eq!(fmt(0.001234, "%.4g"), "0.001234");
    assert_eq!(fmt(12345.0, "%.3g"), "1.23e+04");
    assert_eq!(fmt(1.5, "%#.0f"), "2.");
    assert_eq!(fmt(1.5, "%#.2e"), "1.50e+00");
    assert_eq!(fmt(1.5, "%#.2G"), "1.5");
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn precision_parser_rejects_unsafe_native_format_shapes() {
    for spec in [
        "%n",
        "%s",
        "%*f",
        "%.2f %.2f",
        "%.2lf",
        "prefix %.2f",
        "%4097f",
        "%.4097f",
    ] {
        let err = super::format_float(1.0, spec).expect_err("format should be rejected");
        assert!(
            err.message().contains("precision format")
                || err.message().contains("unsupported precision format"),
            "unexpected error for {spec}: {}",
            err.message()
        );
    }
}

fn platform_newline() -> &'static str {
    LineEnding::platform_default().as_str()
}

mod data;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn dlmwrite_handles_gpu_tensors() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let view = HostTensorView {
            data: &tensor.materialize_f64(),
            shape: tensor.shape(),
        };
        let handle = provider.upload(&view).unwrap();
        let path = temp_path("csv");
        let filename = path.to_string_lossy().into_owned();
        dlmwrite_builtin(Value::from(filename), Value::GpuTensor(handle), Vec::new()).unwrap();
        let contents = fs::read_to_string(&path).unwrap();
        let nl = platform_newline();
        assert_eq!(contents, format!("1,2,3{nl}"));
        let _ = fs::remove_file(path);
    });
}

#[test]
fn dlmwrite_strict_resident_input_rejects_before_provider_or_file_access() {
    test_support::with_test_provider(|provider| {
        let handle = provider
            .upload(&HostTensorView {
                data: &[1.0, 2.0],
                shape: &[1, 2],
            })
            .expect("upload resident data");
        provider.reset_telemetry();
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let path = temp_path("csv");
        assert!(!path.exists());

        let error = dlmwrite_builtin(
            Value::from(path.to_string_lossy().to_string()),
            Value::GpuTensor(handle.clone()),
            Vec::new(),
        )
        .expect_err("resident data must reject before provider or file access");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:DlmwriteResidentInputExtension")
        );
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        assert!(!path.exists());
        handle.release().expect("free resident data");
    });
}

#[test]
fn dlmwrite_byte_count_output_is_a_gated_extension() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let _outputs = crate::output_count::push_output_count(Some(1)).expect("output-count scope");
    let error = dlmwrite_builtin(
        Value::from("unused.csv"),
        Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap()),
        Vec::new(),
    )
    .expect_err("byte-count output must reject in strict mode");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:DlmwriteByteCountOutputExtension")
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
#[cfg(feature = "wgpu")]
fn dlmwrite_handles_wgpu_provider_gather() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let _ = wgpu_provider::register_wgpu_provider(wgpu_provider::WgpuProviderOptions::default());
    let provider = runmat_accelerate_api::provider().expect("wgpu provider");
    let tensor = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
    let view = HostTensorView {
        data: &tensor.materialize_f64(),
        shape: tensor.shape(),
    };
    let handle = provider.upload(&view).unwrap();
    let path = temp_path("csv");
    let filename = path.to_string_lossy().into_owned();
    dlmwrite_builtin(Value::from(filename), Value::GpuTensor(handle), Vec::new()).unwrap();
    let contents = fs::read_to_string(&path).unwrap();
    let nl = platform_newline();
    assert_eq!(contents, format!("1,2{nl}"));
    let _ = fs::remove_file(path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn dlmwrite_interprets_control_sequence_delimiters() {
    let path = temp_path("txt");
    let tensor = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
    let filename = path.to_string_lossy().into_owned();
    dlmwrite_builtin(
        Value::from(filename),
        Value::Tensor(tensor),
        vec![Value::from("delimiter"), Value::from("\\t")],
    )
    .unwrap();
    let contents = fs::read_to_string(&path).unwrap();
    let nl = platform_newline();
    assert_eq!(contents, format!("1\t2{nl}"));
    let _ = fs::remove_file(path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn dlmwrite_rejects_negative_offsets() {
    let path = temp_path("csv");
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
    let filename = path.to_string_lossy().into_owned();
    let message = error_message(
        dlmwrite_builtin(
            Value::from(filename),
            Value::Tensor(tensor),
            vec![Value::from("roffset"), Value::Int(IntValue::I32(-1))],
        )
        .unwrap_err(),
    );
    assert!(message.to_ascii_lowercase().contains("row offset"));
    assert!(!path.exists());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn dlmwrite_rejects_fractional_offsets() {
    let path = temp_path("csv");
    let tensor = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
    let filename = path.to_string_lossy().into_owned();
    let message = error_message(
        dlmwrite_builtin(
            Value::from(filename),
            Value::Tensor(tensor),
            vec![Value::from("coffset"), Value::Num(1.5)],
        )
        .unwrap_err(),
    );
    assert!(message.to_ascii_lowercase().contains("integer"));
    assert!(!path.exists());
}

#[test]
fn dlmwrite_rejects_unmaterializable_offsets_before_file_access() {
    for (label, args) in [
        (
            "row",
            vec![
                Value::from(","),
                Value::Int(IntValue::U64(u64::MAX)),
                Value::Int(IntValue::U8(0)),
            ],
        ),
        (
            "column",
            vec![
                Value::from(","),
                Value::Int(IntValue::U8(0)),
                Value::Int(IntValue::U64(u64::MAX)),
            ],
        ),
    ] {
        let path = temp_path("csv");
        let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let error = dlmwrite_builtin(
            Value::from(path.to_string_lossy().to_string()),
            Value::Tensor(tensor),
            args,
        )
        .expect_err("unmaterializable offset must reject");
        assert!(
            error.message().contains("offset") || error.message().contains("layout"),
            "unexpected {label} error: {}",
            error.message()
        );
        assert!(!path.exists(), "{label} offset must reject before open");
    }
}

#[test]
fn dlmwrite_layout_validation_checks_addition_product_limit_and_byte_count() {
    let row_overflow = DlmWriteOptions {
        roffset: usize::MAX,
        ..DlmWriteOptions::default()
    };
    assert!(validate_output_layout(1, 1, &row_overflow).is_err());

    let column_overflow = DlmWriteOptions {
        coffset: usize::MAX,
        ..DlmWriteOptions::default()
    };
    assert!(validate_output_layout(1, 1, &column_overflow).is_err());

    let multiplication_overflow = DlmWriteOptions::default();
    assert!(validate_output_layout(usize::MAX / 2 + 1, 2, &multiplication_overflow).is_err());

    let bounded_limit = DlmWriteOptions::default();
    assert!(validate_output_layout(10_000, 10_000, &bounded_limit).is_err());

    let row_limit = DlmWriteOptions {
        roffset: MAX_DLMWRITE_LAYOUT_FIELDS,
        ..DlmWriteOptions::default()
    };
    assert!(validate_output_layout(1, 0, &row_limit).is_err());

    let column_limit = DlmWriteOptions {
        coffset: MAX_DLMWRITE_LAYOUT_FIELDS,
        ..DlmWriteOptions::default()
    };
    assert!(validate_output_layout(0, 1, &column_limit).is_err());

    let mut bytes = usize::MAX;
    assert!(add_written_bytes(&mut bytes, 1).is_err());
    assert_eq!(bytes, usize::MAX);
}

#[test]
fn dlmwrite_integer_tensor_options_preserve_exact_bounds() {
    let offset = Tensor::new_integer(IntegerStorage::U16(vec![7]), vec![1, 1]).expect("offset");
    assert_eq!(
        parse_offset_value(&Value::Tensor(offset), "row offset").unwrap(),
        7
    );
    let negative =
        Tensor::new_integer(IntegerStorage::I16(vec![-1]), vec![1, 1]).expect("negative");
    assert!(parse_offset_value(&Value::Tensor(negative), "row offset").is_err());

    let precision =
        Tensor::new_integer(IntegerStorage::U16(vec![7]), vec![1, 1]).expect("precision");
    assert!(matches!(
        parse_precision_value(&Value::Tensor(precision)),
        Ok(PrecisionSpec::Significant(7))
    ));
    let zero = Tensor::new_integer(IntegerStorage::U16(vec![0]), vec![1, 1]).expect("precision");
    assert!(parse_precision_value(&Value::Tensor(zero)).is_err());
    let too_large = Tensor::new_integer(IntegerStorage::U64(vec![u32::MAX as u64 + 1]), vec![1, 1])
        .expect("precision");
    assert!(parse_precision_value(&Value::Tensor(too_large)).is_err());

    assert!(parse_offset_value(&Value::Num(usize::MAX as f64), "row offset").is_err());
    assert!(parse_offset_value(&Value::Num((usize::MAX as f64) + 1.0), "row offset").is_err());
    assert!(parse_precision_value(&Value::Num((u32::MAX as f64) + 1.0)).is_err());
}

#[test]
fn dlmwrite_structural_options_ignore_poisoned_mirrors_for_every_integer_class() {
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
        let offset = Tensor::new_integer(storage.clone(), vec![1, 1]).expect("offset");
        assert_eq!(
            parse_offset_value(&Value::Tensor(offset), "row offset").unwrap(),
            7
        );

        let precision = Tensor::new_integer(storage, vec![1, 1]).expect("precision");
        assert!(matches!(
            parse_precision_value(&Value::Tensor(precision)),
            Ok(PrecisionSpec::Significant(7))
        ));
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn dlmwrite_rejects_empty_delimiter() {
    let path = temp_path("csv");
    let tensor = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
    let filename = path.to_string_lossy().into_owned();
    let message = error_message(
        dlmwrite_builtin(
            Value::from(filename),
            Value::Tensor(tensor),
            vec![Value::from("")],
        )
        .unwrap_err(),
    );
    assert!(message.to_ascii_lowercase().contains("delimiter"));
    assert!(!path.exists());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn dlmwrite_precision_zero_error() {
    let path = temp_path("csv");
    let tensor = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
    let filename = path.to_string_lossy().into_owned();
    let message = error_message(
        dlmwrite_builtin(
            Value::from(filename),
            Value::Tensor(tensor),
            vec![Value::from("precision"), Value::Int(IntValue::I32(0))],
        )
        .unwrap_err(),
    );
    assert!(message.to_ascii_lowercase().contains("precision"));
    assert!(!path.exists());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn dlmwrite_requires_name_value_pairs() {
    let path = temp_path("csv");
    let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
    let filename = path.to_string_lossy().into_owned();
    let message = error_message(
        dlmwrite_builtin(
            Value::from(filename),
            Value::Tensor(tensor),
            vec![Value::from("delimiter")],
        )
        .unwrap_err(),
    );
    assert!(message
        .to_ascii_lowercase()
        .contains("name-value arguments must appear in pairs"));
    assert!(!path.exists());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn dlmwrite_expands_home_directory() {
    let Some(mut home) = fs_helpers::home_directory() else {
        return;
    };
    let filename = format!(
        "runmat_dlmwrite_home_{}_{}.csv",
        std::process::id(),
        NEXT_ID.fetch_add(1, Ordering::Relaxed)
    );
    home.push(&filename);

    let tilde_path = format!("~/{}", filename);
    let tensor = Tensor::new(vec![42.0], vec![1, 1]).unwrap();

    dlmwrite_builtin(Value::from(tilde_path), Value::Tensor(tensor), Vec::new()).unwrap();

    let contents = fs::read_to_string(&home).unwrap();
    let nl = platform_newline();
    assert_eq!(contents, format!("42{nl}"));
    let _ = fs::remove_file(home);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn dlmwrite_rejects_non_numeric_inputs() {
    let path = temp_path("csv");
    let filename = path.to_string_lossy().into_owned();
    let message = error_message(
        dlmwrite_builtin(
            Value::from(filename),
            Value::String("abc".into()),
            Vec::new(),
        )
        .unwrap_err(),
    );
    assert!(message.contains("dlmwrite"));
}
