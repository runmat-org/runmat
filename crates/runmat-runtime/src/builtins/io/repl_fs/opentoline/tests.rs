use super::position::host_position;
use super::*;
use futures::executor::block_on;
use runmat_value::IntValue;
use tempfile::tempdir;

mod selected_context;

fn call(args: Vec<Value>) -> BuiltinResult<Value> {
    let outputs = crate::output_count::current_output_count();
    let (context, _) = crate::context::document_test_support::recording_context();
    let selected = context.clone();
    block_on(context.scope(async move {
        let _outputs = crate::output_count::push_output_count(outputs).expect("output-count scope");
        opentoline_builtin(
            Some(selected),
            crate::builtins::common::test_support::builtin_arguments(args),
        )
        .await
    }))
}

#[test]
fn navigation_delivers_exact_typed_position_and_requires_host_support() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("navigate.m");
    std::fs::write(&path, "value = 1;\n").unwrap();
    let args = vec![
        Value::from(path.to_string_lossy().into_owned()),
        Value::Num(12.0),
        Value::Num(4.0),
        Value::from("select"),
    ];
    assert_eq!(
        block_on(opentoline_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(args.clone())
        ))
        .unwrap_err()
        .identifier(),
        Some("RunMat:host:DocumentInteractionUnavailable")
    );
    let (context, recorder) = crate::context::document_test_support::recording_context();
    block_on(opentoline_builtin(
        Some(context),
        crate::builtins::common::test_support::builtin_arguments(args),
    ))
    .unwrap();
    assert_eq!(
        *recorder.requests.borrow(),
        vec![crate::context::HostDocumentRequest::Navigate {
            path: std::fs::canonicalize(path).unwrap(),
            location: crate::context::HostTextLocation {
                line: std::num::NonZeroUsize::new(12).unwrap(),
                column: std::num::NonZeroUsize::new(4).unwrap(),
                selection: crate::context::HostTextSelection::Select,
            },
        }]
    );
}

#[test]
fn option_and_host_position_admission_precede_any_resident_download() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let resident = Value::GpuTensor(
        crate::builtins::io::repl_fs::test_support::unregistered_numeric_allocation(vec![1, 1])
            .handle,
    );
    let args = vec![
        Value::from("missing.m"),
        resident.clone(),
        Value::Num(1.0),
        Value::from("bad-option"),
    ];
    assert_eq!(
        block_on(opentoline_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(args)
        ))
        .unwrap_err()
        .identifier(),
        OPENTOLINE_ERROR_OPTION.identifier
    );
    let args = vec![Value::from("missing.m"), resident, Value::Num(-1.0)];
    assert_eq!(
        block_on(opentoline_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(args)
        ))
        .unwrap_err()
        .identifier(),
        OPENTOLINE_ERROR_POSITION.identifier
    );
    let pages = Value::CharArray(
        runmat_value::CharArray::from_unicode_scalars_with_shape(vec!['a', 'b'], vec![1, 1, 2])
            .unwrap(),
    );
    assert!(filename_arg(&pages).is_err());
}

#[test]
fn opentoline_resolves_file_and_returns_empty_array() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("target.m");
    std::fs::write(&path, "a = 1;\nb = 2;\n").expect("write file");

    let result = call(vec![
        Value::String(path.to_string_lossy().to_string().into()),
        Value::Num(2.0),
    ])
    .expect("opentoline");

    assert_eq!(result, empty_array());
}

#[test]
fn opentoline_typed_positions_preserve_integer_bounds() {
    assert_eq!(
        host_position(&Value::Int(IntValue::U16(7)), "line").unwrap(),
        7
    );
    assert!(host_position(&Value::Int(IntValue::I8(-1)), "line").is_err());
    assert!(host_position(&Value::Int(IntValue::U8(0)), "column").is_err());

    let large = 9_007_199_254_740_993_u64;
    let parsed = host_position(&Value::Int(IntValue::U64(large)), "line");
    if usize::BITS == 64 {
        assert_eq!(parsed.unwrap(), large as usize);
    } else {
        assert!(parsed.is_err());
    }
}

#[test]
fn every_native_integer_class_decodes_positions_and_rejects_zero() {
    let values = [
        (IntValue::I8(1), IntValue::I8(0)),
        (IntValue::U8(1), IntValue::U8(0)),
        (IntValue::I16(1), IntValue::I16(0)),
        (IntValue::U16(1), IntValue::U16(0)),
        (IntValue::I32(1), IntValue::I32(0)),
        (IntValue::U32(1), IntValue::U32(0)),
        (IntValue::I64(1), IntValue::I64(0)),
        (IntValue::U64(1), IntValue::U64(0)),
    ];
    for (positive, zero) in values {
        assert_eq!(host_position(&Value::Int(positive), "line").unwrap(), 1);
        assert_eq!(
            host_position(&Value::Int(zero), "column")
                .unwrap_err()
                .identifier(),
            OPENTOLINE_ERROR_POSITION.identifier
        );
    }
}

#[test]
fn opentoline_double_positions_reject_unrepresentable_platform_bounds() {
    let boundary = host_position(&Value::Num(usize::MAX as f64), "line");
    if usize::BITS == 64 {
        assert!(boundary.is_err());
    } else {
        assert_eq!(boundary.unwrap(), usize::MAX);
    }
    assert!(host_position(&Value::Num((usize::MAX as f64) + 1.0), "line").is_err());
}

#[test]
fn opentoline_tensor_positions_read_integer_storage_exactly() {
    let line = Tensor::new_integer(runmat_value::IntegerStorage::U64(vec![9]), vec![1, 1]).unwrap();

    assert_eq!(host_position(&Value::Tensor(line), "line").unwrap(), 9);

    let zero = Tensor::new_integer(runmat_value::IntegerStorage::U8(vec![0]), vec![1, 1]).unwrap();
    assert!(host_position(&Value::Tensor(zero), "line").is_err());
}

#[test]
fn opentoline_typed_positions_are_independently_gated_before_file_lookup() {
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let line_error = call(vec![
        Value::String("definitely-missing.m".into()),
        Value::Int(IntValue::U16(7)),
    ])
    .expect_err("typed line must be gated");
    assert_eq!(
        line_error.identifier(),
        Some("RunMat:compatibility:OpentolineIntegerLineExtension")
    );

    let column_error = call(vec![
        Value::String("definitely-missing.m".into()),
        Value::Num(1.0),
        Value::Int(IntValue::U16(3)),
    ])
    .expect_err("typed column must be gated");
    assert_eq!(
        column_error.identifier(),
        Some("RunMat:compatibility:OpentolineIntegerColumnExtension")
    );
}

#[test]
fn opentoline_resident_position_is_gated_without_provider_access() {
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let resident = Value::GpuTensor(
        crate::builtins::io::repl_fs::test_support::unregistered_numeric_allocation(vec![1, 1])
            .handle
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .expect("explicit resident position"),
    );
    let error = call(vec![Value::String("definitely-missing.m".into()), resident])
        .expect_err("resident line must be gated");

    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:OpentolineResidentPositionExtension")
    );
}

#[test]
fn opentoline_automatically_resident_double_position_gathers_in_strict_mode() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("target.m");
    std::fs::write(&path, "a = 1;\n").expect("write file");
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let line = Tensor::new(vec![1.0], vec![1, 1]).expect("line");
        let handle = crate::builtins::common::gpu_helpers::upload_tensor(provider, &line)
            .expect("resident line");
        let handle = handle
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
            .expect("automatic resident position");
        let result = {
            let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
            call(vec![
                Value::String(path.to_string_lossy().into_owned().into()),
                Value::GpuTensor(handle.clone()),
            ])
            .expect("automatic position gathers transparently")
        };
        assert_eq!(result, empty_array());
        handle.allocation_lease().validate_live().unwrap();
    });
}

#[test]
fn opentoline_accepts_column_and_select_option() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("target.m");
    std::fs::write(&path, "abcdef\n").expect("write file");

    let result = call(vec![
        Value::String(path.to_string_lossy().to_string().into()),
        Value::Num(1.0),
        Value::Num(4.0),
        Value::String("select".to_string().into()),
    ])
    .expect("opentoline");

    assert_eq!(result, empty_array());
}

#[test]
fn opentoline_command_form_text_line_and_column_are_supported() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("target.m");
    std::fs::write(&path, "abcdef\n").expect("write file");

    let result = call(vec![
        Value::String(path.to_string_lossy().to_string().into()),
        Value::String("1".to_string().into()),
        Value::String("3".to_string().into()),
    ])
    .expect("opentoline");

    assert_eq!(result, empty_array());
}

#[test]
fn opentoline_rejects_missing_file() {
    let err = call(vec![
        Value::String("definitely_missing_opentoline_target.m".to_string().into()),
        Value::Num(1.0),
    ])
    .expect_err("missing file");

    assert_eq!(err.identifier(), Some("RunMat:opentoline:FileNotFound"));
}

#[test]
fn opentoline_rejects_invalid_positions_and_options() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("target.m");
    std::fs::write(&path, "abcdef\n").expect("write file");
    let filename = Value::String(path.to_string_lossy().to_string().into());

    let err = call(vec![filename.clone(), Value::Num(0.0)]).expect_err("invalid line");
    assert_eq!(err.identifier(), Some("RunMat:opentoline:InvalidPosition"));

    let err = call(vec![filename.clone(), Value::Num(1.5)]).expect_err("fractional line");
    assert_eq!(err.identifier(), Some("RunMat:opentoline:InvalidPosition"));

    let err = call(vec![
        filename,
        Value::Num(1.0),
        Value::Num(1.0),
        Value::String("reuse".to_string().into()),
    ])
    .expect_err("invalid option");
    assert_eq!(err.identifier(), Some("RunMat:opentoline:InvalidOption"));
}

#[test]
fn opentoline_rejects_too_many_outputs() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("target.m");
    std::fs::write(&path, "abcdef\n").expect("write file");
    let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let err = call(vec![
        Value::String(path.to_string_lossy().to_string().into()),
        Value::Num(1.0),
    ])
    .expect_err("too many outputs");

    assert_eq!(err.identifier(), Some("RunMat:opentoline:TooManyOutputs"));
}

#[test]
fn opentoline_descriptor_covers_core_forms() {
    let labels: Vec<&str> = OPENTOLINE_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"opentoline(filename, line)"));
    assert!(labels.contains(&"opentoline(filename, line, column)"));
    assert!(labels.contains(&"opentoline(filename, line, column, option)"));
}
