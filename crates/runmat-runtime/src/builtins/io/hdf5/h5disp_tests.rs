use super::test_arguments::h5write_builtin;
use super::*;
use futures::executor::block_on;
use runmat_value::IntValue;
use tempfile::tempdir;

#[test]
fn h5disp_min_mode_and_utf8_option_use_typed_display_forms() {
    let _fs_lock = runmat_filesystem::provider_override_lock();
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("display.h5");
    let filename = || Value::String(path.display().to_string().into());
    block_on(h5write_builtin(
        None,
        filename(),
        Value::String("/café/measurements".into()),
        Value::Num(3.0),
        Vec::new(),
    ))
    .expect("write Unicode-named dataset");
    block_on(h5writeatt_builtin(
        None,
        filename(),
        Value::String("/café/measurements".into()),
        Value::String("units".into()),
        Value::String("meters".into()),
    ))
    .expect("write attribute");
    block_on(h5writeatt_builtin(
        None,
        filename(),
        Value::String("/café/measurements".into()),
        Value::String("exact".into()),
        Value::Int(IntValue::U64(u64::MAX)),
    ))
    .expect("write exact integer attribute");
    block_on(h5writeatt_builtin(
        None,
        filename(),
        Value::String("/café/measurements".into()),
        Value::String("scale".into()),
        Value::Num(2.5),
    ))
    .expect("write floating attribute");

    let _outputs = crate::output_count::push_output_count(Some(1)).expect("output-count scope");
    let quiet_capture = crate::console::begin_capture().expect("console capture");
    let simple = block_on(h5disp_builtin(
        filename(),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
            "/café".into(),
        )]),
    ))
    .expect("simple display");
    let Value::String(runmat_value::StringElement::Text(simple)) = simple else {
        panic!("expected display text")
    };
    assert!(simple.contains("Dataset: /café/measurements"));
    assert!(simple.contains("Attribute: units"));
    assert!(simple.contains("Value: meters"));
    assert!(simple.contains("Value: 18446744073709551615"));
    assert!(simple.contains("Value: 2.5"));
    assert!(
        quiet_capture.finish().expect("captured text").is_empty(),
        "captured result must not print"
    );

    let minimal = block_on(h5disp_builtin(
        filename(),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("/café".into()),
            Value::String("min".into()),
            Value::String("TextEncoding".into()),
            Value::String("UTF-8".into()),
        ]),
    ))
    .expect("minimal UTF-8 display");
    let Value::String(runmat_value::StringElement::Text(minimal)) = minimal else {
        panic!("expected display text")
    };
    assert!(minimal.contains("Group: /café"));
    assert!(minimal.contains("Dataset: /café/measurements"));
    assert!(!minimal.contains("Attribute: units"));
    assert!(!minimal.contains("[1]"));

    let root_encoded = block_on(h5disp_builtin(
        filename(),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("TextEncoding".into()),
            Value::String("UTF-8".into()),
        ]),
    ))
    .expect("root UTF-8 display without explicit location");
    assert!(
        matches!(root_encoded, Value::String(runmat_value::StringElement::Text(text)) if text.contains("café"))
    );
}

#[test]
fn dispatcher_statement_h5disp_prints_under_suppressed_expression_without_replay() {
    let _fs_lock = runmat_filesystem::provider_override_lock();
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("statement.h5");
    let filename = Value::String(path.display().to_string().into());
    block_on(h5write_builtin(
        None,
        filename.clone(),
        Value::String("/A".into()),
        Value::Num(7.0),
        Vec::new(),
    ))
    .expect("write display fixture");

    let strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let printed = crate::console::begin_capture().expect("console capture");
    block_on(crate::call_builtin_async_with_outputs(
        "h5disp",
        std::slice::from_ref(&filename),
        0,
    ))
    .expect("statement display");
    let printed = printed.finish().expect("captured text");
    assert_eq!(printed.matches("Dataset: /A").count(), 1);

    drop(strict);
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let quiet = crate::console::begin_capture().expect("console capture");
    let value = block_on(crate::call_builtin_async_with_outputs(
        "h5disp",
        &[filename],
        1,
    ))
    .expect("captured text");
    assert!(
        matches!(value.value(), Value::String(runmat_value::StringElement::Text(text)) if text.contains("Dataset: /A"))
    );
    assert!(
        quiet.finish().expect("captured text").is_empty(),
        "captured display must be silent"
    );
}
