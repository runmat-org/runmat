use super::*;
use futures::executor::block_on;
use tempfile::tempdir;

#[test]
fn h5disp_min_mode_and_utf8_option_use_typed_display_forms() {
    let _fs_lock = runmat_filesystem::provider_override_lock();
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("display.h5");
    let filename = || Value::String(path.display().to_string());
    block_on(h5write_builtin(
        filename(),
        Value::String("/café/measurements".into()),
        Value::Num(3.0),
        Vec::new(),
    ))
    .expect("write Unicode-named dataset");
    block_on(h5writeatt_builtin(
        filename(),
        Value::String("/café/measurements".into()),
        Value::String("units".into()),
        Value::String("meters".into()),
    ))
    .expect("write attribute");

    let _outputs = crate::output_count::push_output_count(Some(1));

    let simple = block_on(h5disp_builtin(
        filename(),
        vec![Value::String("/café".into())],
    ))
    .expect("simple display");
    let Value::String(simple) = simple else {
        panic!("expected display text")
    };
    assert!(simple.contains("Dataset: /café/measurements"));
    assert!(simple.contains("Attribute: units"));

    let minimal = block_on(h5disp_builtin(
        filename(),
        vec![
            Value::String("/café".into()),
            Value::String("min".into()),
            Value::String("TextEncoding".into()),
            Value::String("UTF-8".into()),
        ],
    ))
    .expect("minimal UTF-8 display");
    let Value::String(minimal) = minimal else {
        panic!("expected display text")
    };
    assert!(minimal.contains("Group: /café"));
    assert!(minimal.contains("Dataset: /café/measurements"));
    assert!(!minimal.contains("Attribute: units"));
    assert!(!minimal.contains("[1]"));

    let root_encoded = block_on(h5disp_builtin(
        filename(),
        vec![
            Value::String("TextEncoding".into()),
            Value::String("UTF-8".into()),
        ],
    ))
    .expect("root UTF-8 display without explicit location");
    assert!(matches!(root_encoded, Value::String(text) if text.contains("café")));
}
