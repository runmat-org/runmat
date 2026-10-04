use super::*;
use crate::builtins::common::fs::path_to_string;
use crate::builtins::io::repl_fs::REPL_FS_TEST_LOCK;

fn fs_test_guard() -> (
    std::sync::MutexGuard<'static, ()>,
    std::sync::MutexGuard<'static, ()>,
) {
    let provider_guard = runmat_filesystem::provider_override_lock();
    let repl_guard = REPL_FS_TEST_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    (provider_guard, repl_guard)
}

#[test]
fn xml_scalar_text_preserves_exact_uint64() {
    assert_eq!(
        scalar_xml_text(&Value::Int(runmat_value::IntValue::U64(u64::MAX))).expect("xml text"),
        "18446744073709551615"
    );
}

fn run(value: impl std::future::Future<Output = BuiltinResult<Value>>) -> BuiltinResult<Value> {
    futures::executor::block_on(value)
}

#[test]
fn xmlread_returns_lightweight_document_object() {
    let _guard = fs_test_guard();
    let path = std::env::temp_dir().join("runmat_xmlread_test.xml");
    std::fs::write(
        &path,
        "<?xml version=\"1.0\"?><root id=\"r\"><a>text</a></root>",
    )
    .unwrap();
    let value = run(xmlread_builtin(vec![Value::String(path_to_string(&path))])).unwrap();
    match value {
        Value::Object(object) => {
            assert_eq!(object.class_name.display_name(), "org.w3c.dom.Document");
            assert_eq!(
                object.properties.get("DocumentElementName"),
                Some(&char_value("root"))
            );
            let Some(Value::Object(root)) = object.properties.get("DocumentElement") else {
                panic!("expected DocumentElement object");
            };
            assert_eq!(root.properties.get("TagName"), Some(&char_value("root")));
            let Some(Value::Struct(attrs)) = root.properties.get("Attributes") else {
                panic!("expected attributes struct");
            };
            assert_eq!(attrs.fields.get("id"), Some(&char_value("r")));
        }
        other => panic!("unexpected value {other:?}"),
    }
    let _ = std::fs::remove_file(path);
}

#[test]
fn readstruct_imports_xml_children_attributes_and_repeats() {
    let _guard = fs_test_guard();
    let path = std::env::temp_dir().join("runmat_readstruct_test.xml");
    std::fs::write(
            &path,
            r#"<root id="r1"><item code="a">one</item><item code="b">two</item><label>done</label></root>"#,
        )
        .unwrap();
    let value = run(readstruct_builtin(vec![Value::String(path_to_string(
        &path,
    ))]))
    .unwrap();
    let Value::Struct(st) = value else {
        panic!("expected readstruct result");
    };
    assert_eq!(st.fields.get("idAttribute"), Some(&char_value("r1")));
    assert_eq!(st.fields.get("label"), Some(&char_value("done")));
    let Some(Value::Cell(items)) = st.fields.get("item") else {
        panic!("expected repeated item cell");
    };
    assert_eq!(items.rows, 2);
    let Value::Struct(first) = &items.data[0] else {
        panic!("expected first item struct");
    };
    assert_eq!(first.fields.get("codeAttribute"), Some(&char_value("a")));
    assert_eq!(first.fields.get("Text"), Some(&char_value("one")));
    let _ = std::fs::remove_file(path);
}

#[test]
fn readstruct_supports_xml_option_aliases() {
    let _guard = fs_test_guard();
    let path = std::env::temp_dir().join("runmat_readstruct_options_test.xml");
    std::fs::write(&path, r#"<root id="r1">body</root>"#).unwrap();
    let value = run(readstruct_builtin(vec![
        Value::String(path_to_string(&path)),
        char_value("FileType"),
        char_value("xml"),
        char_value("AttributeSuffix"),
        char_value("_attr"),
        char_value("TextNodeName"),
        char_value("Body"),
    ]))
    .unwrap();
    let Value::Struct(st) = value else {
        panic!("expected readstruct result");
    };
    assert_eq!(st.fields.get("id_attr"), Some(&char_value("r1")));
    assert_eq!(st.fields.get("Body"), Some(&char_value("body")));
    let _ = std::fs::remove_file(path);
}

#[test]
fn xmlwrite_serializes_dom_to_string_or_file() {
    let _guard = fs_test_guard();
    let mut root = ObjectInstance::new("org.w3c.dom.Element".to_string());
    root.properties
        .insert("TagName".to_string(), char_value("root"));
    let mut attrs = StructValue::new();
    attrs.insert("id", char_value("r1"));
    root.properties
        .insert("Attributes".to_string(), Value::Struct(attrs));
    root.properties
        .insert("TextContent".to_string(), char_value("hello & goodbye"));
    let xml = run(xmlwrite_builtin(vec![Value::Object(root.clone())])).unwrap();
    assert_eq!(
        xml,
        char_value("<root id=\"r1\">hello &amp; goodbye</root>")
    );

    let path = std::env::temp_dir().join("runmat_xmlwrite_test.xml");
    let bytes = run(xmlwrite_builtin(vec![
        Value::String(path_to_string(&path)),
        Value::Object(root),
    ]))
    .unwrap();
    assert!(matches!(bytes, Value::Num(n) if n > 0.0));
    let text = std::fs::read_to_string(&path).unwrap();
    assert_eq!(text, "<root id=\"r1\">hello &amp; goodbye</root>");
    let _ = std::fs::remove_file(path);
}

#[test]
fn xmlread_rejects_mismatched_tags() {
    let _guard = fs_test_guard();
    let path = std::env::temp_dir().join("runmat_xmlread_bad_test.xml");
    std::fs::write(&path, "<root><a></root>").unwrap();
    let err = run(xmlread_builtin(vec![Value::String(path_to_string(&path))])).unwrap_err();
    assert!(err.message().contains("malformed XML"));
    let _ = std::fs::remove_file(path);
}
