use super::*;

fn element(name: &str) -> runmat_value::UnboundedObjectBuilder {
    let mut object = runmat_value::UnboundedObjectBuilder::new(XML_ELEMENT_CLASS);
    object.insert("TagName".into(), char_value(name));
    object
}

fn serialize(value: Value) -> BuiltinResult<String> {
    let mut cooperation = XmlCooperation::new(None, XmlOperation::XmlWrite);
    futures::executor::block_on(serialize_xml_value(&value, &mut cooperation))
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn optional_dom_fields_reject_invalid_values_instead_of_discarding_them() {
    for (field, expected) in [
        ("Attributes", "Attributes must be a scalar struct"),
        ("TextContent", "TextContent must be a string scalar"),
        ("ChildNodes", "ChildNodes must be a cell array"),
    ] {
        let mut object = element("root");
        object.insert(field.into(), Value::Bool(true));
        let error = serialize(Value::Object(object.finish())).expect_err("invalid DOM field");
        assert!(error.message().contains(expected), "{error}");
    }
    assert_eq!(
        serialize(Value::Object(element("root").finish())).unwrap(),
        "<root/>"
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn unicode_names_and_namespace_attributes_are_preserved() {
    let mut object = element("ns:温度");
    let mut attributes = runmat_value::UnboundedStructValueBuilder::new();
    attributes.insert("xmlns:ns", char_value("urn:example"));
    attributes.insert("値", char_value("a&b"));
    object.insert("Attributes".into(), Value::Struct(attributes.finish()));
    object.insert("TextContent".into(), char_value("23"));
    let xml = serialize(Value::Object(object.finish())).unwrap();
    assert!(xml.starts_with("<ns:温度 "));
    assert!(xml.contains("xmlns:ns=\"urn:example\""));
    assert!(xml.contains("値=\"a&amp;b\""));
    assert!(xml.ends_with(">23</ns:温度>"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn struct_text_uses_scalar_geometry_without_losing_wide_integer_values() {
    let scalar = runmat_value::StringArray::new(vec!["text".into()], vec![1, 1]).unwrap();
    assert_eq!(
        scalar_xml_text(&Value::StringArray(scalar)).unwrap(),
        "text"
    );
    let paged =
        runmat_value::CharArray::from_unicode_scalars_with_shape(vec!['a', 'b'], vec![1, 1, 2])
            .unwrap();
    assert!(scalar_xml_text(&Value::CharArray(paged)).is_err());
    let trailing =
        runmat_value::CharArray::from_unicode_scalars_with_shape(vec!['a', 'b'], vec![1, 2, 1])
            .unwrap();
    assert_eq!(scalar_xml_text(&Value::CharArray(trailing)).unwrap(), "ab");
    assert_eq!(
        scalar_xml_text(&Value::Int(runmat_value::IntValue::U64(u64::MAX))).unwrap(),
        u64::MAX.to_string()
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn text_and_attributes_reject_invalid_xml_characters_without_filtering() {
    for ch in [
        '\0', '\u{8}', '\u{B}', '\u{C}', '\u{1F}', '\u{FFFE}', '\u{FFFF}',
    ] {
        let mut object = element("root");
        object.insert(
            "TextContent".into(),
            Value::String(format!("a{ch}b").into()),
        );
        let error = serialize(Value::Object(object.finish())).expect_err("invalid text character");
        assert!(error.message().contains("invalid XML character"));
        let mut object = element("root");
        let mut attributes = runmat_value::UnboundedStructValueBuilder::new();
        attributes.insert("value", Value::String(format!("a{ch}b").into()));
        object.insert("Attributes".into(), Value::Struct(attributes.finish()));
        assert!(serialize(Value::Object(object.finish())).is_err());
    }
    let mut object = element("root");
    object.insert("TextContent".into(), Value::String("a\tb\nc\rd😀".into()));
    assert_eq!(
        serialize(Value::Object(object.finish())).unwrap(),
        "<root>a\tb\nc\rd😀</root>"
    );
}

#[test]
fn malformed_dom_does_not_overwrite_an_existing_file() {
    use runmat_filesystem::FsProvider;
    use std::path::Path;
    use std::sync::Arc;

    let _lock = runmat_filesystem::provider_override_lock();
    let provider = Arc::new(runmat_filesystem::MemoryFsProvider::new());
    futures::executor::block_on(provider.write(Path::new("/output.xml"), b"sentinel")).unwrap();
    let _owner = runmat_filesystem::replace_provider(provider.clone());
    for name in ["1root", "a<b", "a/b", "a=b"] {
        let error = futures::executor::block_on(xmlwrite_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("/output.xml".into()),
                Value::Object(element(name).finish()),
            ]),
        ))
        .expect_err("invalid element name");
        assert!(error.message().contains("invalid XML name"));
        assert_eq!(
            futures::executor::block_on(provider.read(Path::new("/output.xml"))).unwrap(),
            b"sentinel"
        );
    }
    let mut object = element("root");
    let mut attributes = runmat_value::UnboundedStructValueBuilder::new();
    attributes.insert("invalid<attribute", char_value("value"));
    object.insert("Attributes".into(), Value::Struct(attributes.finish()));
    assert!(serialize(Value::Object(object.finish())).is_err());
}
