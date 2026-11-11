use super::*;
async fn readstruct_builtin(
    runtime: Option<RuntimeContext>,
    args: Vec<Value>,
) -> BuiltinResult<Value> {
    super::readstruct_builtin(
        runtime,
        crate::execution::RootedValues::new(args)
            .map_err(|error| error.into_runtime_error())?
            .into(),
    )
    .await
}
async fn xmlwrite_builtin(
    runtime: Option<RuntimeContext>,
    args: Vec<Value>,
) -> BuiltinResult<Value> {
    super::xmlwrite_builtin(
        runtime,
        crate::execution::RootedValues::new(args)
            .map_err(|error| error.into_runtime_error())?
            .into(),
    )
    .await
}
use crate::builtins::common::fs::path_to_string;
use runmat_filesystem::FsProvider;

// The DOM projection contains only XML metadata, not arbitrary GC children.
// Preserve the old test adapter while cloning under its original root owner.
async fn xmlread_builtin(
    runtime: Option<RuntimeContext>,
    args: Vec<Value>,
) -> BuiltinResult<Value> {
    super::xmlread_builtin(
        runtime,
        crate::execution::RootedValues::new(args)
            .map_err(|error| error.into_runtime_error())?
            .into(),
    )
    .await
    .map(|rooted| rooted.value().clone())
}

#[path = "tests/write_owner.rs"]
mod write_owner;

fn fs_test_guard() -> std::sync::MutexGuard<'static, ()> {
    runmat_filesystem::provider_override_lock()
}

#[test]
fn xml_scalar_text_preserves_exact_uint64() {
    assert_eq!(
        scalar_xml_text(&Value::Int(runmat_value::IntValue::U64(u64::MAX))).expect("xml text"),
        "18446744073709551615"
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn xml_diagnostics_keep_original_message_and_source() {
    let error = xml_error_with_source(
        XmlOperation::ReadStruct,
        XmlDiagnosticKind::InvalidOption,
        compat_error("readstruct", "readstruct: unsupported option 'bogus'"),
    );
    assert_eq!(error.identifier(), Some("RunMat:readstruct:InvalidOption"));
    assert_eq!(error.message(), "readstruct: unsupported option 'bogus'");
    assert!(std::error::Error::source(&error).is_some());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn parser_and_serializer_observe_execution_service_cancellation() {
    use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy};
    use runmat_execution::cancellation::CancellationSource;
    use runmat_execution::CancellationReason;
    use std::num::NonZeroU64;

    let source = CancellationSource::new();
    let cancelled = source.clone();
    let budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
        source.token(),
        move || {
            cancelled.cancel(CancellationReason::User);
            Box::pin(async { Ok(()) })
        },
    );
    let mut cooperation = XmlCooperation::with_budget(budget, XmlOperation::XmlRead);
    let error = futures::executor::block_on(parse_xml_document_with_context(
        "<root><child/></root>",
        &mut cooperation,
    ))
    .unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:xmlread:Cancelled"));
    assert!(std::error::Error::source(&error)
        .expect("typed cancellation source")
        .downcast_ref::<crate::execution::ExecutionServiceError>()
        .is_some());

    let source = CancellationSource::new();
    let cancelled = source.clone();
    let budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
        source.token(),
        move || {
            cancelled.cancel(CancellationReason::User);
            Box::pin(async { Ok(()) })
        },
    );
    let mut cooperation = XmlCooperation::with_budget(budget, XmlOperation::XmlWrite);
    let error = futures::executor::block_on(serialize_xml_value(
        &Value::String("<root/>".into()),
        &mut cooperation,
    ))
    .unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:xmlwrite:Cancelled"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn execution_service_failure_is_not_relabelled_as_malformed_xml() {
    use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy, ExecutionServiceError};
    use runmat_execution::cancellation::CancellationSource;
    use std::num::NonZeroU64;

    let source = CancellationSource::new();
    let budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
        source.token(),
        || Box::pin(async { Err(ExecutionServiceError::WorkerLost("fixture".into())) }),
    );
    let mut cooperation = XmlCooperation::with_budget(budget, XmlOperation::XmlRead);
    let error =
        futures::executor::block_on(parse_xml_document_with_context("<root/>", &mut cooperation))
            .unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:xmlread:ExecutionFailed"));
    assert!(std::error::Error::source(&error)
        .expect("typed service source")
        .downcast_ref::<ExecutionServiceError>()
        .is_some());

    let wrapped = build_runtime_error("outer XML operation")
        .with_source(error)
        .build();
    assert!(cooperation.is_execution_error(&wrapped));
    let forged = build_runtime_error("ordinary malformed source")
        .with_identifier("RunMat:xmlread:ExecutionFailed")
        .build();
    assert!(!cooperation.is_execution_error(&forged));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn precancelled_xml_io_does_not_poll_provider_future() {
    use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy, ExecutionServiceError};
    use runmat_execution::cancellation::CancellationSource;
    use runmat_execution::CancellationReason;
    use std::cell::Cell;
    use std::rc::Rc;
    use std::task::Poll;

    let source = CancellationSource::new();
    source.cancel(CancellationReason::User);
    let polled = Rc::new(Cell::new(false));
    let marked = polled.clone();
    let budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::NeverYield,
        source.token(),
        || Box::pin(async { Ok(()) }),
    );
    let mut cooperation = XmlCooperation::with_budget(budget, XmlOperation::XmlRead);
    let error =
        futures::executor::block_on(cooperation.await_io(futures::future::poll_fn(move |_| {
            marked.set(true);
            Poll::<Result<(), std::io::Error>>::Pending
        })))
        .expect_err("pre-cancelled invocation");
    assert_eq!(error.identifier(), Some("RunMat:xmlread:Cancelled"));
    assert!(!polled.get());
    assert!(std::error::Error::source(&error)
        .expect("typed cancellation cause")
        .downcast_ref::<ExecutionServiceError>()
        .is_some());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn pending_xml_io_cancellation_drops_the_awaited_operation() {
    use crate::execution::{CpuCooperationBudget, CpuCooperationPolicy};
    use runmat_execution::cancellation::CancellationSource;
    use runmat_execution::CancellationReason;
    use std::cell::Cell;
    use std::rc::Rc;
    use std::task::Poll;

    let source = CancellationSource::new();
    let cancel_on_poll = source.clone();
    let dropped = Rc::new(Cell::new(false));
    struct PendingIo(Rc<Cell<bool>>);
    impl Drop for PendingIo {
        fn drop(&mut self) {
            self.0.set(true);
        }
    }
    let owner = PendingIo(dropped.clone());
    let budget = CpuCooperationBudget::with_test_turn(
        CpuCooperationPolicy::NeverYield,
        source.token(),
        || Box::pin(async { Ok(()) }),
    );
    let mut cooperation = XmlCooperation::with_budget(budget, XmlOperation::XmlRead);
    let error = futures::executor::block_on(cooperation.await_io(async move {
        let _owner = owner;
        futures::future::poll_fn(|_| {
            cancel_on_poll.cancel(CancellationReason::User);
            Poll::<Result<(), std::io::Error>>::Pending
        })
        .await
    }))
    .expect_err("pending I/O is cancelled");
    assert_eq!(error.identifier(), Some("RunMat:xmlread:Cancelled"));
    assert!(dropped.get(), "the pending I/O owner must be dropped");
}

#[test]
fn known_invalid_xml_controls_reject_before_filesystem_effects() {
    let _guard = fs_test_guard();
    let missing = Value::String("missing-structured-fixture.xml".into());
    let err = run(readstruct_builtin(
        None,
        vec![
            missing.clone(),
            Value::String("Bogus".into()),
            Value::String("x".into()),
        ],
    ))
    .unwrap_err();
    assert!(err.message().contains("unsupported option"));
    let err = run(xmlwrite_builtin(
        None,
        vec![
            Value::String("missing-structured-output.xml".into()),
            Value::Bool(true),
        ],
    ))
    .unwrap_err();
    assert!(err.message().contains("expected XML document"));
    let err = run(xmlread_builtin(None, vec![missing, Value::Bool(true)])).unwrap_err();
    assert!(err.message().contains("exactly one filename"));
}

#[test]
fn xmlwrite_accepts_scalar_string_array_and_rejects_nonscalar_text() {
    let one = runmat_value::StringArray::new(vec!["<root/>".into()], vec![1, 1]).unwrap();
    assert_eq!(
        run(xmlwrite_builtin(None, vec![Value::StringArray(one)])).unwrap(),
        char_value("<root/>")
    );
    let many = runmat_value::StringArray::new(vec!["a".into(), "b".into()], vec![1, 2]).unwrap();
    let err = run(xmlwrite_builtin(None, vec![Value::StringArray(many)])).unwrap_err();
    assert!(err.message().contains("document must be a string scalar"));
}

#[test]
fn paged_character_text_rejects_but_trailing_singletons_remain_valid() {
    let paged =
        runmat_value::CharArray::from_unicode_scalars_with_shape(vec!['a', 'b'], vec![1, 1, 2])
            .unwrap();
    let error = run(xmlread_builtin(None, vec![Value::CharArray(paged)])).unwrap_err();
    assert!(error.message().contains("filename must be a string scalar"));
    let trailing = runmat_value::CharArray::from_unicode_scalars_with_shape(
        "<root/>".chars().collect(),
        vec![1, 7, 1],
    )
    .unwrap();
    assert_eq!(
        run(xmlwrite_builtin(None, vec![Value::CharArray(trailing)])).unwrap(),
        char_value("<root/>")
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
    let value = run(xmlread_builtin(
        None,
        vec![Value::String(path_to_string(&path).into())],
    ))
    .unwrap();
    match value {
        Value::Object(object) => {
            assert_eq!(object.class_name().display_name(), "org.w3c.dom.Document");
            assert_eq!(
                object.property("DocumentElementName"),
                Some(&char_value("root"))
            );
            let Some(Value::Object(root)) = object.property("DocumentElement") else {
                panic!("expected DocumentElement object");
            };
            assert_eq!(root.property("TagName"), Some(&char_value("root")));
            let Some(Value::Struct(attrs)) = root.property("Attributes") else {
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
    let value = run(readstruct_builtin(
        None,
        vec![Value::String(path_to_string(&path).into())],
    ))
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
    let Value::Struct(first) = &items.data()[0] else {
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
    let value = run(readstruct_builtin(
        None,
        vec![
            Value::String(path_to_string(&path).into()),
            char_value("FileType"),
            char_value("xml"),
            char_value("AttributeSuffix"),
            char_value("_attr"),
            char_value("TextNodeName"),
            char_value("Body"),
        ],
    ))
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
    let mut root = runmat_value::UnboundedObjectBuilder::new(XML_ELEMENT_CLASS);
    root.insert("TagName".to_string(), char_value("root"));
    let mut attrs = runmat_value::UnboundedStructValueBuilder::new();
    attrs.insert("id", char_value("r1"));
    root.insert("Attributes".to_string(), Value::Struct(attrs.finish()));
    root.insert("TextContent".to_string(), char_value("hello & goodbye"));
    let root = root.finish();
    let xml = run(xmlwrite_builtin(None, vec![Value::Object(root.clone())])).unwrap();
    assert_eq!(
        xml,
        char_value("<root id=\"r1\">hello &amp; goodbye</root>")
    );

    let path = std::env::temp_dir().join("runmat_xmlwrite_test.xml");
    let bytes = run(xmlwrite_builtin(
        None,
        vec![
            Value::String(path_to_string(&path).into()),
            Value::Object(root),
        ],
    ))
    .unwrap();
    assert!(matches!(bytes, Value::Num(n) if n > 0.0));
    let text = std::fs::read_to_string(&path).unwrap();
    assert_eq!(text, "<root id=\"r1\">hello &amp; goodbye</root>");
    let _ = std::fs::remove_file(path);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn encoded_xmlread_xmlwrite_xmlread_roundtrip_labels_utf8_bytes_truthfully() {
    let _guard = fs_test_guard();
    let directory = "/xml";
    let provider = std::sync::Arc::new(runmat_filesystem::MemoryFsProvider::with_current_dir(
        directory,
    ));
    let _installed = runmat_filesystem::replace_provider(provider.clone());
    runmat_filesystem::set_current_dir(directory).expect("virtual cwd");
    let source = "/xml/source.xml";
    let output = "/xml/output.xml";

    let mut samples = Vec::new();
    let little = "<?xml version='1.0' encoding='UTF-16LE'?><root>温度</root>";
    let mut little_bytes = vec![0xFF, 0xFE];
    for unit in little.encode_utf16() {
        little_bytes.extend_from_slice(&unit.to_le_bytes());
    }
    samples.push((little_bytes, "温度"));
    let big = "<?xml version='1.0' encoding='UTF-16BE'?><root>é</root>";
    let mut big_bytes = vec![0xFE, 0xFF];
    for unit in big.encode_utf16() {
        big_bytes.extend_from_slice(&unit.to_be_bytes());
    }
    samples.push((big_bytes, "é"));
    samples.push((
        b"<?xml version='1.0' encoding='windows-1252'?><root>caf\xE9</root>".to_vec(),
        "café",
    ));

    for (bytes, expected_text) in samples {
        provider
            .write_project_path(source, &bytes)
            .expect("encoded source fixture");
        let document =
            run(xmlread_builtin(None, vec![Value::from(source)])).expect("decoded XML document");
        let Value::Object(original) = &document else {
            panic!("expected decoded XML document");
        };
        let Some(Value::String(runmat_value::StringElement::Text(original_text))) =
            original.property("Text")
        else {
            panic!("expected retained decoded source text");
        };
        assert!(original_text.contains(expected_text));
        assert!(original_text.contains("encoding='"));
        let count = run(xmlwrite_builtin(None, vec![Value::from(output), document]))
            .expect("UTF-8 XML write");
        let published = futures::executor::block_on(provider.read(std::path::Path::new(output)))
            .expect("published output through captured provider");
        assert_eq!(count, Value::Num(published.len() as f64));
        let utf8 = std::str::from_utf8(&published).expect("UTF-8 bytes");
        assert!(utf8.contains("encoding=\"UTF-8\""));
        assert_eq!(
            utf8.starts_with('\u{FEFF}'),
            bytes.starts_with(&[0xFF, 0xFE]) || bytes.starts_with(&[0xFE, 0xFF])
        );
        assert!(!utf8.contains("encoding='UTF-16"));
        assert!(!utf8.contains("encoding='windows-1252'"));
        let copied =
            run(xmlread_builtin(None, vec![Value::from(output)])).expect("round-trip XML document");
        let Value::Object(object) = copied else {
            panic!("expected XML document");
        };
        let Some(Value::Object(root)) = object.property("DocumentElement") else {
            panic!("expected round-trip root");
        };
        assert_eq!(
            root.property("TextContent"),
            Some(&char_value(expected_text))
        );
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn document_tree_mutation_wins_over_retained_source_text() {
    let _guard = fs_test_guard();
    let directory = "/xml";
    let path = "/xml/mutation.xml";
    let provider = std::sync::Arc::new(runmat_filesystem::MemoryFsProvider::with_current_dir(
        directory,
    ));
    let _installed = runmat_filesystem::replace_provider(provider.clone());
    runmat_filesystem::set_current_dir(directory).expect("virtual cwd");
    provider
        .write_project_path(
            path,
            b"<?xml version='1.0' encoding='UTF-8'?><root>old</root>",
        )
        .expect("source fixture");
    let Value::Object(mut document) =
        run(xmlread_builtin(None, vec![Value::from(path)])).expect("document")
    else {
        panic!("expected document");
    };
    let Some(Value::Object(element)) = document.property("DocumentElement") else {
        panic!("expected root element");
    };
    let mut element = element.clone();
    crate::execution::object::replacement::replace_property(
        None,
        &mut element,
        "TextContent",
        char_value("changed"),
        |message, source| {
            crate::context::host_allocation::host_allocation_runtime_error(
                None,
                "RunMat:xmlwrite:Test",
                message,
                source,
            )
        },
        |message| compat_error("xmlwrite", message),
    )
    .expect("replace original element generation");
    crate::execution::object::replacement::replace_property(
        None,
        &mut document,
        "DocumentElement",
        Value::Object(element),
        |message, source| {
            crate::context::host_allocation::host_allocation_runtime_error(
                None,
                "RunMat:xmlwrite:Test",
                message,
                source,
            )
        },
        |message| compat_error("xmlwrite", message),
    )
    .expect("publish replacement tree");
    let result = run(xmlwrite_builtin(None, vec![Value::Object(document)]))
        .expect("tree-owned serialization");
    assert_eq!(
        result,
        char_value("<?xml version=\"1.0\" encoding=\"UTF-8\"?><root>changed</root>")
    );
}

#[test]
fn xmlread_rejects_mismatched_tags() {
    let _guard = fs_test_guard();
    let path = std::env::temp_dir().join("runmat_xmlread_bad_test.xml");
    std::fs::write(&path, "<root><a></root>").unwrap();
    let err = run(xmlread_builtin(
        None,
        vec![Value::String(path_to_string(&path).into())],
    ))
    .unwrap_err();
    assert!(err.message().contains("malformed XML"));
    assert_eq!(err.identifier(), Some("RunMat:xmlread:MalformedXml"));
    assert!(std::error::Error::source(&err).is_some());
    let _ = std::fs::remove_file(path);
}
