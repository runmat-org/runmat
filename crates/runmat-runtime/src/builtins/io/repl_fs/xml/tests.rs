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
    let paged = runmat_value::CharArray::new_with_shape(vec!['a', 'b'], vec![1, 1, 2]).unwrap();
    let error = run(xmlread_builtin(None, vec![Value::CharArray(paged)])).unwrap_err();
    assert!(error.message().contains("filename must be a string scalar"));
    let trailing =
        runmat_value::CharArray::new_with_shape("<root/>".chars().collect(), vec![1, 7, 1])
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
        vec![Value::String(path_to_string(&path))],
    ))
    .unwrap();
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
    let value = run(readstruct_builtin(
        None,
        vec![Value::String(path_to_string(&path))],
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
    let value = run(readstruct_builtin(
        None,
        vec![
            Value::String(path_to_string(&path)),
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
    let mut root = ObjectInstance::new("org.w3c.dom.Element".to_string());
    root.properties
        .insert("TagName".to_string(), char_value("root"));
    let mut attrs = StructValue::new();
    attrs.insert("id", char_value("r1"));
    root.properties
        .insert("Attributes".to_string(), Value::Struct(attrs));
    root.properties
        .insert("TextContent".to_string(), char_value("hello & goodbye"));
    let xml = run(xmlwrite_builtin(None, vec![Value::Object(root.clone())])).unwrap();
    assert_eq!(
        xml,
        char_value("<root id=\"r1\">hello &amp; goodbye</root>")
    );

    let path = std::env::temp_dir().join("runmat_xmlwrite_test.xml");
    let bytes = run(xmlwrite_builtin(
        None,
        vec![Value::String(path_to_string(&path)), Value::Object(root)],
    ))
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
    let err = run(xmlread_builtin(
        None,
        vec![Value::String(path_to_string(&path))],
    ))
    .unwrap_err();
    assert!(err.message().contains("malformed XML"));
    assert_eq!(err.identifier(), Some("RunMat:xmlread:MalformedXml"));
    assert!(std::error::Error::source(&err).is_some());
    let _ = std::fs::remove_file(path);
}
