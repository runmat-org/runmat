#[cfg(not(target_arch = "wasm32"))]
use std::ffi::OsStr;

#[cfg(not(target_arch = "wasm32"))]
use runmat_core::{execute_text_request_for_testing, RunMatSession};
#[cfg(not(target_arch = "wasm32"))]
use runmat_gc::gc_test_context;
#[cfg(not(target_arch = "wasm32"))]
use runmat_python::{discover_python, PythonDiscoveryRequest};

#[cfg(not(target_arch = "wasm32"))]
mod started_project_fixture {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/support/started_python_project.rs"
    ));
}

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    if std::env::args_os().nth(1).as_deref()
        == Some(OsStr::new(
            runmat_process_host::HiddenMode::ExtensionHost.marker(),
        ))
    {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("Python extension host runtime")
            .block_on(runmat_runtime::foreign::run_extension_host())
            .expect("Python extension host");
        return;
    }
    python_namespace_keywords_and_persistent_code_execute_from_source();
    typed_arrays_and_object_attributes_cross_the_shared_foreign_boundary();
    started_project_fixture::run_started_python_project(
        runmat_python::PythonExecutionMode::OutOfProcess,
        |session, runtime| {
            let request = runmat_core::abi::ExecutionRequest::for_source(
                runmat_core::abi::SourceInput::Text {
                    name: "started-python-project.m".into(),
                    text: "projectAnswer = pyrun(\"import fixture; answer = fixture.answer()\", \"answer\"); projectAnswer".into(),
                },
                session.compat_mode(),
                runmat_core::abi::HostExecutionPolicy::default(),
                session.workspace_handle(),
            );
            let response = runtime.block_on(session.execute_request(request));
            let outcome = response
                .result
                .as_ref()
                .expect("invoke started project wheel");
            assert!(
                !outcome.diagnostics.iter().any(|diagnostic| {
                    diagnostic.severity == runmat_core::abi::DiagnosticSeverity::Error
                }),
                "started project invocation diagnostics: {:?}",
                outcome.diagnostics
            );
            let value = outcome.flow.durable_workspace_value().unwrap_or_else(|| {
                panic!(
                    "one project answer: flow={:?}, suspension={:?}",
                    outcome.flow, outcome.suspension
                )
            });
            let runmat_value::Value::Int(value) = value else {
                panic!("project answer must be an exact integer: {value:?}");
            };
            value.try_to_i32().expect("small exact project answer")
        },
        |_| (),
        |_, _| {},
    );
}

#[cfg(not(target_arch = "wasm32"))]
fn session() -> Option<RunMatSession> {
    discover_python(&PythonDiscoveryRequest::default()).ok()?;
    Some(gc_test_context(RunMatSession::new).expect("create RunMat session"))
}

#[cfg(not(target_arch = "wasm32"))]
fn python_namespace_keywords_and_persistent_code_execute_from_source() {
    let Some(mut session) = session() else {
        return;
    };
    let result = execute_text_request_for_testing(
        &mut session,
        r#"
closeEnough = py.math.isclose(1.0, 1.000001, pyargs("rel_tol", 0.00001));
answer = pyrun("answer = input_value + 2", "answer", "input_value", int32(40));
items = py.list({int32(10), int32(20)});
firstItem = items(1);
items(2) = int32(42);
secondItem = items(2);
mapped = py.list(py.map(@pythonCallback, {int64(40)}));
callbackItem = mapped(1);

function y = pythonCallback(x)
    y = py.operator.add(x, int64(2));
end
"#,
    )
    .expect("execute Python source");
    assert!(result.error.is_none(), "{:?}", result.error);

    let close_enough =
        execute_text_request_for_testing(&mut session, "closeEnough").expect("read keyword result");
    assert_eq!(close_enough.value, Some(runmat_value::Value::Bool(true)));
    let answer = execute_text_request_for_testing(&mut session, "answer")
        .expect("read persistent Python result");
    assert_eq!(
        answer.value.as_ref().map(|value| value.to_string()),
        Some("42".into())
    );
    for (name, expected) in [
        ("firstItem", "10"),
        ("secondItem", "42"),
        ("callbackItem", "42"),
    ] {
        let value = execute_text_request_for_testing(&mut session, name)
            .expect("read indexed Python result");
        assert_eq!(
            value.value.as_ref().map(|value| value.to_string()),
            Some(expected.into())
        );
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn typed_arrays_and_object_attributes_cross_the_shared_foreign_boundary() {
    let Some(mut session) = session() else {
        return;
    };
    let result = execute_text_request_for_testing(
        &mut session,
        r#"
values = uint64([9007199254740993, 18446744073709551615]);
incremented = py.numpy.add(values, uint64([1, 0]));
holder = py.types.SimpleNamespace();
holder.answer = int32(42);
fromObject = holder.answer;
pendingMetadata = py.map(@metadataNestedCallback, {int64(40)});
metadataObject = pyrun("Meta = type('Meta', (type,), {'__getattribute__': lambda cls, name, pending=pending: ('Metadata73' if callback(pending) == 42 else 'MetadataWrong') if name == '__name__' else type.__getattribute__(cls, name)}); Probe = Meta('Probe', (), {'__module__': 'fixture', 'answer': 73}); result = Probe()", "result", "callback", @metadataCallback, "pending", pendingMetadata);
metadataAnswer = metadataObject.answer;

function y = metadataCallback(pending)
    nested = py.list(pending);
    y = py.operator.add(nested(1), int64(1));
end

function y = metadataNestedCallback(x)
    y = py.operator.add(x, int64(1));
end
"#,
    )
    .expect("execute Python array and object source");
    assert!(result.error.is_none(), "{:?}", result.error);

    let class_name = execute_text_request_for_testing(&mut session, "class(incremented)")
        .expect("inspect returned array class");
    assert_eq!(
        class_name.value,
        Some(runmat_value::Value::String("uint64".into()))
    );
    let object_value = execute_text_request_for_testing(&mut session, "fromObject")
        .expect("read Python object attribute");
    assert_eq!(
        object_value.value.as_ref().map(|value| value.to_string()),
        Some("42".into())
    );
    let metadata_class = execute_text_request_for_testing(&mut session, "class(metadataObject)")
        .expect("inspect source-level metadata callback result");
    assert_eq!(
        metadata_class.value,
        Some(runmat_value::Value::String("fixture.Metadata73".into()))
    );
    let metadata_answer = execute_text_request_for_testing(&mut session, "metadataAnswer")
        .expect("read opaque object after nested metadata callback");
    let answer = match &metadata_answer.value {
        Some(runmat_value::Value::Int(answer)) => answer,
        other => panic!("metadata object answer must be an exact integer: {other:?}"),
    };
    assert_eq!(
        answer.try_to_i32().expect("small exact metadata answer"),
        73
    );
}
