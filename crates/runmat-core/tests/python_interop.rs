#![cfg(not(target_arch = "wasm32"))]

use runmat_core::{execute_text_request_for_testing, RunMatSession};
use runmat_gc::gc_test_context;
use runmat_python::{discover_python, PythonDiscoveryRequest};

fn session() -> Option<RunMatSession> {
    discover_python(&PythonDiscoveryRequest::default()).ok()?;
    Some(gc_test_context(RunMatSession::new).expect("create RunMat session"))
}

#[test]
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
        answer.value.map(|value| value.to_string()),
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
            value.value.map(|value| value.to_string()),
            Some(expected.into())
        );
    }
}

#[test]
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
        object_value.value.map(|value| value.to_string()),
        Some("42".into())
    );
}
