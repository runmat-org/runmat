use super::*;
use crate::builtins::io::repl_fs::REPL_FS_TEST_LOCK;
use runmat_value::{CharArray, StringArray, Value};

fn run(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    futures::executor::block_on(getenv_builtin(args))
}

#[test]
fn scalar_and_shaped_queries_preserve_public_result_forms() {
    let _guard = REPL_FS_TEST_LOCK.lock().unwrap();
    crate::builtins::common::env::set_var("RUNMAT_GETENV_TEST_A", "alpha");
    crate::builtins::common::env::set_var("RUNMAT_GETENV_TEST_B", "beta");
    assert_eq!(
        run(vec![Value::CharArray(CharArray::new_row(
            "RUNMAT_GETENV_TEST_A"
        ))])
        .unwrap(),
        Value::CharArray(CharArray::new_row("alpha"))
    );
    let names = StringArray::new(
        vec!["RUNMAT_GETENV_TEST_A".into(), "RUNMAT_GETENV_TEST_B".into()],
        vec![1, 2],
    )
    .unwrap();
    let Value::StringArray(values) = run(vec![Value::StringArray(names)]).unwrap() else {
        panic!("expected string array");
    };
    assert_eq!(values.shape, vec![1, 2]);
    assert_eq!(values.data, vec!["alpha", "beta"]);
    crate::builtins::common::env::remove_var("RUNMAT_GETENV_TEST_A");
    crate::builtins::common::env::remove_var("RUNMAT_GETENV_TEST_B");
}

#[test]
fn undefined_scalar_is_an_empty_character_row() {
    let _guard = REPL_FS_TEST_LOCK.lock().unwrap();
    crate::builtins::common::env::remove_var("RUNMAT_GETENV_TEST_MISSING");
    assert_eq!(
        run(vec![Value::String("RUNMAT_GETENV_TEST_MISSING".into())]).unwrap(),
        Value::CharArray(CharArray::new_row(""))
    );
}

#[test]
fn all_variables_are_a_typed_dictionary() {
    let _guard = REPL_FS_TEST_LOCK.lock().unwrap();
    crate::builtins::common::env::set_var("RUNMAT_GETENV_TEST_ALL", "visible");
    let Value::Object(object) = run(vec![]).unwrap() else {
        panic!("expected dictionary");
    };
    assert!(object.is_class(runmat_types::standard::DICTIONARY));
    let entries = super::super::dictionary::entries(&object).unwrap();
    assert!(entries.iter().any(|(key, value)| {
        key == &Value::String("RUNMAT_GETENV_TEST_ALL".into())
            && value == &Value::String("visible".into())
    }));
    crate::builtins::common::env::remove_var("RUNMAT_GETENV_TEST_ALL");
}

#[test]
fn invalid_resident_name_rejects_without_provider_access() {
    let value = Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle::new(
        vec![1, 1],
        99,
        1,
    ));
    let error = run(vec![value]).unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:getenv:InvalidName"));
    assert!(!error.message().contains("provider"));
}
