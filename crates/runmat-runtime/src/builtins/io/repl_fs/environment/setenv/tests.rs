use super::*;
use crate::builtins::io::repl_fs::REPL_FS_TEST_LOCK;
use runmat_value::{IntValue, StringArray, Value};

fn run(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    futures::executor::block_on(setenv_builtin(args))
}

#[test]
fn scalar_and_exact_integer_values_update_the_portable_environment() {
    let _guard = REPL_FS_TEST_LOCK.lock().unwrap();
    for (suffix, value, expected) in [
        ("TEXT", Value::String("ready".into()), "ready"),
        (
            "INTEGER",
            Value::Int(IntValue::U64(u64::MAX)),
            "18446744073709551615",
        ),
    ] {
        let name = format!("RUNMAT_SETENV_TEST_{suffix}");
        assert_eq!(
            run(vec![Value::String(name.clone()), value]).unwrap(),
            Value::Num(0.0)
        );
        assert_eq!(crate::builtins::common::env::var(&name).unwrap(), expected);
        crate::builtins::common::env::remove_var(&name);
    }
}

#[test]
fn shaped_values_are_validated_before_batch_mutation() {
    let _guard = REPL_FS_TEST_LOCK.lock().unwrap();
    let names = StringArray::new(
        vec!["RUNMAT_SETENV_TEST_A".into(), "RUNMAT_SETENV_TEST_B".into()],
        vec![1, 2],
    )
    .unwrap();
    let values = StringArray::new(vec!["alpha".into(), "beta".into()], vec![1, 2]).unwrap();
    run(vec![Value::StringArray(names), Value::StringArray(values)]).unwrap();
    assert_eq!(
        crate::builtins::common::env::var("RUNMAT_SETENV_TEST_A").unwrap(),
        "alpha"
    );
    assert_eq!(
        crate::builtins::common::env::var("RUNMAT_SETENV_TEST_B").unwrap(),
        "beta"
    );
    crate::builtins::common::env::remove_var("RUNMAT_SETENV_TEST_A");
    crate::builtins::common::env::remove_var("RUNMAT_SETENV_TEST_B");
}

#[test]
fn invalid_later_name_does_not_apply_an_earlier_update() {
    let _guard = REPL_FS_TEST_LOCK.lock().unwrap();
    crate::builtins::common::env::remove_var("RUNMAT_SETENV_TEST_ATOMIC");
    let names = StringArray::new(
        vec!["RUNMAT_SETENV_TEST_ATOMIC".into(), "BAD=NAME".into()],
        vec![1, 2],
    )
    .unwrap();
    let values = StringArray::new(vec!["first".into(), "second".into()], vec![1, 2]).unwrap();
    assert_eq!(
        run(vec![Value::StringArray(names), Value::StringArray(values)]).unwrap(),
        Value::Num(1.0)
    );
    assert!(crate::builtins::common::env::var("RUNMAT_SETENV_TEST_ATOMIC").is_err());
}

#[test]
fn dictionary_updates_use_typed_dictionary_identity() {
    let _guard = REPL_FS_TEST_LOCK.lock().unwrap();
    let dictionary = super::super::dictionary::from_pairs(vec![(
        "RUNMAT_SETENV_TEST_DICT".into(),
        "value".into(),
    )])
    .unwrap();
    run(vec![dictionary]).unwrap();
    assert_eq!(
        crate::builtins::common::env::var("RUNMAT_SETENV_TEST_DICT").unwrap(),
        "value"
    );
    crate::builtins::common::env::remove_var("RUNMAT_SETENV_TEST_DICT");
}

#[test]
fn invalid_name_is_a_failure_result_without_partial_mutation() {
    let _guard = REPL_FS_TEST_LOCK.lock().unwrap();
    let result = execute::evaluate(&[
        Value::String("BAD=NAME".into()),
        Value::String("value".into()),
    ]);
    assert_eq!(result.unwrap().status(), 1);
}
