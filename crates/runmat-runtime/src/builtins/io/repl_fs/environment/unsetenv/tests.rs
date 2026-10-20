use super::*;
use runmat_value::{StringArray, Value};

fn run(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
    futures::executor::block_on(unsetenv_builtin(
        crate::builtins::common::test_support::builtin_arguments(args),
    ))
}

#[test]
fn removes_scalar_and_shaped_names_idempotently() {
    let _guard = runmat_filesystem::provider_override_lock();
    for name in ["RUNMAT_UNSETENV_TEST_A", "RUNMAT_UNSETENV_TEST_B"] {
        crate::builtins::common::env::set_var(name, "value");
    }
    let names = StringArray::new(
        vec![
            "RUNMAT_UNSETENV_TEST_A".into(),
            "RUNMAT_UNSETENV_TEST_B".into(),
        ],
        vec![2, 1],
    )
    .unwrap();
    assert_eq!(
        run(vec![Value::StringArray(names.clone())]).unwrap(),
        Value::Num(0.0)
    );
    assert_eq!(
        run(vec![Value::StringArray(names)]).unwrap(),
        Value::Num(0.0)
    );
    assert!(crate::builtins::common::env::var("RUNMAT_UNSETENV_TEST_A").is_err());
    assert!(crate::builtins::common::env::var("RUNMAT_UNSETENV_TEST_B").is_err());
}

#[test]
fn validates_the_whole_batch_before_removing_any_name() {
    let _guard = runmat_filesystem::provider_override_lock();
    crate::builtins::common::env::set_var("RUNMAT_UNSETENV_TEST_ATOMIC", "value");
    let names = StringArray::new(
        vec!["RUNMAT_UNSETENV_TEST_ATOMIC".into(), "BAD=NAME".into()],
        vec![1, 2],
    )
    .unwrap();
    assert_eq!(
        run(vec![Value::StringArray(names)]).unwrap(),
        Value::Num(1.0)
    );
    assert_eq!(
        crate::builtins::common::env::var("RUNMAT_UNSETENV_TEST_ATOMIC").unwrap(),
        "value"
    );
    crate::builtins::common::env::remove_var("RUNMAT_UNSETENV_TEST_ATOMIC");
}

#[test]
fn rejects_numeric_and_resident_names_without_provider_access() {
    for value in [
        Value::Num(1.0),
        Value::GpuTensor(
            crate::builtins::io::repl_fs::test_support::unregistered_numeric_allocation(vec![1, 1])
                .handle,
        ),
    ] {
        assert_eq!(
            run(vec![value]).unwrap_err().identifier(),
            Some("RunMat:unsetenv:InvalidName")
        );
    }
}
