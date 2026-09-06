use super::*;
use crate::builtins::io::repl_fs::REPL_FS_TEST_LOCK;
use runmat_value::{StringArray, Value};

fn run(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    futures::executor::block_on(isenv_builtin(args))
}

#[test]
fn scalar_and_string_array_queries_follow_environment_state() {
    let _guard = REPL_FS_TEST_LOCK.lock().unwrap();
    crate::builtins::common::env::set_var("RUNMAT_ISENV_TEST_A", "");
    crate::builtins::common::env::remove_var("RUNMAT_ISENV_TEST_B");
    assert_eq!(
        run(vec![Value::String("RUNMAT_ISENV_TEST_A".into())]).unwrap(),
        Value::Bool(true)
    );
    let names = StringArray::new(
        vec!["RUNMAT_ISENV_TEST_A".into(), "RUNMAT_ISENV_TEST_B".into()],
        vec![1, 2],
    )
    .unwrap();
    let Value::LogicalArray(values) = run(vec![Value::StringArray(names)]).unwrap() else {
        panic!("expected logical array");
    };
    assert_eq!(values.shape, vec![1, 2]);
    assert_eq!(values.data.as_slice(), &[1, 0]);
    crate::builtins::common::env::remove_var("RUNMAT_ISENV_TEST_A");
}

#[test]
fn invalid_names_reject_before_environment_access() {
    for value in [
        Value::Num(1.0),
        Value::GpuTensor(runmat_accelerate_api::GpuTensorHandle::new(
            vec![1, 1],
            99,
            2,
        )),
    ] {
        assert_eq!(
            run(vec![value]).unwrap_err().identifier(),
            Some("RunMat:isenv:InvalidName")
        );
    }
}
