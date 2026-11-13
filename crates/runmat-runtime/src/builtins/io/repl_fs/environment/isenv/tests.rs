use super::*;
use runmat_value::{StringArray, Value};

fn run(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    futures::executor::block_on(isenv_builtin(
        crate::builtins::common::test_support::builtin_arguments(args),
    ))
}

#[test]
fn direct_requested_output_boundary_rejects_before_name_parsing() {
    let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    assert_eq!(
        run(vec![Value::Num(1.0)]).unwrap_err().identifier(),
        Some("RunMat:isenv:TooManyOutputs")
    );
}

#[test]
fn scalar_and_string_array_queries_follow_environment_state() {
    let _guard = runmat_filesystem::provider_override_lock();
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
        Value::GpuTensor(
            crate::builtins::io::repl_fs::test_support::unregistered_numeric_allocation(vec![1, 1])
                .handle,
        ),
    ] {
        assert_eq!(
            run(vec![value]).unwrap_err().identifier(),
            Some("RunMat:isenv:InvalidName")
        );
    }
}
