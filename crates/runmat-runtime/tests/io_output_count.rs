//! Dispatcher tests link one runtime library, unlike runtime's unit executable
//! whose acceleration dev-dependency also links a non-test runtime copy.
use futures::executor::block_on;
use runmat_runtime::call_builtin_async_with_outputs;
use runmat_value::Value;

#[test]
fn invalid_output_counts_reject_before_reading_names() {
    for name in ["getenv", "isenv"] {
        let error =
            block_on(call_builtin_async_with_outputs(name, &[Value::Num(1.0)], 2)).unwrap_err();
        assert_eq!(
            error.identifier(),
            Some(format!("RunMat:{name}:TooManyOutputs").as_str())
        );
    }
}

#[test]
fn excessive_outputs_are_rejected_before_path_parsing() {
    for name in ["isfile", "isfolder", "matlabroot"] {
        let error =
            block_on(call_builtin_async_with_outputs(name, &[Value::Num(1.0)], 2)).unwrap_err();
        assert_eq!(
            error.identifier(),
            Some(format!("RunMat:{name}:TooManyOutputs").as_str())
        );
    }
}

#[test]
fn fixed_output_counts_reject_without_selecting_a_path() {
    for name in ["tempdir", "tempname", "pathsep"] {
        let error = block_on(call_builtin_async_with_outputs(name, &[], 2)).unwrap_err();
        assert_eq!(
            error.identifier(),
            Some(format!("RunMat:{name}:TooManyOutputs").as_str())
        );
    }
}
