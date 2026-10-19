use runmat_value::{ObjectInstance, Value};

use crate::BuiltinResult;

#[runmat_macros::runtime_builtin(
    name = "matlab.unittest.plugins.TestRunnerPlugin",
    binding_variant = "default",
    builtin_path = "crate::builtins::testing::plugins"
)]
fn test_runner_plugin(args: Vec<Value>) -> BuiltinResult<Value> {
    crate::testing::ensure_testing_classes();
    if !args.is_empty() {
        return Err(
            crate::build_runtime_error("TestRunnerPlugin: too many input arguments")
                .with_identifier("RunMat:Testing:InvalidPlugin")
                .with_builtin("TestRunnerPlugin")
                .build(),
        );
    }
    Ok(Value::Object(ObjectInstance::new(
        runmat_types::standard::UNIT_TEST_RUNNER_PLUGIN,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_plugin_has_concrete_class_and_rejects_inputs() {
        let plugin = test_runner_plugin(vec![]).unwrap();
        assert!(
            matches!(plugin, Value::Object(object) if object.class_name.is(runmat_types::standard::UNIT_TEST_RUNNER_PLUGIN))
        );
        let error = test_runner_plugin(vec![Value::Bool(true)]).unwrap_err();
        assert_eq!(
            error.identifier.as_deref(),
            Some("RunMat:Testing:InvalidPlugin")
        );
    }
}
