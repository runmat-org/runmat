use runmat_builtins::{ObjectInstance, Value};

use crate::BuiltinResult;

#[runmat_macros::runtime_builtin(
    name = "matlab.unittest.plugins.TestRunnerPlugin",
    category = "testing/plugins",
    summary = "Construct the base test-runner plugin object.",
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
        "matlab.unittest.plugins.TestRunnerPlugin".into(),
    )))
}
