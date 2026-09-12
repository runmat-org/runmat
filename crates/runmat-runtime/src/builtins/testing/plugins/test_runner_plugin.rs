use runmat_builtins::{ResolveContext, Type};
use runmat_value::{ObjectInstance, Value};

use crate::BuiltinResult;

#[runmat_macros::runtime_builtin(
    name = "matlab.unittest.plugins.TestRunnerPlugin",
    category = "testing/plugins",
    summary = "Construct the base test-runner plugin object.",
    type_resolver(test_runner_plugin_type),
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

fn test_runner_plugin_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::Object {
        class_name: Some(runmat_types::standard::UNIT_TEST_RUNNER_PLUGIN.into()),
        shape: Some(vec![Some(1), Some(1)]),
    }
}
