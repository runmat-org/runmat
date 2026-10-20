#[cfg(test)]
use runmat_value::Value;

use crate::BuiltinResult;

#[runmat_macros::runtime_builtin(
    name = "matlab.unittest.plugins.TestRunnerPlugin",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::testing::plugins"
)]
fn test_runner_plugin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<crate::execution::RootedValue> {
    crate::testing::ensure_testing_classes();
    if !args.is_empty() {
        return Err(
            crate::build_runtime_error("TestRunnerPlugin: too many input arguments")
                .with_identifier("RunMat:Testing:InvalidPlugin")
                .with_builtin("TestRunnerPlugin")
                .build(),
        );
    }
    crate::execution::object::construct_object_from_borrowed_fields(
        runtime.as_ref(),
        runmat_types::standard::UNIT_TEST_RUNNER_PLUGIN.display_name(),
        std::iter::empty(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_plugin_has_concrete_class_and_rejects_inputs() {
        let plugin = test_runner_plugin(
            None,
            crate::execution::RootedValues::new(vec![]).unwrap().into(),
        )
        .unwrap();
        assert!(
            matches!(plugin.value(), Value::Object(object) if object.class_name().is(runmat_types::standard::UNIT_TEST_RUNNER_PLUGIN))
        );
        let error = test_runner_plugin(
            None,
            crate::execution::RootedValues::new(vec![Value::Bool(true)])
                .unwrap()
                .into(),
        )
        .unwrap_err();
        assert_eq!(
            error.identifier.as_deref(),
            Some("RunMat:Testing:InvalidPlugin")
        );
    }
}
