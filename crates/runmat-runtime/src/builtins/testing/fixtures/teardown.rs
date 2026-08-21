use runmat_value::Value;

use crate::{build_runtime_error, BuiltinResult};

#[runmat_macros::runtime_builtin(
    name = "matlab.unittest.TestCase.addTeardown",
    category = "testing/fixtures",
    summary = "Register a callback that runs when the active test scope is torn down.",
    builtin_path = "crate::builtins::testing::fixtures::teardown"
)]
fn add_teardown(receiver: Value, callback: Value, arguments: Vec<Value>) -> BuiltinResult<Value> {
    validate_test_case(&receiver)?;
    crate::testing::record_runtime_teardown(callback, arguments).map_err(|message| {
        build_runtime_error(message)
            .with_identifier("RunMat:Testing:AddTeardown")
            .with_builtin("addTeardown")
            .build()
    })?;
    empty_value()
}

fn validate_test_case(receiver: &Value) -> BuiltinResult<()> {
    crate::testing::ensure_testing_classes();
    let class_name = match receiver {
        Value::Object(object) => object.class_name.as_str(),
        Value::HandleObject(handle) => handle.class_name.as_str(),
        _ => "",
    };
    if crate::class_registry::is_class_or_subclass(class_name, crate::testing::TEST_CASE_CLASS) {
        Ok(())
    } else {
        Err(
            build_runtime_error("addTeardown: first argument must be a TestCase")
                .with_identifier("RunMat:Testing:InvalidTestCase")
                .with_builtin("addTeardown")
                .build(),
        )
    }
}

fn empty_value() -> BuiltinResult<Value> {
    runmat_value::Tensor::new(Vec::new(), vec![0, 0])
        .map(Value::Tensor)
        .map_err(|error| build_runtime_error(error).build())
}

#[cfg(test)]
mod tests {
    use runmat_test::context::TestExecutionContext;
    use runmat_test::descriptor::FixtureScope;
    use runmat_test::identity::{RunId, TestId};
    use runmat_test::lifecycle::{ExecutionPhase, FixtureScopeKey};
    use runmat_test::protocol::ProtocolLimits;

    use super::*;

    #[test]
    fn registers_portable_command_and_runtime_callback() {
        let guard = crate::testing::install_test_context(
            crate::testing::ActiveTestContext {
                execution: TestExecutionContext {
                    run_id: RunId::derive("revision", "run"),
                    test_id: TestId::derive(&runmat_test::identity::TestIdentityInput {
                        owner_identity: "owner",
                        relative_source_identity: "test.m",
                        semantic_scheme: "function",
                        semantic_item_path: "test",
                        parameter_identity: "",
                        fixture_identity: "",
                    }),
                    attempt: 1,
                    random_seed: 7,
                },
                phase: ExecutionPhase::TestBody,
                scope: FixtureScopeKey {
                    scope: FixtureScope::Test,
                    identity: "test".into(),
                },
            },
            ProtocolLimits::default(),
        );
        add_teardown(
            crate::testing::test_case_object("test", false),
            Value::FunctionHandle("cleanup".into()),
            vec![Value::Num(1.0)],
        )
        .unwrap();
        assert_eq!(guard.handle().commands().len(), 1);
        let callbacks = guard.handle().runtime_teardowns();
        assert_eq!(callbacks.len(), 1);
        assert_eq!(callbacks[0].arguments, vec![Value::Num(1.0)]);
    }
}
