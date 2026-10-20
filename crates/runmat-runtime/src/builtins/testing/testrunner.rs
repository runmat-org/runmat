use runmat_value::Value;

use crate::{build_runtime_error, BuiltinResult};

const TEST_RUNNER_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("matlab.unittest.TestRunner");

#[runmat_macros::runtime_builtin(
    name = "matlab.unittest.TestRunner.withTextOutput",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::testing::testrunner"
)]
async fn with_text_output(
    runtime: Option<crate::context::RuntimeContext>,
    options: crate::call::BuiltinArguments,
) -> BuiltinResult<crate::execution::RootedValue> {
    crate::testing::ensure_testing_classes();
    if !options.len().is_multiple_of(2) {
        return Err(runner_error(
            "TestRunner.withTextOutput options must be name-value pairs",
        ));
    }
    if !options.is_empty() {
        return Err(
            build_runtime_error("TestRunner.withTextOutput options are not supported yet")
                .with_identifier("RunMat:Testing:UnsupportedRunnerOption")
                .with_builtin("matlab.unittest.TestRunner.withTextOutput")
                .build(),
        );
    }
    let create = crate::new_handle_object_builtin(TEST_RUNNER_CLASS.into());
    let runner = match &runtime {
        Some(runtime) => runtime.scope(create).await?,
        None => create.await?,
    };
    let runner = crate::execution::RootedValue::in_optional_context(runner, runtime.as_ref())?;
    let plugins = crate::execution::structure::cell::from_values(runtime.as_ref(), &[1, 0], &[])?;
    set_plugins(runtime.as_ref(), runner.value(), &plugins)?;
    Ok(runner)
}

#[runmat_macros::runtime_builtin(
    name = "matlab.unittest.TestRunner.addPlugin",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::testing::testrunner"
)]
fn add_plugin(
    runtime: Option<crate::context::RuntimeContext>,
    runner: Value,
    plugin: Value,
) -> BuiltinResult<crate::execution::RootedValue> {
    let runtime = runtime.as_ref();
    crate::testing::ensure_testing_classes();
    let plugin_class = match &plugin {
        Value::Object(object) => Some(object.class_name()),
        Value::HandleObject(handle) => Some(&*handle.class_name),
        _ => None,
    };
    if !plugin_class.is_some_and(|identity| {
        crate::class_registry::is_class_or_subclass(
            identity,
            &runmat_types::standard::UNIT_TEST_RUNNER_PLUGIN.owned(),
        )
    }) {
        return Err(runner_error("addPlugin requires a TestRunnerPlugin"));
    }
    let Value::HandleObject(handle) = &runner else {
        return Err(runner_error("addPlugin requires a TestRunner handle"));
    };
    if !crate::class_registry::is_class_or_subclass(&handle.class_name, &TEST_RUNNER_CLASS.owned())
    {
        return Err(runner_error("addPlugin requires a TestRunner handle"));
    }
    let existing = runmat_gc::gc_with_value(&handle.target, |target| match target {
        Value::Object(object) => match object.property("Plugins") {
            Some(Value::Cell(plugins))
                if plugins.rows == 1
                    && plugins.cols == plugins.data().len()
                    && plugins.shape() == [1, plugins.data().len()] =>
            {
                Ok(Some(plugins.clone()))
            }
            None => Ok(None),
            _ => Err(runner_error("TestRunner plugin storage is invalid")),
        },
        _ => Err(runner_error("TestRunner plugin storage is invalid")),
    })
    .map_err(|error| runner_error(format!("addPlugin failed: {error}")))??;
    let runner = crate::execution::RootedValue::in_optional_context(runner, runtime)?;
    let plugin = crate::execution::RootedValue::in_optional_context(plugin, runtime)?;
    let previous = existing.as_ref().map_or(&[][..], |cell| cell.data());
    let count = previous
        .len()
        .checked_add(1)
        .ok_or_else(|| runner_error("TestRunner plugin count overflow"))?;
    let plugins = crate::execution::structure::cell::from_borrowed_values(
        runtime,
        &[1, count],
        previous.iter().chain(std::iter::once(plugin.value())),
    )?;
    set_plugins(runtime, runner.value(), &plugins)?;
    Ok(runner)
}

#[runmat_macros::runtime_builtin(
    name = "matlab.unittest.TestRunner.run",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::testing::testrunner"
)]
async fn run(
    runtime: Option<crate::context::RuntimeContext>,
    runner: Value,
    suite: Value,
) -> BuiltinResult<crate::execution::RootedValue> {
    let Value::HandleObject(handle) = &runner else {
        return Err(runner_error("TestRunner.run requires a TestRunner handle"));
    };
    if !crate::class_registry::is_class_or_subclass(&handle.class_name, &TEST_RUNNER_CLASS.owned())
    {
        return Err(runner_error("TestRunner.run requires a TestRunner handle"));
    }
    let plugins = runmat_gc::gc_with_value(&handle.target, |target| {
        let Value::Object(object) = target else {
            return None;
        };
        match object.properties().get("Plugins") {
            Some(Value::Cell(plugins)) => Some(plugins.data_owner().clone()),
            _ => Some(runmat_value::OutputListStorage::unbounded(Vec::new())),
        }
    })
    .map_err(|error| runner_error(format!("failed to read runner plugins: {error}")))?
    .ok_or_else(|| runner_error("TestRunner plugin storage is invalid"))?;
    crate::testing::run_test_suite_in_context(runtime, suite, plugins)?.await
}

fn set_plugins(
    runtime: Option<&crate::context::RuntimeContext>,
    runner: &Value,
    plugins: &crate::execution::RootedValue,
) -> BuiltinResult<()> {
    let Value::HandleObject(handle) = runner else {
        return Err(runner_error(
            "TestRunner construction did not return a handle",
        ));
    };
    runmat_gc::gc_with_value_mut(&handle.target, |target| -> BuiltinResult<()> {
        match target {
            Value::Object(object) => {
                runmat_gc::gc_record_handle_write(&handle.target, plugins.value());
                crate::execution::object::replacement::replace_property(
                    runtime,
                    object,
                    "Plugins",
                    plugins.value().clone(),
                    |message, source| {
                        crate::context::host_allocation::host_allocation_runtime_error(
                            runtime,
                            "RunMat:Testing:InvalidRunner",
                            message,
                            source,
                        )
                    },
                    runner_error,
                )?;
                Ok(())
            }
            _ => Err(runner_error("TestRunner handle target is invalid")),
        }
    })
    .map_err(|error| runner_error(format!("TestRunner construction failed: {error}")))
    .and_then(std::convert::identity)
}

fn runner_error(message: impl Into<String>) -> crate::RuntimeError {
    build_runtime_error(message)
        .with_identifier("RunMat:Testing:InvalidRunner")
        .with_builtin("matlab.unittest.TestRunner")
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn with_text_output(
        runtime: Option<crate::context::RuntimeContext>,
        args: Vec<Value>,
    ) -> BuiltinResult<crate::execution::RootedValue> {
        super::with_text_output(
            runtime,
            crate::execution::RootedValues::new(args)
                .map_err(|error| error.into_runtime_error())?
                .into(),
        )
        .await
    }
    use futures::executor::block_on;

    #[test]
    fn text_runner_rejects_options_that_it_cannot_apply() {
        let odd = futures::executor::block_on(with_text_output(
            None,
            vec![Value::String("OutputDetail".into())],
        ))
        .unwrap_err();
        assert_eq!(odd.identifier(), Some("RunMat:Testing:InvalidRunner"));

        let ignored = futures::executor::block_on(with_text_output(
            None,
            vec![
                Value::String("OutputDetail".into()),
                Value::String("Verbose".into()),
            ],
        ))
        .unwrap_err();
        assert_eq!(
            ignored.identifier(),
            Some("RunMat:Testing:UnsupportedRunnerOption")
        );
    }

    #[test]
    fn add_plugin_updates_the_public_cell_shape() {
        let runner = block_on(with_text_output(None, Vec::new())).unwrap();
        let plugin = Value::Object(runmat_value::ObjectInstance::new(
            runmat_types::standard::UNIT_TEST_RUNNER_PLUGIN.owned(),
        ));

        let updated = add_plugin(None, runner.value().clone(), plugin).unwrap();
        assert_eq!(updated.value(), runner.value());
        let plugins = block_on(crate::object::resolve::load_member(
            runner.value().clone(),
            "Plugins".into(),
            false,
            None,
        ))
        .unwrap();
        let Value::Cell(plugins) = plugins.value() else {
            panic!("runner.Plugins should be a cell array");
        };
        assert_eq!(plugins.shape, [1, 1]);
        assert_eq!(plugins.rows, 1);
        assert_eq!(plugins.cols, 1);
        assert_eq!(plugins.data().len(), 1);
    }
}
