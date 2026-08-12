use runmat_builtins::{ResolveContext, Type};
use runmat_value::Value;

use crate::{build_runtime_error, BuiltinResult};

const TEST_RUNNER_CLASS: &str = "matlab.unittest.TestRunner";

fn runner_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::Object {
        class_name: Some(TEST_RUNNER_CLASS.into()),
        shape: Some(vec![Some(1), Some(1)]),
    }
}

#[runmat_macros::runtime_builtin(
    name = "matlab.unittest.TestRunner.withTextOutput",
    category = "testing",
    summary = "Construct a test runner that emits text progress through the host reporter.",
    type_resolver(runner_type),
    builtin_path = "crate::builtins::testing::testrunner"
)]
async fn with_text_output(options: Vec<Value>) -> BuiltinResult<Value> {
    crate::testing::ensure_testing_classes();
    if !options.len().is_multiple_of(2) {
        return Err(runner_error(
            "TestRunner.withTextOutput options must be name-value pairs",
        ));
    }
    let runner = crate::new_handle_object_builtin(TEST_RUNNER_CLASS.into()).await?;
    set_plugins(&runner, Vec::new())?;
    Ok(runner)
}

#[runmat_macros::runtime_builtin(
    name = "matlab.unittest.TestRunner.addPlugin",
    category = "testing/plugins",
    summary = "Attach a compatible TestRunnerPlugin to a runner.",
    builtin_path = "crate::builtins::testing::testrunner"
)]
fn add_plugin(runner: Value, plugin: Value) -> BuiltinResult<Value> {
    crate::testing::ensure_testing_classes();
    let plugin_class = match &plugin {
        Value::Object(object) => object.class_name.as_str(),
        Value::HandleObject(handle) => handle.class_name.as_str(),
        _ => "",
    };
    if !runmat_builtins::is_class_or_subclass(
        plugin_class,
        "matlab.unittest.plugins.TestRunnerPlugin",
    ) {
        return Err(runner_error("addPlugin requires a TestRunnerPlugin"));
    }
    let Value::HandleObject(handle) = &runner else {
        return Err(runner_error("addPlugin requires a TestRunner handle"));
    };
    if !runmat_builtins::is_class_or_subclass(&handle.class_name, TEST_RUNNER_CLASS) {
        return Err(runner_error("addPlugin requires a TestRunner handle"));
    }
    runmat_gc::gc_with_value_mut(&handle.target, |target| match target {
        Value::Object(object) => {
            let plugins = object
                .properties
                .entry("Plugins".into())
                .or_insert_with(|| {
                    Value::Cell(runmat_value::CellArray::new(Vec::new(), 1, 0).unwrap())
                });
            if let Value::Cell(plugins) = plugins {
                plugins.data.push(plugin);
                plugins.cols += 1;
                true
            } else {
                false
            }
        }
        _ => false,
    })
    .map_err(|error| runner_error(format!("addPlugin failed: {error}")))
    .and_then(|updated| {
        if updated {
            Ok(runner)
        } else {
            Err(runner_error("TestRunner plugin storage is invalid"))
        }
    })
}

#[runmat_macros::runtime_builtin(
    name = "matlab.unittest.TestRunner.run",
    category = "testing",
    summary = "Run a suite through the active Core test executor.",
    builtin_path = "crate::builtins::testing::testrunner"
)]
async fn run(runner: Value, suite: Value) -> BuiltinResult<Value> {
    let Value::HandleObject(handle) = &runner else {
        return Err(runner_error("TestRunner.run requires a TestRunner handle"));
    };
    if !runmat_builtins::is_class_or_subclass(&handle.class_name, TEST_RUNNER_CLASS) {
        return Err(runner_error("TestRunner.run requires a TestRunner handle"));
    }
    let plugins = runmat_gc::gc_with_value(&handle.target, |target| {
        let Value::Object(object) = target else {
            return None;
        };
        match object.properties.get("Plugins") {
            Some(Value::Cell(plugins)) => Some(plugins.data.clone()),
            _ => Some(Vec::new()),
        }
    })
    .map_err(|error| runner_error(format!("failed to read runner plugins: {error}")))?
    .ok_or_else(|| runner_error("TestRunner plugin storage is invalid"))?;
    crate::testing::run_test_suite(suite, plugins).await
}

fn set_plugins(runner: &Value, plugins: Vec<Value>) -> BuiltinResult<()> {
    let Value::HandleObject(handle) = runner else {
        return Err(runner_error(
            "TestRunner construction did not return a handle",
        ));
    };
    let count = plugins.len();
    runmat_gc::gc_with_value_mut(&handle.target, |target| match target {
        Value::Object(object) => {
            object.properties.insert(
                "Plugins".into(),
                Value::Cell(
                    runmat_value::CellArray::new(plugins, 1, count)
                        .expect("plugin row shape is valid"),
                ),
            );
            true
        }
        _ => false,
    })
    .map_err(|error| runner_error(format!("TestRunner construction failed: {error}")))
    .and_then(|updated| {
        if updated {
            Ok(())
        } else {
            Err(runner_error("TestRunner handle target is invalid"))
        }
    })
}

fn runner_error(message: impl Into<String>) -> crate::RuntimeError {
    build_runtime_error(message)
        .with_identifier("RunMat:Testing:InvalidRunner")
        .with_builtin("matlab.unittest.TestRunner")
        .build()
}
