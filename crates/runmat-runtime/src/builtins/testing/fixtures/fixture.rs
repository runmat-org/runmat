use std::sync::atomic::{AtomicU64, Ordering};

use runmat_value::{HandleRef, Value};

use crate::{build_runtime_error, BuiltinResult};

const FIXTURE_CLASS: &str = "matlab.unittest.fixtures.Fixture";
const PATH_FIXTURE_CLASS: &str = "matlab.unittest.fixtures.PathFixture";
const TEMPORARY_FOLDER_FIXTURE_CLASS: &str = "matlab.unittest.fixtures.TemporaryFolderFixture";

static TEMPORARY_FOLDER_SEQUENCE: AtomicU64 = AtomicU64::new(1);

#[runmat_macros::runtime_builtin(
    name = "matlab.unittest.fixtures.PathFixture",
    category = "testing/fixtures",
    summary = "Construct a fixture that adds a folder to the session search path.",
    builtin_path = "crate::builtins::testing::fixtures::fixture"
)]
async fn path_fixture(path: Value) -> BuiltinResult<Value> {
    crate::testing::ensure_testing_classes();
    let path = scalar_text("PathFixture", path)?;
    let fixture = crate::new_handle_object_builtin(PATH_FIXTURE_CLASS.into()).await?;
    set_handle_property(&fixture, "Name", Value::String("PathFixture".into()))?;
    set_handle_property(&fixture, "Path", Value::String(path))?;
    Ok(fixture)
}

#[runmat_macros::runtime_builtin(
    name = "matlab.unittest.fixtures.TemporaryFolderFixture",
    category = "testing/fixtures",
    summary = "Construct a fixture that creates an isolated temporary folder when applied.",
    builtin_path = "crate::builtins::testing::fixtures::fixture"
)]
async fn temporary_folder_fixture(args: Vec<Value>) -> BuiltinResult<Value> {
    crate::testing::ensure_testing_classes();
    if !args.is_empty() {
        return Err(testing_error(
            "TemporaryFolderFixture",
            "TemporaryFolderFixture: too many input arguments",
        ));
    }
    let fixture = crate::new_handle_object_builtin(TEMPORARY_FOLDER_FIXTURE_CLASS.into()).await?;
    set_handle_property(
        &fixture,
        "Name",
        Value::String("TemporaryFolderFixture".into()),
    )?;
    set_handle_property(&fixture, "Folder", Value::String(String::new()))?;
    Ok(fixture)
}

#[runmat_macros::runtime_builtin(
    name = "matlab.unittest.TestCase.applyFixture",
    category = "testing/fixtures",
    summary = "Apply a fixture to the active test and register its cleanup.",
    builtin_path = "crate::builtins::testing::fixtures::fixture"
)]
async fn apply_fixture(receiver: Value, fixture: Value) -> BuiltinResult<Value> {
    validate_test_case(&receiver)?;
    let handle = fixture_handle(&fixture)?;
    if crate::class_registry::is_class_or_subclass(&handle.class_name, PATH_FIXTURE_CLASS) {
        let path = handle_text_property(&handle, "Path")?;
        crate::call_builtin_async("addpath", &[Value::String(path.clone())]).await?;
        crate::testing::record_runtime_teardown(
            Value::FunctionHandle("rmpath".into()),
            vec![Value::String(path)],
        )
        .map_err(|message| testing_error("applyFixture", message))?;
    } else if crate::class_registry::is_class_or_subclass(
        &handle.class_name,
        TEMPORARY_FOLDER_FIXTURE_CLASS,
    ) {
        let folder = temporary_folder_path();
        runmat_filesystem::create_dir_all_async(&folder)
            .await
            .map_err(|error| {
                testing_error(
                    "applyFixture",
                    format!("applyFixture: could not create '{folder}': {error}"),
                )
            })?;
        set_handle_property(&fixture, "Folder", Value::String(folder.clone()))?;
        crate::testing::record_runtime_teardown(
            Value::FunctionHandle("rmdir".into()),
            vec![Value::String(folder), Value::String("s".into())],
        )
        .map_err(|message| testing_error("applyFixture", message))?;
    } else if !crate::class_registry::is_class_or_subclass(&handle.class_name, FIXTURE_CLASS) {
        return Err(testing_error(
            "applyFixture",
            format!(
                "applyFixture: '{}' is not a matlab.unittest.fixtures.Fixture",
                handle.class_name
            ),
        ));
    }
    Ok(fixture)
}

fn temporary_folder_path() -> String {
    let sequence = TEMPORARY_FOLDER_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let identity = crate::testing::active_test_context()
        .map(|context| {
            format!(
                "{}-{}",
                &context.execution.test_id.as_str()[..12],
                context.execution.attempt
            )
        })
        .unwrap_or_else(|| "interactive-0".into());
    format!(".runmat/test-tmp/{identity}-{sequence}")
}

fn fixture_handle(value: &Value) -> BuiltinResult<HandleRef> {
    match value {
        Value::HandleObject(handle) => Ok(handle.clone()),
        _ => Err(testing_error(
            "applyFixture",
            "applyFixture: fixture must be a handle fixture object",
        )),
    }
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
        Err(testing_error(
            "applyFixture",
            "applyFixture: first argument must be a TestCase",
        ))
    }
}

fn handle_text_property(handle: &HandleRef, name: &str) -> BuiltinResult<String> {
    runmat_gc::gc_with_value(&handle.target, |target| match target {
        Value::Object(object) => object.properties.get(name).and_then(|value| match value {
            Value::String(value) => Some(value.clone()),
            _ => None,
        }),
        _ => None,
    })
    .map_err(|error| testing_error("applyFixture", format!("fixture access failed: {error}")))?
    .ok_or_else(|| {
        testing_error(
            "applyFixture",
            format!("applyFixture: fixture property '{name}' must be scalar text"),
        )
    })
}

fn set_handle_property(handle: &Value, name: &str, value: Value) -> BuiltinResult<()> {
    let Value::HandleObject(handle) = handle else {
        return Err(testing_error(
            "applyFixture",
            "testing fixture construction did not produce a handle object",
        ));
    };
    runmat_gc::gc_with_value_mut(&handle.target, |target| match target {
        Value::Object(object) => {
            object.properties.insert(name.into(), value);
            true
        }
        _ => false,
    })
    .map_err(|error| testing_error("applyFixture", format!("fixture mutation failed: {error}")))
    .and_then(|updated| {
        if updated {
            Ok(())
        } else {
            Err(testing_error(
                "applyFixture",
                "fixture handle does not reference an object",
            ))
        }
    })
}

fn scalar_text(builtin: &'static str, value: Value) -> BuiltinResult<String> {
    match value {
        Value::String(value) => Ok(value),
        Value::CharArray(chars) if chars.rows <= 1 => Ok(chars.data.iter().collect()),
        _ => Err(testing_error(
            builtin,
            format!("{builtin}: expected scalar text"),
        )),
    }
}

fn testing_error(builtin: &'static str, message: impl Into<String>) -> crate::RuntimeError {
    build_runtime_error(message)
        .with_identifier("RunMat:Testing:InvalidFixture")
        .with_builtin(builtin)
        .build()
}
