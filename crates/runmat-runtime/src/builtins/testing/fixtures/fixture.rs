use std::sync::atomic::{AtomicU64, Ordering};

use runmat_value::{HandleRef, Value};

use crate::{build_runtime_error, BuiltinResult};

const FIXTURE_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("matlab.unittest.fixtures.Fixture");
const PATH_FIXTURE_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("matlab.unittest.fixtures.PathFixture");
const TEMPORARY_FOLDER_FIXTURE_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("matlab.unittest.fixtures.TemporaryFolderFixture");

static TEMPORARY_FOLDER_SEQUENCE: AtomicU64 = AtomicU64::new(1);

#[runmat_macros::runtime_builtin(
    name = "matlab.unittest.fixtures.PathFixture",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::testing::fixtures::fixture"
)]
async fn path_fixture(
    runtime: Option<crate::context::RuntimeContext>,
    path: Value,
) -> BuiltinResult<crate::execution::RootedValue> {
    crate::testing::ensure_testing_classes();
    let path = scalar_text("PathFixture", path)?;
    let create = crate::new_handle_object_builtin(PATH_FIXTURE_CLASS.into());
    let fixture = match &runtime {
        Some(runtime) => runtime.scope(create).await?,
        None => create.await?,
    };
    let fixture = crate::execution::RootedValue::in_optional_context(fixture, runtime.as_ref())?;
    set_handle_property(
        runtime.as_ref(),
        fixture.value(),
        "Name",
        Value::String("PathFixture".into()),
    )?;
    set_handle_property(
        runtime.as_ref(),
        fixture.value(),
        "Path",
        Value::string(path),
    )?;
    Ok(fixture)
}

#[runmat_macros::runtime_builtin(
    name = "matlab.unittest.fixtures.TemporaryFolderFixture",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::testing::fixtures::fixture"
)]
async fn temporary_folder_fixture(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<crate::execution::RootedValue> {
    crate::testing::ensure_testing_classes();
    if !args.is_empty() {
        return Err(testing_error(
            "TemporaryFolderFixture",
            "TemporaryFolderFixture: too many input arguments",
        ));
    }
    let create = crate::new_handle_object_builtin(TEMPORARY_FOLDER_FIXTURE_CLASS.into());
    let fixture = match &runtime {
        Some(runtime) => runtime.scope(create).await?,
        None => create.await?,
    };
    let fixture = crate::execution::RootedValue::in_optional_context(fixture, runtime.as_ref())?;
    set_handle_property(
        runtime.as_ref(),
        fixture.value(),
        "Name",
        Value::String("TemporaryFolderFixture".into()),
    )?;
    set_handle_property(
        runtime.as_ref(),
        fixture.value(),
        "Folder",
        Value::string(String::new()),
    )?;
    Ok(fixture)
}

#[runmat_macros::runtime_builtin(
    name = "matlab.unittest.TestCase.applyFixture",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::testing::fixtures::fixture"
)]
async fn apply_fixture(
    runtime: Option<crate::context::RuntimeContext>,
    receiver: Value,
    fixture: Value,
) -> BuiltinResult<crate::execution::RootedValue> {
    validate_test_case(&receiver)?;
    let handle = fixture_handle(&fixture)?;
    let kind = fixture_kind(&handle)?;
    crate::testing::ensure_runtime_teardown_available_in(runtime.as_ref())
        .map_err(|message| testing_error("applyFixture", message))?;
    let fixture = crate::execution::RootedValue::in_optional_context(fixture, runtime.as_ref())?;
    match kind {
        FixtureKind::Path => {
            let path = handle_text_property(&handle, "Path")?;
            crate::call_builtin_async("addpath", &[Value::string(path.clone())]).await?;
            crate::testing::record_runtime_teardown(
                runtime.as_ref(),
                Value::FunctionHandle("rmpath".into()),
                crate::call::BuiltinArguments::from_borrowed(
                    &[Value::string(path)],
                    runtime.as_ref(),
                )?,
            )
            .map_err(|source| {
                build_runtime_error(source.message())
                    .with_builtin("applyFixture")
                    .with_identifier("RunMat:Testing:InvalidFixture")
                    .with_source(source)
                    .build()
            })?;
        }
        FixtureKind::TemporaryFolder => {
            let folder = temporary_folder_path();
            runmat_filesystem::create_dir_all_async(&folder)
                .await
                .map_err(|error| {
                    testing_error(
                        "applyFixture",
                        format!("applyFixture: could not create '{folder}': {error}"),
                    )
                })?;
            set_handle_property(
                runtime.as_ref(),
                fixture.value(),
                "Folder",
                Value::string(folder.clone()),
            )?;
            crate::testing::record_runtime_teardown(
                runtime.as_ref(),
                Value::FunctionHandle("rmdir".into()),
                crate::call::BuiltinArguments::from_borrowed(
                    &[Value::string(folder), Value::String("s".into())],
                    runtime.as_ref(),
                )?,
            )
            .map_err(|source| {
                build_runtime_error(source.message())
                    .with_builtin("applyFixture")
                    .with_identifier("RunMat:Testing:InvalidFixture")
                    .with_source(source)
                    .build()
            })?;
        }
    }
    Ok(fixture)
}

enum FixtureKind {
    Path,
    TemporaryFolder,
}

fn fixture_kind(handle: &HandleRef) -> BuiltinResult<FixtureKind> {
    if crate::class_registry::is_class_or_subclass(&handle.class_name, &PATH_FIXTURE_CLASS.owned())
    {
        return Ok(FixtureKind::Path);
    }
    if crate::class_registry::is_class_or_subclass(
        &handle.class_name,
        &TEMPORARY_FOLDER_FIXTURE_CLASS.owned(),
    ) {
        return Ok(FixtureKind::TemporaryFolder);
    }
    let message = if crate::class_registry::is_class_or_subclass(
        &handle.class_name,
        &FIXTURE_CLASS.owned(),
    ) {
        format!(
            "applyFixture: fixture class '{}' does not have a setup implementation",
            handle.class_name
        )
    } else {
        format!(
            "applyFixture: '{}' is not a matlab.unittest.fixtures.Fixture",
            handle.class_name
        )
    };
    Err(testing_error("applyFixture", message))
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
        Value::Object(object) => Some(object.class_name()),
        Value::HandleObject(handle) => Some(&*handle.class_name),
        _ => None,
    };
    if class_name.is_some_and(|identity| {
        crate::class_registry::is_class_or_subclass(
            identity,
            &crate::testing::TEST_CASE_CLASS.owned(),
        )
    }) {
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
        Value::Object(object) => object.properties().get(name).and_then(|value| match value {
            Value::String(runmat_value::StringElement::Text(value)) => {
                Some(value.as_str().to_owned())
            }
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

fn set_handle_property(
    runtime: Option<&crate::context::RuntimeContext>,
    handle: &Value,
    name: &str,
    value: Value,
) -> BuiltinResult<()> {
    let value = crate::execution::RootedValue::in_optional_context(value, runtime)?;
    let Value::HandleObject(handle) = handle else {
        return Err(testing_error(
            "applyFixture",
            "testing fixture construction did not produce a handle object",
        ));
    };
    runmat_gc::gc_with_value_mut(&handle.target, |target| -> BuiltinResult<()> {
        match target {
            Value::Object(object) => {
                runmat_gc::gc_record_handle_write(&handle.target, value.value());
                crate::execution::object::replacement::replace_property(
                    runtime,
                    object,
                    name,
                    value.value().clone(),
                    |message, source| {
                        crate::context::host_allocation::host_allocation_runtime_error(
                            runtime,
                            "RunMat:Testing:InvalidFixture",
                            message,
                            source,
                        )
                    },
                    |message| testing_error("applyFixture", message),
                )?;
                Ok(())
            }
            _ => Err(testing_error(
                "applyFixture",
                "fixture handle does not reference an object",
            )),
        }
    })
    .map_err(|error| testing_error("applyFixture", format!("fixture mutation failed: {error}")))
    .and_then(std::convert::identity)
}

fn scalar_text(builtin: &'static str, value: Value) -> BuiltinResult<String> {
    match value {
        Value::String(runmat_value::StringElement::Text(value)) => {
            Ok(value.into_string_unbounded())
        }
        Value::CharArray(chars) if chars.rows <= 1 => {
            crate::builtins::common::text::decode_utf16_with_error(chars.data(), || {
                testing_error(builtin, format!("{builtin}: text is not valid UTF-16"))
            })
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    async fn temporary_folder_fixture(
        runtime: Option<crate::context::RuntimeContext>,
        args: Vec<Value>,
    ) -> BuiltinResult<crate::execution::RootedValue> {
        super::temporary_folder_fixture(
            runtime,
            crate::execution::RootedValues::new(args)
                .map_err(|error| error.into_runtime_error())?
                .into(),
        )
        .await
    }
    use futures::executor::block_on;

    #[test]
    fn constructors_create_concrete_handle_fixtures_without_applying_them() {
        let path = block_on(path_fixture(None, Value::String("not-yet-created".into()))).unwrap();
        let Value::HandleObject(path_handle) = path.value() else {
            panic!("PathFixture must be a handle")
        };
        assert_eq!(path_handle.class_name, PATH_FIXTURE_CLASS.owned());
        assert_eq!(
            handle_text_property(path_handle, "Path").unwrap(),
            "not-yet-created"
        );

        let temporary = block_on(temporary_folder_fixture(None, vec![])).unwrap();
        let Value::HandleObject(temp_handle) = temporary.value() else {
            panic!("TemporaryFolderFixture must be a handle")
        };
        assert_eq!(
            temp_handle.class_name,
            TEMPORARY_FOLDER_FIXTURE_CLASS.owned()
        );
        assert_eq!(handle_text_property(temp_handle, "Folder").unwrap(), "");
        assert_eq!(
            block_on(temporary_folder_fixture(None, vec![Value::Num(1.0)]))
                .unwrap_err()
                .identifier(),
            Some("RunMat:Testing:InvalidFixture")
        );
    }

    #[test]
    fn application_outside_test_rejects_before_path_or_filesystem_work() {
        let receiver = crate::testing::test_case_object("", false);
        let path = block_on(path_fixture(None, Value::String("not-yet-created".into()))).unwrap();
        let error =
            block_on(apply_fixture(None, receiver.clone(), path.value().clone())).unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:Testing:InvalidFixture"));
        assert!(error.to_string().contains("active test lifecycle"));

        let temporary = block_on(temporary_folder_fixture(None, vec![])).unwrap();
        let error = block_on(apply_fixture(None, receiver, temporary.value().clone())).unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:Testing:InvalidFixture"));
        assert!(error.to_string().contains("active test lifecycle"));
    }
}
