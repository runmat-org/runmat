use std::collections::HashMap;

use runmat_value::{CellArray, Value};

use crate::{build_runtime_error, BuiltinResult};

#[runmat_macros::runtime_builtin(
    name = "functiontests",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::testing::functiontests"
)]
fn functiontests_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    handles: Value,
) -> BuiltinResult<crate::execution::RootedValue> {
    crate::testing::ensure_testing_classes();
    let Value::Cell(handles) = handles else {
        return Err(functiontests_error(
            "functiontests: expected the cell array returned by localfunctions",
        ));
    };
    build_function_suite(runtime.as_ref(), handles)
}

fn build_function_suite(
    runtime: Option<&crate::context::RuntimeContext>,
    handles: CellArray,
) -> BuiltinResult<crate::execution::RootedValue> {
    let _source =
        crate::execution::RootedValue::in_optional_context(Value::Cell(handles.clone()), runtime)?;
    let mut fixtures = HashMap::<String, Value>::new();
    let mut tests = Vec::<(runmat_value::Utf8Text, Value)>::new();
    let mut classify = |handle: Value| -> BuiltinResult<()> {
        let Some(name) = handle_name(&handle) else {
            return Err(functiontests_error(
                "functiontests: every input element must be a named function handle",
            ));
        };
        match name.to_ascii_lowercase().as_str() {
            "setuponce" | "teardownonce" | "setup" | "teardown" => {
                fixtures.insert(name.to_ascii_lowercase(), handle);
            }
            name_lower if name_lower.starts_with("test") => tests.push((name, handle)),
            _ => {}
        }
        Ok(())
    };
    match handles.into_parts().0.try_into_owned() {
        Ok(owned) => {
            for handle in owned {
                classify(handle)?;
            }
        }
        Err(shared) => {
            for handle in shared.iter() {
                classify(clone_named_callable(handle)?)?;
            }
        }
    }

    let source = crate::source_context::current_source_info();
    let suite_name = source
        .as_ref()
        .and_then(|source| {
            std::path::Path::new(
                source
                    .fullpath_name
                    .as_deref()
                    .unwrap_or(source.name.as_ref()),
            )
            .file_stem()
        })
        .and_then(|stem| stem.to_str())
        .unwrap_or("functiontests")
        .to_string();
    let source_path = source
        .map(|source| {
            source
                .fullpath_name
                .as_deref()
                .unwrap_or(source.name.as_ref())
                .to_string()
        })
        .unwrap_or_default();

    let mut values = tests
        .into_iter()
        .map(
            |(name, handle)| -> BuiltinResult<crate::execution::RootedValue> {
                let fields = [
                    ("Name", Value::string(format!("{suite_name}/{name}"))),
                    (
                        "ProcedureName",
                        Value::String(runmat_value::StringElement::Text(name)),
                    ),
                    ("TestFile", Value::string(source_path.clone())),
                    (
                        "Tags",
                        Value::StringArray(
                            runmat_value::StringArray::new(Vec::new(), vec![1, 0])
                                .expect("empty tag row is valid"),
                        ),
                    ),
                    ("__runmat_procedure", handle),
                ];
                let fixture_names = [
                    ("setuponce", "__runmat_fixture_setuponce"),
                    ("teardownonce", "__runmat_fixture_teardownonce"),
                    ("setup", "__runmat_fixture_setup"),
                    ("teardown", "__runmat_fixture_teardown"),
                ];
                let properties = fields.iter().map(|(name, value)| (*name, value)).chain(
                    fixture_names.iter().filter_map(|(name, property)| {
                        fixtures.get(*name).map(|value| (*property, value))
                    }),
                );
                crate::execution::object::construct_object_from_borrowed_fields(
                    runtime,
                    crate::testing::TEST_SUITE_CLASS.display_name(),
                    properties,
                )
            },
        )
        .collect::<BuiltinResult<Vec<_>>>()?;
    if values.len() == 1 {
        return Ok(values.pop().expect("one rooted suite"));
    }
    let mut output = crate::execution::object_array::ObjectArrayConstruction::new(
        runtime,
        crate::execution::object_array::Class::Name(
            crate::testing::TEST_SUITE_CLASS.display_name(),
        ),
        values.len(),
        &[1, values.len()],
        std::iter::empty(),
        true,
    )?;
    for value in &values {
        output.push(value.value().clone())?;
    }
    output.finish()
}

fn clone_named_callable(handle: &Value) -> BuiltinResult<Value> {
    match handle {
        Value::FunctionHandle(value) => Ok(Value::FunctionHandle(value.clone())),
        Value::ExternalFunctionHandle(value) => Ok(Value::ExternalFunctionHandle(value.clone())),
        Value::MethodFunctionHandle(value) => Ok(Value::MethodFunctionHandle(value.clone())),
        Value::BoundFunctionHandle(value) => Ok(Value::BoundFunctionHandle(value.clone())),
        Value::Closure(value) => Ok(Value::Closure(value.clone())),
        _ => Err(functiontests_error(
            "functiontests: every input element must be a named function handle",
        )),
    }
}

fn handle_name(handle: &Value) -> Option<runmat_value::Utf8Text> {
    match handle {
        Value::FunctionHandle(name)
        | Value::ExternalFunctionHandle(name)
        | Value::MethodFunctionHandle(name) => Some(name.clone().into_text_owner()),
        Value::BoundFunctionHandle(handle) => Some(handle.name.clone()),
        Value::Closure(closure) => Some(closure.function_name.clone()),
        _ => None,
    }
}

fn functiontests_error(message: impl Into<String>) -> crate::RuntimeError {
    build_runtime_error(message)
        .with_identifier("RunMat:functiontests:InvalidInput")
        .with_builtin("functiontests")
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filters_helpers_and_attaches_conventional_fixtures() {
        let runtime = crate::context::RuntimeContext::with_frozen_environment(
            std::rc::Rc::new(crate::execution::RuntimeExecutionService::new()),
            std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            vec![("PWD".into(), "/not-the-selected-source".into())],
        )
        .expect("isolated environment");
        let _runtime = runtime.enter();
        let source = crate::source_context::lease_source_context_in_context(
            &runtime,
            "metadataSuite.m",
            Some(std::sync::Arc::from("/frozen/metadataSuite.m")),
            "function tests = metadataSuite()\ntests = functiontests(localfunctions);\nend\n",
        )
        .expect("source record, without opening a file");
        let _source = crate::source_context::replace_current_source_id_in_context(
            &runtime,
            Some(source.source_id()),
        );
        let handles = [
            ("setup", 1),
            ("testFirst", 2),
            ("helper", 3),
            ("testSecond", 4),
            ("teardown", 5),
        ]
        .into_iter()
        .map(|(name, function)| {
            Value::BoundFunctionHandle(runmat_value::BoundFunctionHandle::from_data(
                runmat_value::BoundFunctionHandleData {
                    name: name.into(),
                    function,
                    authority: runmat_value::BoundFunctionAuthority::unavailable(function),
                    source_role: runmat_types::FunctionSourceRole::Unknown,
                    source_file: None,
                },
            ))
        })
        .collect();
        let input = Value::Cell(CellArray::new(handles, 1, 5).unwrap());
        let value = functiontests_builtin(Some(runtime.clone()), input).unwrap();
        let Value::ObjectArray(array) = value.value() else {
            panic!("expected suite object array");
        };
        assert!(array.class_name().is(crate::testing::TEST_SUITE_CLASS));
        assert_eq!(array.shape(), &[1, 2]);
        for value in array.data() {
            let Value::Object(object) = value else {
                panic!("expected suite object");
            };
            assert!(object.property("__runmat_fixture_setup").is_some());
            assert!(object.property("__runmat_fixture_teardown").is_some());
            let Value::String(runmat_value::StringElement::Text(procedure)) =
                object.property("ProcedureName").unwrap()
            else {
                panic!("procedure name");
            };
            assert_eq!(
                object.property("Name"),
                Some(&Value::string(format!("metadataSuite/{procedure}")))
            );
            assert_eq!(
                object.property("TestFile"),
                Some(&Value::string("/frozen/metadataSuite.m"))
            );
        }
    }
}
