use std::collections::HashMap;

use runmat_builtins::{ResolveContext, Type};
use runmat_value::{CellArray, ObjectInstance, Value};

use crate::{build_runtime_error, BuiltinResult};

fn functiontests_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::Object {
        class_name: Some(crate::testing::TEST_SUITE_CLASS.into()),
        shape: None,
    }
}

#[runmat_macros::runtime_builtin(
    name = "functiontests",
    category = "testing",
    summary = "Create a TestSuite from local function handles.",
    keywords = "functiontests,localfunctions,testing,suite",
    type_resolver(functiontests_type),
    builtin_path = "crate::builtins::testing::functiontests"
)]
fn functiontests_builtin(handles: Value) -> BuiltinResult<Value> {
    crate::testing::ensure_testing_classes();
    let Value::Cell(handles) = handles else {
        return Err(functiontests_error(
            "functiontests: expected the cell array returned by localfunctions",
        ));
    };
    build_function_suite(handles)
}

fn build_function_suite(handles: CellArray) -> BuiltinResult<Value> {
    let mut fixtures = HashMap::<String, Value>::new();
    let mut tests = Vec::<(String, Value)>::new();
    for handle in handles.data {
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

    let values = tests
        .into_iter()
        .map(|(name, handle)| {
            let mut object = ObjectInstance::new(crate::testing::TEST_SUITE_CLASS.into());
            object
                .properties
                .insert("Name".into(), Value::String(format!("{suite_name}/{name}")));
            object
                .properties
                .insert("ProcedureName".into(), Value::String(name));
            object
                .properties
                .insert("TestFile".into(), Value::String(source_path.clone()));
            object.properties.insert(
                "Tags".into(),
                Value::StringArray(
                    runmat_value::StringArray::new(Vec::new(), vec![1, 0])
                        .expect("empty tag row is valid"),
                ),
            );
            object
                .properties
                .insert("__runmat_procedure".into(), handle);
            for (fixture_name, fixture) in &fixtures {
                object
                    .properties
                    .insert(format!("__runmat_fixture_{fixture_name}"), fixture.clone());
            }
            Value::Object(object)
        })
        .collect::<Vec<_>>();
    crate::testing::object_array_or_scalar(crate::testing::TEST_SUITE_CLASS, values)
        .map_err(functiontests_error)
}

fn handle_name(handle: &Value) -> Option<String> {
    match handle {
        Value::FunctionHandle(name)
        | Value::ExternalFunctionHandle(name)
        | Value::MethodFunctionHandle(name) => Some(name.clone()),
        Value::BoundFunctionHandle { name, .. } => Some(name.clone()),
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
        let handles = [
            ("setup", 1),
            ("testFirst", 2),
            ("helper", 3),
            ("testSecond", 4),
            ("teardown", 5),
        ]
        .into_iter()
        .map(|(name, function)| Value::BoundFunctionHandle {
            name: name.into(),
            function,
        })
        .collect();
        let input = Value::Cell(CellArray::new(handles, 1, 5).unwrap());
        let value = functiontests_builtin(input).unwrap();
        let Value::ObjectArray(array) = value else {
            panic!("expected suite object array");
        };
        assert_eq!(array.class_name(), crate::testing::TEST_SUITE_CLASS);
        assert_eq!(array.shape(), &[1, 2]);
        for value in array.data() {
            let Value::Object(object) = value else {
                panic!("expected suite object");
            };
            assert!(object.properties.contains_key("__runmat_fixture_setup"));
            assert!(object.properties.contains_key("__runmat_fixture_teardown"));
        }
    }
}
