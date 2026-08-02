use runmat_builtins::{ObjectInstance, ResolveContext, StringArray, Type, Value};

use crate::{build_runtime_error, BuiltinResult};

fn testsuite_type(_args: &[Type], _context: &ResolveContext) -> Type {
    Type::Object {
        class_name: Some(crate::testing::TEST_SUITE_CLASS.into()),
        shape: None,
    }
}

#[runmat_macros::runtime_builtin(
    name = "testsuite",
    category = "testing",
    summary = "Discover tests and return a homogeneous TestSuite array without executing them.",
    keywords = "testsuite,testing,discovery",
    type_resolver(testsuite_type),
    builtin_path = "crate::builtins::testing::testsuite"
)]
async fn testsuite(args: Vec<Value>) -> BuiltinResult<Value> {
    crate::testing::ensure_testing_classes();
    let resolved = crate::builtins::diagnostics::runtests::resolve_runtests_targets(args).await?;
    let values = resolved
        .targets
        .into_iter()
        .map(|case| {
            let mut object = ObjectInstance::new(crate::testing::TEST_SUITE_CLASS.into());
            object
                .properties
                .insert("Name".into(), Value::String(case.name));
            object
                .properties
                .insert("ProcedureName".into(), Value::String(String::new()));
            object.properties.insert(
                "TestFile".into(),
                Value::String(case.source_path.to_string_lossy().into_owned()),
            );
            object.properties.insert(
                "Tags".into(),
                Value::StringArray(
                    StringArray::new(Vec::new(), vec![1, 0]).expect("empty tag row is valid"),
                ),
            );
            object
                .properties
                .insert("__runmat_source".into(), Value::String(case.source));
            object.properties.insert(
                "__runmat_display_name".into(),
                Value::String(case.display_name),
            );
            Value::Object(object)
        })
        .collect();
    crate::testing::object_array_or_scalar(crate::testing::TEST_SUITE_CLASS, values).map_err(
        |error| {
            build_runtime_error(error)
                .with_identifier("RunMat:testsuite:InvalidSuite")
                .with_builtin("testsuite")
                .build()
        },
    )
}
