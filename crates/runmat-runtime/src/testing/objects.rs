use runmat_test::descriptor::{FixtureDescriptor, TestDescriptor};
use runmat_test::result::{Diagnostic, TestResult};
use runmat_value::{ObjectArray, ObjectInstance, UnboundedObjectBuilder, Value};

mod contextual;
pub use contextual::{
    object_array_or_scalar_in_context, object_from_fields_in_context,
    replace_object_properties_in_context, test_result_object_in_context,
    test_suite_object_in_context,
};

pub const TEST_CASE_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("matlab.unittest.TestCase");
pub const FUNCTION_TEST_CASE_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("matlab.unittest.FunctionTestCase");
pub const TEST_SUITE_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("matlab.unittest.TestSuite");
pub const TEST_RESULT_CLASS: runmat_types::StaticClassIdentity =
    runmat_types::StaticClassIdentity::new("matlab.unittest.TestResult");

pub fn test_case_object(name: impl Into<String>, function_based: bool) -> Value {
    let mut object = UnboundedObjectBuilder::new(if function_based {
        FUNCTION_TEST_CASE_CLASS
    } else {
        TEST_CASE_CLASS
    });
    object.insert("Name".into(), Value::string(name.into()));
    Value::Object(object.finish())
}

/// Retain an original name owner while admitting the final TestCase object.
/// Any nonempty name backing must already have its own allocation authority.
pub fn test_case_object_in_context(
    runtime: Option<&crate::context::RuntimeContext>,
    name: runmat_value::Utf8Text,
    function_based: bool,
) -> crate::BuiltinResult<crate::execution::RootedValue> {
    let class = if function_based {
        FUNCTION_TEST_CASE_CLASS
    } else {
        TEST_CASE_CLASS
    };
    let fields = [("Name", Value::String(name.into()))];
    crate::execution::object::construct_object_from_borrowed_fields(
        runtime,
        class.display_name(),
        fields.iter().map(|(name, value)| (*name, value)),
    )
}

pub fn test_suite_object(test: &TestDescriptor) -> Value {
    let mut object = UnboundedObjectBuilder::new(TEST_SUITE_CLASS);
    object.insert("Name".into(), Value::string(test.display_name.clone()));
    object.insert(
        "ProcedureName".into(),
        Value::string(
            test.procedure
                .display_name
                .rsplit('/')
                .next()
                .unwrap_or(test.procedure.display_name.as_str())
                .to_owned(),
        ),
    );
    object.insert(
        "TestFile".into(),
        Value::string(test.procedure.source.relative_path.clone()),
    );
    object.insert(
        "Tags".into(),
        Value::StringArray(
            runmat_value::StringArray::new(test.tags.clone(), vec![1, test.tags.len()])
                .expect("test tags have a valid row shape"),
        ),
    );
    object.insert(
        "__runmat_test_id".into(),
        Value::String(test.id.as_str().into()),
    );
    Value::Object(object.finish())
}

pub fn test_result_object(name: impl Into<String>, result: &TestResult) -> Value {
    let mut object = UnboundedObjectBuilder::new(TEST_RESULT_CLASS);
    object.insert("Name".into(), Value::string(name.into()));
    object.insert("Passed".into(), Value::Bool(result.state.is_success()));
    object.insert("Failed".into(), Value::Bool(result.state.failed));
    object.insert("Incomplete".into(), Value::Bool(result.state.incomplete));
    object.insert("Flaky".into(), Value::Bool(result.flaky));
    object.insert(
        "Details".into(),
        Value::string(
            result
                .attempts
                .last()
                .map(|attempt| attempt.output.clone())
                .unwrap_or_default(),
        ),
    );
    object.insert(
        "__runmat_test_id".into(),
        Value::String(result.test_id.as_str().into()),
    );
    Value::Object(object.finish())
}

pub fn fixture_object(fixture: &FixtureDescriptor) -> Value {
    let mut object = UnboundedObjectBuilder::new("matlab.unittest.fixtures.Fixture");
    object.insert("Name".into(), Value::string(fixture.display_name.clone()));
    object.insert(
        "__runmat_fixture_id".into(),
        Value::String(fixture.id.as_str().into()),
    );
    Value::Object(object.finish())
}

pub fn diagnostic_object(diagnostic: &Diagnostic) -> Value {
    let mut object = UnboundedObjectBuilder::new("matlab.unittest.diagnostics.Diagnostic");
    object.insert(
        "Identifier".into(),
        Value::string(diagnostic.identifier.clone()),
    );
    object.insert("Message".into(), Value::string(diagnostic.message.clone()));
    Value::Object(object.finish())
}

pub fn plugin_object(class_name: impl Into<runmat_types::ClassIdentity>) -> Value {
    Value::Object(ObjectInstance::new(class_name.into()))
}

pub fn object_array_or_scalar(
    class_name: impl Into<runmat_types::ClassIdentity>,
    values: Vec<Value>,
) -> Result<Value, String> {
    let class_name: runmat_types::ClassIdentity = class_name.into();
    if values.len() == 1 {
        Ok(values.into_iter().next().expect("one value"))
    } else {
        ObjectArray::row(class_name, values).map(Value::ObjectArray)
    }
}
