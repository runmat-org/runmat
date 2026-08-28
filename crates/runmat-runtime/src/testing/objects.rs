use runmat_test::descriptor::{FixtureDescriptor, TestDescriptor};
use runmat_test::result::{Diagnostic, TestResult};
use runmat_value::{ObjectArray, ObjectInstance, Value};

pub const TEST_CASE_CLASS: &str = "matlab.unittest.TestCase";
pub const FUNCTION_TEST_CASE_CLASS: &str = "matlab.unittest.FunctionTestCase";
pub const TEST_SUITE_CLASS: &str = "matlab.unittest.TestSuite";
pub const TEST_RESULT_CLASS: &str = "matlab.unittest.TestResult";

pub fn test_case_object(name: impl Into<String>, function_based: bool) -> Value {
    let mut object = ObjectInstance::new(
        if function_based {
            FUNCTION_TEST_CASE_CLASS
        } else {
            TEST_CASE_CLASS
        }
        .into(),
    );
    object
        .properties
        .insert("Name".into(), Value::String(name.into()));
    Value::Object(object)
}

pub fn test_suite_object(test: &TestDescriptor) -> Value {
    let mut object = ObjectInstance::new(TEST_SUITE_CLASS.into());
    object
        .properties
        .insert("Name".into(), Value::String(test.display_name.clone()));
    object.properties.insert(
        "ProcedureName".into(),
        Value::String(
            test.procedure
                .display_name
                .rsplit('/')
                .next()
                .unwrap_or(test.procedure.display_name.as_str())
                .to_owned(),
        ),
    );
    object.properties.insert(
        "TestFile".into(),
        Value::String(test.procedure.source.relative_path.clone()),
    );
    object.properties.insert(
        "Tags".into(),
        Value::StringArray(
            runmat_value::StringArray::new(test.tags.clone(), vec![1, test.tags.len()])
                .expect("test tags have a valid row shape"),
        ),
    );
    object.properties.insert(
        "__runmat_test_id".into(),
        Value::String(test.id.as_str().into()),
    );
    Value::Object(object)
}

pub fn test_result_object(name: impl Into<String>, result: &TestResult) -> Value {
    let mut object = ObjectInstance::new(TEST_RESULT_CLASS.into());
    object
        .properties
        .insert("Name".into(), Value::String(name.into()));
    object
        .properties
        .insert("Passed".into(), Value::Bool(result.state.is_success()));
    object
        .properties
        .insert("Failed".into(), Value::Bool(result.state.failed));
    object
        .properties
        .insert("Incomplete".into(), Value::Bool(result.state.incomplete));
    object
        .properties
        .insert("Flaky".into(), Value::Bool(result.flaky));
    object.properties.insert(
        "Details".into(),
        Value::String(
            result
                .attempts
                .last()
                .map(|attempt| attempt.output.clone())
                .unwrap_or_default(),
        ),
    );
    object.properties.insert(
        "__runmat_test_id".into(),
        Value::String(result.test_id.as_str().into()),
    );
    Value::Object(object)
}

pub fn fixture_object(fixture: &FixtureDescriptor) -> Value {
    let mut object = ObjectInstance::new("matlab.unittest.fixtures.Fixture".into());
    object
        .properties
        .insert("Name".into(), Value::String(fixture.display_name.clone()));
    object.properties.insert(
        "__runmat_fixture_id".into(),
        Value::String(fixture.id.as_str().into()),
    );
    Value::Object(object)
}

pub fn diagnostic_object(diagnostic: &Diagnostic) -> Value {
    let mut object = ObjectInstance::new("matlab.unittest.diagnostics.Diagnostic".into());
    object.properties.insert(
        "Identifier".into(),
        Value::String(diagnostic.identifier.clone()),
    );
    object
        .properties
        .insert("Message".into(), Value::String(diagnostic.message.clone()));
    Value::Object(object)
}

pub fn plugin_object(class_name: impl Into<String>) -> Value {
    Value::Object(ObjectInstance::new(class_name.into()))
}

pub fn object_array_or_scalar(
    class_name: impl Into<String>,
    values: Vec<Value>,
) -> Result<Value, String> {
    if values.len() == 1 {
        Ok(values.into_iter().next().expect("one value"))
    } else {
        ObjectArray::row(class_name, values).map(Value::ObjectArray)
    }
}
