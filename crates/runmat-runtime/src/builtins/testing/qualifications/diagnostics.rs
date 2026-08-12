use runmat_test::context::TestCommand;
use runmat_test::lifecycle::ExecutionPhase;
use runmat_test::result::{Diagnostic, DiagnosticSeverity};
use runmat_value::{ObjectInstance, Value};

use crate::BuiltinResult;

#[runmat_macros::runtime_builtin(
    name = "matlab.unittest.TestCase",
    category = "testing",
    summary = "Construct a MATLAB-compatible test case object.",
    builtin_path = "crate::builtins::testing::qualifications::diagnostics"
)]
fn test_case_constructor(args: Vec<Value>) -> BuiltinResult<Value> {
    crate::testing::ensure_testing_classes();
    if !args.is_empty() {
        return Err(crate::build_runtime_error(
            "matlab.unittest.TestCase: too many input arguments",
        )
        .with_identifier("RunMat:Testing:InvalidInput")
        .with_builtin("matlab.unittest.TestCase")
        .build());
    }
    Ok(crate::testing::test_case_object("", false))
}

#[runmat_macros::runtime_builtin(
    name = "matlab.unittest.diagnostics.StringDiagnostic",
    category = "testing/diagnostics",
    summary = "Construct a string-backed test diagnostic.",
    builtin_path = "crate::builtins::testing::qualifications::diagnostics"
)]
fn string_diagnostic(value: Value) -> BuiltinResult<Value> {
    crate::testing::ensure_testing_classes();
    let message = match value {
        Value::String(message) => message,
        Value::CharArray(chars) if chars.rows <= 1 => chars.data.iter().collect(),
        _ => {
            return Err(
                crate::build_runtime_error("StringDiagnostic: expected scalar text")
                    .with_identifier("RunMat:Testing:InvalidDiagnostic")
                    .with_builtin("StringDiagnostic")
                    .build(),
            )
        }
    };
    let mut object = ObjectInstance::new("matlab.unittest.diagnostics.Diagnostic".to_string());
    object
        .properties
        .insert("Identifier".into(), Value::String(String::new()));
    object
        .properties
        .insert("Message".into(), Value::String(message));
    Ok(Value::Object(object))
}

#[runmat_macros::runtime_builtin(
    name = "matlab.unittest.TestCase.log",
    category = "testing/diagnostics",
    summary = "Record a diagnostic message in the active test result.",
    builtin_path = "crate::builtins::testing::qualifications::diagnostics"
)]
fn log(receiver: Value, level: Value, diagnostic: Value) -> BuiltinResult<Value> {
    validate_test_case(&receiver)?;
    let severity = log_severity(level)?;
    let message = diagnostic_message(diagnostic)?;
    let phase = crate::testing::active_test_context()
        .map(|context| context.phase)
        .unwrap_or(ExecutionPhase::TestBody);
    crate::testing::record_test_command(TestCommand::RecordDiagnostic {
        diagnostic: Diagnostic {
            identifier: "RunMat:Test:Log".into(),
            message,
            severity,
            phase,
            source: None,
            details: Vec::new(),
        },
    })
    .map_err(testing_diagnostic_error)?;
    empty_value()
}

fn validate_test_case(receiver: &Value) -> BuiltinResult<()> {
    crate::testing::ensure_testing_classes();
    let class_name = match receiver {
        Value::Object(object) => object.class_name.as_str(),
        Value::HandleObject(handle) => handle.class_name.as_str(),
        _ => "",
    };
    if runmat_builtins::is_class_or_subclass(class_name, crate::testing::TEST_CASE_CLASS) {
        Ok(())
    } else {
        Err(testing_diagnostic_error(
            "log: first argument must be a TestCase",
        ))
    }
}

fn log_severity(level: Value) -> BuiltinResult<DiagnosticSeverity> {
    let level = match level {
        Value::Num(level) => level,
        Value::Int(level) => level.to_f64(),
        _ => {
            return Err(testing_diagnostic_error(
                "log: verbosity level must be a numeric scalar",
            ))
        }
    };
    Ok(if level <= 1.0 {
        DiagnosticSeverity::Information
    } else if level <= 3.0 {
        DiagnosticSeverity::Warning
    } else {
        DiagnosticSeverity::Error
    })
}

fn diagnostic_message(value: Value) -> BuiltinResult<String> {
    match value {
        Value::String(message) => Ok(message),
        Value::CharArray(chars) if chars.rows <= 1 => Ok(chars.data.iter().collect()),
        Value::Object(object)
            if runmat_builtins::is_class_or_subclass(
                &object.class_name,
                "matlab.unittest.diagnostics.Diagnostic",
            ) =>
        {
            object
                .properties
                .get("Message")
                .and_then(|value| match value {
                    Value::String(message) => Some(message.clone()),
                    _ => None,
                })
                .ok_or_else(|| testing_diagnostic_error("log: diagnostic has no text Message"))
        }
        _ => Err(testing_diagnostic_error(
            "log: diagnostic must be scalar text or a Diagnostic object",
        )),
    }
}

fn empty_value() -> BuiltinResult<Value> {
    runmat_value::Tensor::new(Vec::new(), vec![0, 0])
        .map(Value::Tensor)
        .map_err(|error| testing_diagnostic_error(error))
}

fn testing_diagnostic_error(message: impl Into<String>) -> crate::RuntimeError {
    crate::build_runtime_error(message)
        .with_identifier("RunMat:Testing:InvalidDiagnostic")
        .with_builtin("log")
        .build()
}
