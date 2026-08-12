use runmat_builtins::{ObjectInstance, Value};
use runmat_test::context::TestCommand;
use runmat_test::lifecycle::{ExecutionPhase, QualificationKind};
use runmat_test::result::{Diagnostic, DiagnosticDetail, DiagnosticSeverity};

use crate::{build_runtime_error, BuiltinResult};

#[derive(Clone, Copy)]
pub enum BinaryPredicate {
    Equal,
    NotEqual,
    GreaterThan,
    LessThan,
}

pub async fn qualify_binary(
    builtin: &'static str,
    qualification: QualificationKind,
    predicate: BinaryPredicate,
    receiver: Value,
    actual: Value,
    expected: Value,
    diagnostic_args: Vec<Value>,
) -> BuiltinResult<Value> {
    validate_receiver(builtin, &receiver)?;
    let passed = match predicate {
        BinaryPredicate::Equal => values_equal(&actual, &expected),
        BinaryPredicate::NotEqual => !values_equal(&actual, &expected),
        BinaryPredicate::GreaterThan => compare_numeric(&actual, &expected, |a, b| a > b),
        BinaryPredicate::LessThan => compare_numeric(&actual, &expected, |a, b| a < b),
    };
    finish_qualification(
        builtin,
        qualification,
        passed,
        format!("{builtin} failed"),
        diagnostic_args,
        vec![
            DiagnosticDetail {
                label: "Actual".into(),
                value: bounded_value(&actual),
            },
            DiagnosticDetail {
                label: "Expected".into(),
                value: bounded_value(&expected),
            },
        ],
    )
}

pub async fn qualify_logical(
    builtin: &'static str,
    qualification: QualificationKind,
    expected: bool,
    receiver: Value,
    actual: Value,
    diagnostic_args: Vec<Value>,
) -> BuiltinResult<Value> {
    validate_receiver(builtin, &receiver)?;
    let passed = logical_scalar(&actual).is_some_and(|actual| actual == expected);
    finish_qualification(
        builtin,
        qualification,
        passed,
        format!("{builtin} failed"),
        diagnostic_args,
        vec![
            DiagnosticDetail {
                label: "Actual".into(),
                value: bounded_value(&actual),
            },
            DiagnosticDetail {
                label: "Expected".into(),
                value: expected.to_string(),
            },
        ],
    )
}

pub async fn qualify_empty(
    builtin: &'static str,
    qualification: QualificationKind,
    expected_empty: bool,
    receiver: Value,
    actual: Value,
    diagnostic_args: Vec<Value>,
) -> BuiltinResult<Value> {
    validate_receiver(builtin, &receiver)?;
    let empty = crate::builtins::common::shape::value_numel(&actual).await? == 0;
    finish_qualification(
        builtin,
        qualification,
        empty == expected_empty,
        format!("{builtin} failed"),
        diagnostic_args,
        vec![DiagnosticDetail {
            label: "ActualSize".into(),
            value: format!(
                "{:?}",
                crate::builtins::common::shape::value_dimensions(&actual).await?
            ),
        }],
    )
}

pub fn qualify_that(
    builtin: &'static str,
    qualification: QualificationKind,
    receiver: Value,
    actual: Value,
    constraint: Value,
    diagnostic_args: Vec<Value>,
) -> BuiltinResult<Value> {
    validate_receiver(builtin, &receiver)?;
    let Value::Object(constraint) = constraint else {
        return Err(build_runtime_error(format!(
            "{builtin}: expected a matlab.unittest.constraints.Constraint"
        ))
        .with_identifier("RunMat:Testing:InvalidConstraint")
        .with_builtin(builtin)
        .build());
    };
    if !runmat_builtins::is_class_or_subclass(
        &constraint.class_name,
        "matlab.unittest.constraints.Constraint",
    ) {
        return Err(build_runtime_error(format!(
            "{builtin}: '{}' is not a compatible Constraint",
            constraint.class_name
        ))
        .with_identifier("RunMat:Testing:InvalidConstraint")
        .with_builtin(builtin)
        .build());
    }
    let passed = match constraint.class_name.as_str() {
        "matlab.unittest.constraints.IsEqualTo" => constraint
            .properties
            .get("__runmat_expected")
            .is_some_and(|expected| values_equal(&actual, expected)),
        "matlab.unittest.constraints.IsTrue" => logical_scalar(&actual) == Some(true),
        "matlab.unittest.constraints.IsFalse" => logical_scalar(&actual) == Some(false),
        _ => {
            return Err(build_runtime_error(format!(
                "{builtin}: constraint '{}' requires an unsupported custom satisfaction hook",
                constraint.class_name
            ))
            .with_identifier("RunMat:Testing:UnsupportedConstraint")
            .with_builtin(builtin)
            .build())
        }
    };
    finish_qualification(
        builtin,
        qualification,
        passed,
        format!("{builtin} failed for {}", constraint.class_name),
        diagnostic_args,
        vec![DiagnosticDetail {
            label: "Actual".into(),
            value: bounded_value(&actual),
        }],
    )
}

fn finish_qualification(
    builtin: &'static str,
    qualification: QualificationKind,
    passed: bool,
    default_message: String,
    diagnostic_args: Vec<Value>,
    details: Vec<DiagnosticDetail>,
) -> BuiltinResult<Value> {
    if passed {
        return empty_value();
    }
    let message = diagnostic_message(&diagnostic_args).unwrap_or(default_message);
    let phase = crate::testing::active_test_context()
        .map(|context| context.phase)
        .unwrap_or(ExecutionPhase::TestBody);
    let identifier = qualification_identifier(qualification);
    let diagnostic = Diagnostic {
        identifier: identifier.into(),
        message: message.clone(),
        severity: DiagnosticSeverity::Error,
        phase,
        source: None,
        details,
    };
    let recorded = crate::testing::record_test_command(TestCommand::Qualify {
        qualification,
        diagnostic,
    })
    .is_ok();
    if recorded && !qualification.aborts_test() {
        return empty_value();
    }
    Err(build_runtime_error(message)
        .with_identifier(identifier)
        .with_builtin(builtin)
        .build())
}

fn validate_receiver(builtin: &'static str, receiver: &Value) -> BuiltinResult<()> {
    crate::testing::ensure_testing_classes();
    let class_name = match receiver {
        Value::Object(object) => object.class_name.as_str(),
        Value::HandleObject(handle) => handle.class_name.as_str(),
        _ => {
            return Err(build_runtime_error(format!(
                "{builtin}: first argument must be a matlab.unittest.TestCase"
            ))
            .with_identifier("RunMat:Testing:InvalidTestCase")
            .with_builtin(builtin)
            .build())
        }
    };
    if runmat_builtins::is_class_or_subclass(class_name, crate::testing::TEST_CASE_CLASS) {
        Ok(())
    } else {
        Err(build_runtime_error(format!(
            "{builtin}: receiver class '{class_name}' is not a matlab.unittest.TestCase"
        ))
        .with_identifier("RunMat:Testing:InvalidTestCase")
        .with_builtin(builtin)
        .build())
    }
}

fn values_equal(actual: &Value, expected: &Value) -> bool {
    numeric_scalar(actual)
        .zip(numeric_scalar(expected))
        .map_or_else(
            || actual == expected,
            |(actual, expected)| actual == expected,
        )
}

fn compare_numeric(
    actual: &Value,
    expected: &Value,
    compare: impl FnOnce(f64, f64) -> bool,
) -> bool {
    numeric_scalar(actual)
        .zip(numeric_scalar(expected))
        .is_some_and(|(actual, expected)| compare(actual, expected))
}

fn numeric_scalar(value: &Value) -> Option<f64> {
    match value {
        Value::Num(value) => Some(*value),
        Value::Int(value) => Some(value.to_f64()),
        Value::Bool(value) => Some(if *value { 1.0 } else { 0.0 }),
        Value::Tensor(tensor) if tensor.len() == 1 => tensor.materialize_f64().first().copied(),
        _ => None,
    }
}

fn logical_scalar(value: &Value) -> Option<bool> {
    match value {
        Value::Bool(value) => Some(*value),
        Value::LogicalArray(array) if array.data.len() == 1 => {
            array.data.first().map(|value| *value != 0)
        }
        _ => None,
    }
}

fn diagnostic_message(values: &[Value]) -> Option<String> {
    values.first().and_then(|value| match value {
        Value::String(message) => Some(message.clone()),
        Value::CharArray(chars) if chars.rows <= 1 => Some(chars.data.iter().collect::<String>()),
        Value::Object(ObjectInstance {
            class_name,
            properties,
            ..
        }) if class_name == "matlab.unittest.diagnostics.Diagnostic" => {
            properties.get("Message").and_then(|value| match value {
                Value::String(message) => Some(message.clone()),
                _ => None,
            })
        }
        _ => None,
    })
}

fn bounded_value(value: &Value) -> String {
    let mut rendered = value.to_string();
    if rendered.len() > 4_096 {
        let mut end = 4_096;
        while !rendered.is_char_boundary(end) {
            end -= 1;
        }
        rendered.truncate(end);
        rendered.push('…');
    }
    rendered
}

fn qualification_identifier(qualification: QualificationKind) -> &'static str {
    match qualification {
        QualificationKind::VerificationFailed => "RunMat:Test:VerificationFailed",
        QualificationKind::AssumptionFailed => "RunMat:Test:AssumptionFailed",
        QualificationKind::AssertionFailed => "RunMat:Test:AssertionFailed",
        QualificationKind::FatalAssertionFailed => "RunMat:Test:FatalAssertionFailed",
    }
}

fn empty_value() -> BuiltinResult<Value> {
    runmat_builtins::Tensor::new(Vec::new(), vec![0, 0])
        .map(Value::Tensor)
        .map_err(|error| build_runtime_error(error).build())
}

#[cfg(test)]
mod tests {
    use runmat_test::context::TestExecutionContext;
    use runmat_test::descriptor::FixtureScope;
    use runmat_test::identity::{RunId, TestId};
    use runmat_test::lifecycle::FixtureScopeKey;
    use runmat_test::protocol::ProtocolLimits;

    use super::*;

    fn context() -> crate::testing::TestContextGuard {
        crate::testing::install_test_context(
            crate::testing::ActiveTestContext {
                execution: TestExecutionContext {
                    run_id: RunId::derive("revision", "run"),
                    test_id: TestId::derive(&runmat_test::identity::TestIdentityInput {
                        owner_identity: "owner",
                        relative_source_identity: "test.m",
                        semantic_scheme: "function",
                        semantic_item_path: "test",
                        parameter_identity: "",
                        fixture_identity: "",
                    }),
                    attempt: 1,
                    random_seed: 7,
                },
                phase: ExecutionPhase::TestBody,
                scope: FixtureScopeKey {
                    scope: FixtureScope::Test,
                    identity: "test".into(),
                },
            },
            ProtocolLimits::default(),
        )
    }

    #[test]
    fn failed_verification_records_and_continues() {
        let guard = context();
        let value = futures::executor::block_on(qualify_binary(
            "verifyEqual",
            QualificationKind::VerificationFailed,
            BinaryPredicate::Equal,
            crate::testing::test_case_object("test", false),
            Value::Num(1.0),
            Value::Num(2.0),
            Vec::new(),
        ))
        .unwrap();
        assert!(matches!(value, Value::Tensor(tensor) if tensor.is_empty()));
        assert!(matches!(
            guard.handle().commands().as_slice(),
            [TestCommand::Qualify {
                qualification: QualificationKind::VerificationFailed,
                ..
            }]
        ));
    }

    #[test]
    fn failed_assertion_records_and_aborts() {
        let guard = context();
        let error = futures::executor::block_on(qualify_logical(
            "assertTrue",
            QualificationKind::AssertionFailed,
            true,
            crate::testing::test_case_object("test", false),
            Value::Bool(false),
            Vec::new(),
        ))
        .unwrap_err();
        assert_eq!(
            error.identifier.as_deref(),
            Some("RunMat:Test:AssertionFailed")
        );
        assert_eq!(guard.handle().commands().len(), 1);
    }
}
