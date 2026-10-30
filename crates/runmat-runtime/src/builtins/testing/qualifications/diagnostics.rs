use runmat_test::context::TestCommand;
use runmat_test::lifecycle::ExecutionPhase;
use runmat_test::result::{Diagnostic, DiagnosticDetail, DiagnosticSeverity};
use runmat_value::{NumericScalar, Value};

use crate::BuiltinResult;

#[runmat_macros::runtime_builtin(
    name = "matlab.unittest.TestCase",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::testing::qualifications::diagnostics"
)]
fn test_case_constructor(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<crate::execution::RootedValue> {
    let runtime = runtime.as_ref();
    crate::testing::ensure_testing_classes();
    if !args.is_empty() {
        return Err(crate::build_runtime_error(
            "matlab.unittest.TestCase: too many input arguments",
        )
        .with_identifier("RunMat:Testing:InvalidInput")
        .with_builtin("matlab.unittest.TestCase")
        .build());
    }
    crate::testing::test_case_object_in_context(runtime, "".into(), false)
}

#[runmat_macros::runtime_builtin(
    name = "matlab.unittest.diagnostics.StringDiagnostic",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::testing::qualifications::diagnostics"
)]
fn string_diagnostic(
    runtime: Option<crate::context::RuntimeContext>,
    value: Value,
) -> BuiltinResult<crate::execution::RootedValue> {
    let runtime = runtime.as_ref();
    crate::testing::ensure_testing_classes();
    let message = match &value {
        Value::String(_) => crate::call::argument_text::convert(&value, runtime)?,
        Value::CharArray(chars) if chars.rows == 0 => "".into(),
        Value::CharArray(chars) if chars.rows == 1 => {
            crate::call::argument_text::convert(&value, runtime)?
        }
        _ => {
            return Err(
                crate::build_runtime_error("StringDiagnostic: expected scalar text")
                    .with_identifier("RunMat:Testing:InvalidDiagnostic")
                    .with_builtin("StringDiagnostic")
                    .build(),
            )
        }
    };
    let fields = [
        ("Identifier", Value::String("".into())),
        (
            "Message",
            Value::String(runmat_value::StringElement::Text(message.clone())),
        ),
        (
            "DiagnosticText",
            Value::String(runmat_value::StringElement::Text(message)),
        ),
    ];
    crate::execution::object::construct_object_from_borrowed_fields(
        runtime,
        runmat_types::standard::UNIT_TEST_STRING_DIAGNOSTIC.display_name(),
        fields.iter().map(|(name, value)| (*name, value)),
    )
}

#[runmat_macros::runtime_builtin(
    name = "matlab.unittest.TestCase.log",
    binding_variant = "default",
    builtin_path = "crate::builtins::testing::qualifications::diagnostics"
)]
fn log(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    let (receiver, level, diagnostic) = match args.len() {
        2 => (&args[0], 2, &args[1]),
        3 => (&args[0], log_verbosity(&args[1])?, &args[2]),
        _ => {
            return Err(testing_diagnostic_error(
                "log: expected a TestCase, optional verbosity, and diagnostic",
            ))
        }
    };
    validate_test_case(receiver)?;
    let message = diagnostic_message(diagnostic)?;
    let phase = crate::testing::active_test_context()
        .map(|context| context.phase)
        .unwrap_or(ExecutionPhase::TestBody);
    crate::testing::record_test_command(TestCommand::RecordDiagnostic {
        diagnostic: Diagnostic {
            identifier: "RunMat:Test:Log".into(),
            message,
            severity: DiagnosticSeverity::Information,
            phase,
            source: None,
            details: vec![DiagnosticDetail {
                label: "Verbosity".into(),
                value: level.to_string(),
            }],
        },
    })
    .map_err(testing_diagnostic_error)?;
    empty_value()
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
        Err(testing_diagnostic_error(
            "log: first argument must be a TestCase",
        ))
    }
}

fn log_verbosity(level: &Value) -> BuiltinResult<u8> {
    let scalar = match level {
        Value::Num(level) => NumericScalar::F64(*level),
        Value::Int(level) => NumericScalar::from(level.clone()),
        Value::Tensor(tensor) if tensor.len() == 1 => tensor
            .numeric_value_at(0)
            .expect("a one-element numeric tensor has a scalar value"),
        Value::String(runmat_value::StringElement::Text(level)) => return named_verbosity(level),
        Value::CharArray(chars) if chars.rows <= 1 => {
            let text =
                crate::builtins::common::text::decode_utf16_with_error(chars.data(), || {
                    testing_diagnostic_error("log: verbosity is not valid UTF-16 text")
                })?;
            return named_verbosity(&text);
        }
        _ => {
            return Err(testing_diagnostic_error(
                "log: verbosity must be a level from 1 to 4 or its name",
            ))
        }
    };
    let valid = match scalar.into_int_value() {
        Some(integer) => (1..=4).contains(&integer.to_i128()),
        None => {
            let number = scalar.materialize_f64();
            number.is_finite() && (1.0..=4.0).contains(&number) && number.fract() == 0.0
        }
    };
    if !valid {
        return Err(testing_diagnostic_error(
            "log: verbosity must be an integer from 1 to 4",
        ));
    }
    Ok(scalar.materialize_f64() as u8)
}

fn named_verbosity(level: &str) -> BuiltinResult<u8> {
    match level.to_ascii_lowercase().as_str() {
        "terse" => Ok(1),
        "concise" => Ok(2),
        "detailed" => Ok(3),
        "verbose" => Ok(4),
        _ => Err(testing_diagnostic_error("log: unknown verbosity name")),
    }
}

fn diagnostic_message(value: &Value) -> BuiltinResult<String> {
    match value {
        Value::String(runmat_value::StringElement::Text(message)) => {
            Ok(message.as_str().to_owned())
        }
        Value::CharArray(chars) if chars.rows <= 1 => {
            crate::builtins::common::text::decode_utf16_with_error(chars.data(), || {
                testing_diagnostic_error("log: diagnostic message is not valid UTF-16 text")
            })
        }
        Value::Object(object)
            if crate::class_registry::is_class_or_subclass(
                object.class_name(),
                &runmat_types::standard::UNIT_TEST_DIAGNOSTIC.owned(),
            ) =>
        {
            object
                .properties()
                .get("Message")
                .and_then(|value| match value {
                    Value::String(runmat_value::StringElement::Text(message)) => {
                        Some(message.as_str().to_owned())
                    }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn test_case_constructor(
        runtime: Option<crate::context::RuntimeContext>,
        args: Vec<Value>,
    ) -> BuiltinResult<crate::execution::RootedValue> {
        super::test_case_constructor(
            runtime,
            crate::execution::RootedValues::new(args)
                .map_err(|error| error.into_runtime_error())?
                .into(),
        )
    }

    fn log(args: Vec<Value>) -> BuiltinResult<Value> {
        super::log(
            crate::execution::RootedValues::new(args)
                .map_err(|error| error.into_runtime_error())?
                .into(),
        )
    }
    use crate::testing::ActiveTestContext;
    use runmat_test::context::TestExecutionContext;
    use runmat_test::descriptor::FixtureScope;
    use runmat_test::identity::{RunId, TestId, TestIdentityInput};
    use runmat_test::lifecycle::FixtureScopeKey;
    use runmat_test::protocol::ProtocolLimits;

    #[test]
    fn string_diagnostic_retains_its_concrete_class_and_text() {
        let diagnostic = string_diagnostic(None, Value::String("ready".into())).unwrap();
        let Value::Object(object) = diagnostic.value() else {
            panic!("StringDiagnostic must return an object");
        };
        assert!(object
            .class_name()
            .is(runmat_types::standard::UNIT_TEST_STRING_DIAGNOSTIC));
        assert_eq!(
            object.properties().get("DiagnosticText"),
            Some(&Value::String("ready".into()))
        );
        assert_eq!(
            object.properties().get("Message"),
            Some(&Value::String("ready".into()))
        );
        assert!(string_diagnostic(None, Value::Num(4.0)).is_err());
    }

    #[test]
    fn log_accepts_default_and_explicit_verbosity_and_records_commands() {
        let guard = crate::testing::install_test_context(
            ActiveTestContext {
                execution: TestExecutionContext {
                    run_id: RunId::derive("diagnostics", "run"),
                    test_id: TestId::derive(&TestIdentityInput {
                        owner_identity: "diagnostics",
                        relative_source_identity: "diagnostics.m",
                        semantic_scheme: "function",
                        semantic_item_path: "log",
                        parameter_identity: "",
                        fixture_identity: "",
                    }),
                    attempt: 1,
                    random_seed: 1,
                },
                phase: ExecutionPhase::TestBody,
                scope: FixtureScopeKey {
                    scope: FixtureScope::Test,
                    identity: "diagnostics".into(),
                },
            },
            ProtocolLimits::default(),
        );
        let receiver = test_case_constructor(None, Vec::new()).unwrap();
        log(vec![
            receiver.value().clone(),
            Value::String("default".into()),
        ])
        .unwrap();
        log(vec![
            receiver.value().clone(),
            Value::String("Verbose".into()),
            Value::String("explicit".into()),
        ])
        .unwrap();
        let diagnostic =
            string_diagnostic(None, Value::String("object diagnostic".into())).unwrap();
        log(vec![receiver.value().clone(), diagnostic.value().clone()]).unwrap();
        let commands = guard.handle().commands();
        assert_eq!(commands.len(), 3);
        let TestCommand::RecordDiagnostic {
            diagnostic: default,
        } = &commands[0]
        else {
            panic!("expected recorded diagnostic")
        };
        let TestCommand::RecordDiagnostic {
            diagnostic: explicit,
        } = &commands[1]
        else {
            panic!("expected recorded diagnostic")
        };
        assert_eq!(default.message, "default");
        assert_eq!(default.details[0].value, "2");
        assert_eq!(explicit.message, "explicit");
        assert_eq!(explicit.details[0].value, "4");
        assert_eq!(explicit.severity, DiagnosticSeverity::Information);
        let TestCommand::RecordDiagnostic { diagnostic: object } = &commands[2] else {
            panic!("expected recorded object diagnostic")
        };
        assert_eq!(object.message, "object diagnostic");
    }

    #[test]
    fn log_verbosity_accepts_scalar_tensors_without_rounding_wide_integers() {
        let double = runmat_value::Tensor::new(vec![3.0], vec![1, 1]).unwrap();
        assert_eq!(log_verbosity(&Value::Tensor(double)).unwrap(), 3);
        let integer = runmat_value::Tensor::new_integer(
            runmat_value::IntegerStorage::U16(vec![4]),
            vec![1, 1],
        )
        .unwrap();
        assert_eq!(log_verbosity(&Value::Tensor(integer)).unwrap(), 4);
        let wide = runmat_value::Tensor::new_integer(
            runmat_value::IntegerStorage::U64(vec![(1_u64 << 53) + 1]),
            vec![1, 1],
        )
        .unwrap();
        assert!(log_verbosity(&Value::Tensor(wide)).is_err());
        assert!(log_verbosity(&Value::Tensor(
            runmat_value::Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()
        ))
        .is_err());
    }

    #[test]
    fn invalid_log_calls_do_not_record_commands() {
        assert!(log(vec![]).is_err());
        assert!(log(vec![Value::Num(1.0), Value::String("x".into())]).is_err());
        for level in [
            Value::Num(0.0),
            Value::Num(2.5),
            Value::Num(f64::NAN),
            Value::String("trace".into()),
        ] {
            assert!(log(vec![Value::Num(1.0), level, Value::String("x".into())]).is_err());
        }
    }
}
