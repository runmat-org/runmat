use super::EvalHookOutcome;
use crate::build_runtime_error;
use crate::runtime_error::RuntimeFailureOrigin;

#[test]
fn language_failure_is_a_diagnostic_outcome() {
    let error = build_runtime_error("undefined expression")
        .with_failure_origin(RuntimeFailureOrigin::LanguageEvaluation)
        .with_identifier("RunMat:UndefinedVariable")
        .build();
    let outcome = EvalHookOutcome::from_evaluation(Err(error)).expect("language outcome");
    let EvalHookOutcome::InvalidExpression(diagnostic) = outcome else {
        panic!("expected invalid expression");
    };
    assert_eq!(diagnostic.identifier(), Some("RunMat:UndefinedVariable"));
}

#[test]
fn operational_failure_retains_source_and_is_not_a_retry_outcome() {
    let error = build_runtime_error("provider failed")
        .with_source(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "denied",
        ))
        .build();
    let error = EvalHookOutcome::from_evaluation(Err(error)).expect_err("propagate provider error");
    let source = std::error::Error::source(&error)
        .expect("source")
        .downcast_ref::<std::io::Error>()
        .expect("original I/O error");
    assert_eq!(source.kind(), std::io::ErrorKind::PermissionDenied);
}

#[test]
fn diagnostic_identifier_alone_does_not_authorize_retry() {
    let error = build_runtime_error("Undefined function: ordinary")
        .with_identifier("RunMat:UndefinedFunction")
        .build();
    assert!(EvalHookOutcome::from_evaluation(Err(error)).is_err());
}

#[test]
fn cancellation_is_preserved_as_an_operational_failure() {
    let error = build_runtime_error("cancelled")
        .with_source(crate::execution::ExecutionServiceError::Cancelled)
        .build();
    let error = EvalHookOutcome::from_evaluation(Err(error)).expect_err("cancel must propagate");
    assert!(std::error::Error::source(&error)
        .expect("typed cancellation source")
        .downcast_ref::<crate::execution::ExecutionServiceError>()
        .is_some());
}
