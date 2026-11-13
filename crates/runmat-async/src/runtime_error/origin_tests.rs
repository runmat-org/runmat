use super::{runtime_error, RuntimeFailureOrigin};

#[test]
fn diagnostic_spelling_does_not_establish_failure_origin() {
    let error = runtime_error("Undefined function: example")
        .with_identifier("RunMat:UndefinedFunction")
        .build();
    assert_eq!(error.failure_origin(), RuntimeFailureOrigin::Unclassified);
}

#[test]
fn declared_language_origin_retains_owning_source() {
    let error = runtime_error("invalid expression")
        .with_failure_origin(RuntimeFailureOrigin::LanguageEvaluation)
        .with_source(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "detail",
        ))
        .build();
    assert_eq!(
        error.failure_origin(),
        RuntimeFailureOrigin::LanguageEvaluation
    );
    assert!(std::error::Error::source(&error)
        .expect("retained source")
        .downcast_ref::<std::io::Error>()
        .is_some());
}
