use super::*;

#[test]
fn missing_environment_cannot_decode_as_an_ambient_request() {
    let request = super::tests::request();
    request.validate_for_portable_host().unwrap();
    let mut wire = serde_json::to_value(&request).unwrap();
    assert_eq!(
        serde_json::from_value::<ProgramExecutionRequest>(wire.clone()).unwrap(),
        request
    );
    wire.as_object_mut().unwrap().remove("environment");
    let error = serde_json::from_value::<ProgramExecutionRequest>(wire).unwrap_err();
    assert!(error.to_string().contains("environment"));
}

#[test]
fn original_environment_is_bound_to_request_identity_and_borrowed_transport() {
    let mut request = super::tests::request();
    let empty = request.canonical_digest().unwrap();
    request.environment =
        EnvironmentSnapshot::new(vec![("RUNMAT_SAMPLE".into(), "original".into())]).unwrap();
    request.validate_for_portable_host().unwrap();
    let original = request.canonical_digest().unwrap();
    assert_ne!(empty, original);
    assert_eq!(original, request.as_borrowed().canonical_digest().unwrap());
    let bytes = serde_json::to_vec(&request).unwrap();
    assert_eq!(bytes, serde_json::to_vec(&request.as_borrowed()).unwrap());
    let cold: ProgramExecutionRequest = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(cold.canonical_digest().unwrap(), original);
    request.environment =
        EnvironmentSnapshot::new(vec![("RUNMAT_SAMPLE".into(), "substituted".into())]).unwrap();
    assert_ne!(request.canonical_digest().unwrap(), original);
}
