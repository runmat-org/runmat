use super::*;
use runmat_execution_artifact::*;
use runmat_value::{CharArray, Value};

async fn original_request() -> ProgramExecutionRequest {
    let mut producer = runmat_core::RunMatSession::with_options(false, false).unwrap();
    let source = runmat_core::ExecutableSource::frozen(
        "frozen-host",
        "environment.m",
        "environment.m",
        "function y = environment(); y = getenv('RUNMAT_NATIVE_CONTEXT'); end",
    );
    let unit = producer
        .compile_executable_unit(source, None)
        .await
        .unwrap();
    let envelope = unit.portable_envelope().unwrap();
    let recipe = ProgramBuildRecipe {
        schema_version: PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
        program_revision: envelope.manifest.identity.program.clone(),
        entrypoint: envelope.manifest.identity.entrypoint_function.0.to_string(),
        outputs: runmat_execution::OutputContract {
            requested_outputs: 1,
        },
        execution_mode: "interpreter".into(),
        target: ProgramTarget::portable("frozen-environment"),
        features: Default::default(),
        compile_options: Default::default(),
        source_objects: Vec::new(),
        expected_artifact_id: None,
    };
    let request: ProgramExecutionRequest = ProgramExecutionRequest {
        environment: runmat_execution::EnvironmentSnapshot::default(),
        schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_V2,
        function: envelope.manifest.identity.entrypoint_function.0 as usize,
        artifact: ProgramArtifact::materialize(
            &recipe,
            ExecutableForm::ExecutableUnitV4,
            envelope.canonical_bytes().unwrap(),
        )
        .unwrap(),
        recipe,
        arguments: Vec::new(),
        requested_outputs: 1,
    };
    serde_json::from_slice(&serde_json::to_vec(&request).unwrap()).unwrap()
}

#[tokio::test]
async fn original_unit_consumes_exact_frozen_environment_and_live_cancellation() {
    let request = original_request().await;
    for value in ["first", "second"] {
        let environment = [("RUNMAT_NATIVE_CONTEXT".into(), value.into())];
        let mut request = request.clone();
        request.environment =
            runmat_execution::EnvironmentSnapshot::new(environment.to_vec()).unwrap();
        let mismatch = [("RUNMAT_NATIVE_CONTEXT".into(), "substituted".into())];
        assert!(matches!(
            execute_unit(&request, None, Some(&mismatch), None).await,
            ProgramExecutionResponse::Failure { .. }
        ));
        let response = execute_unit(&request, None, None, None).await;
        response.validate_against(&request).unwrap();
        let ProgramExecutionResponse::Success { value: output } = response else {
            panic!("original execution failed: {response:?}")
        };
        assert_eq!(
            runmat_runtime::execution::value_codec::decode_inline_value(&output).unwrap(),
            Value::CharArray(CharArray::new_row(value))
        );
    }
    let cancelled = Arc::new(AtomicBool::new(true));
    assert!(matches!(
        execute_unit(&request, None, Some(&[]), Some(cancelled)).await,
        ProgramExecutionResponse::Failure { .. }
    ));
}
