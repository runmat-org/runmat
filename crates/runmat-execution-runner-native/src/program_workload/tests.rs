use super::*;
use runmat_execution_artifact::*;
use runmat_value::{CharArray, Value};

async fn original_request() -> ProgramExecutionRequest {
    request_for_source("function y = environment(); y = getenv('RUNMAT_NATIVE_CONTEXT'); end").await
}

async fn request_for_source(source_text: &str) -> ProgramExecutionRequest {
    let mut producer = runmat_core::RunMatSession::with_options(false, false).unwrap();
    let source = runmat_core::ExecutableSource::frozen(
        "frozen-host",
        "environment.m",
        "environment.m",
        source_text,
    );
    let unit = producer
        .compile_executable_unit(source, None)
        .await
        .unwrap();
    let envelope = unit.portable_envelope().unwrap();
    let recipe = ProgramBuildRecipe {
        accelerators: Vec::new(),
        interop: runmat_types::InteropManifest::empty(),
        foreign_policy: None,
        schema_version: PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
        program_revision: envelope.manifest.identity.program.clone(),
        entrypoint: envelope.manifest.identity.entrypoint.clone(),
        outputs: runmat_execution::OutputContract {
            owned_work: None,
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
        console: None,
        environment: runmat_execution::EnvironmentSnapshot::default(),
        schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
        callable: runmat_execution::ProgramCallable::semantic(
            envelope
                .manifest
                .identity
                .entrypoint_function
                .expect("compiled semantic entrypoint"),
            None,
        ),
        context: runmat_execution::ProgramInvocationContext::Direct,
        assignment: None,
        job_id: None,
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
async fn owned_transcript_preserves_success_and_original_failure_prefix() {
    use runmat_execution::console::{ConsoleCompletion, ConsoleLimits};
    for (source, succeeds) in [
        ("function y = environment(); disp('owned sentinel'); y = 7; end", true),
        ("function y = environment(); fprintf('failure prefix'); error('original failure'); y = 7; end", false),
    ] {
        let mut request = request_for_source(source).await;
        request.console = Some(ConsoleLimits { max_records: 16, max_text_bytes: 1024 });
        assert!(matches!(execute_unit(&request, None, None, None).await, ProgramExecutionResponse::Failure { .. }));
        let (response, transcript) = execute_unit_with_transcript(&request, None, None, None, None).await;
        assert_eq!(matches!(response, ProgramExecutionResponse::Success { .. }), succeeds, "{response:?}");
        let transcript = transcript.expect("requested capture retains its original owner inventory");
        assert_eq!(transcript.completion(), ConsoleCompletion::Complete);
        let text: String = transcript.records().iter().map(|record| record.text.as_ref()).collect();
        assert!(text.contains(if succeeds { "owned sentinel" } else { "failure prefix" }));
        if !succeeds {
            assert!(format!("{response:?}").contains("original failure"));
        }
    }
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
        let ProgramExecutionResponse::Success { mut outputs } = response else {
            panic!("original execution failed: {response:?}")
        };
        assert_eq!(outputs.len(), 1, "one original semantic output");
        let output = outputs.remove(0);
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
