#![cfg(target_arch = "wasm32")]

use std::collections::BTreeSet;

use runmat_execution::value::{InlineValue, ValuePayload};
use runmat_execution::OutputContract;
use runmat_execution_artifact::{
    ExecutableForm, ProgramArtifact, ProgramBuildRecipe, ProgramExecutionRequest,
    ProgramExecutionResponse, ProgramTarget, PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
    PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
};
use wasm_bindgen_test::wasm_bindgen_test;

wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
async fn browser_executes_the_exact_portable_artifact_without_a_project() {
    let mut session = runmat_core::RunMatSession::with_options(false, false).unwrap();
    let unit = session
        .compile_executable_unit(
            runmat_core::ExecutableSource::new("root", "answer.m", "ans = 42"),
            None,
        )
        .await
        .unwrap();
    let envelope = unit.portable_envelope().unwrap();
    let function = usize::try_from(envelope.manifest.identity.entrypoint_function.0).unwrap();
    let recipe = ProgramBuildRecipe {
        schema_version: PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
        program_revision: envelope.manifest.identity.program.clone(),
        entrypoint: function.to_string(),
        outputs: OutputContract {
            requested_outputs: 1,
        },
        execution_mode: "interpreter".into(),
        target: ProgramTarget::portable("portable-executable-unit-v3"),
        features: BTreeSet::new(),
        compile_options: BTreeSet::new(),
        source_objects: Vec::new(),
        expected_artifact_id: None,
    };
    let artifact = ProgramArtifact::materialize(
        &recipe,
        ExecutableForm::ExecutableUnitV3,
        envelope.canonical_bytes().unwrap(),
    )
    .unwrap();
    let request = ProgramExecutionRequest {
        schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
        recipe,
        artifact,
        function,
        arguments: Vec::new(),
        requested_outputs: 1,
    };

    let response =
        runmat_wasm::execute_program_artifact(serde_wasm_bindgen::to_value(&request).unwrap())
            .await
            .unwrap();
    let response: ProgramExecutionResponse = serde_wasm_bindgen::from_value(response).unwrap();
    let ProgramExecutionResponse::Success {
        value: ValuePayload::Inline(value),
    } = response
    else {
        panic!("browser rejected an exact portable program artifact");
    };
    assert_eq!(*value, InlineValue::F64Bits(42.0_f64.to_bits()));
}
