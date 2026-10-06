#![cfg(target_arch = "wasm32")]

use runmat_execution::OutputContract;
use runmat_execution_artifact::*;
use runmat_value::Value;
use serde::Serialize;
use wasm_bindgen_test::wasm_bindgen_test;

async fn execute(request: ProgramExecutionRequest) -> ProgramExecutionResponse {
    let request = request
        .serialize(
            &serde_wasm_bindgen::Serializer::new().serialize_large_number_types_as_bigints(true),
        )
        .unwrap();
    let response = runmat_wasm::execute_program_artifact(request)
        .await
        .unwrap();
    serde_wasm_bindgen::from_value(response).unwrap()
}

async fn original(text: &str, selected: Option<&str>, outputs: u16) -> ProgramExecutionRequest {
    runmat_runtime::builtins::wasm_registry::register_all();
    let mut session = runmat_core::RunMatSession::with_options(false, false).unwrap();
    let unit = session
        .compile_executable_unit(
            runmat_core::ExecutableSource::new("browser-original", "outer.m", text),
            None,
        )
        .await
        .unwrap();
    let envelope = unit.portable_envelope_for(selected).unwrap();
    let recipe = ProgramBuildRecipe {
        schema_version: PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
        program_revision: envelope.manifest.identity.program.clone(),
        entrypoint: envelope.manifest.identity.entrypoint_function.0.to_string(),
        outputs: OutputContract {
            requested_outputs: outputs,
        },
        execution_mode: "interpreter".into(),
        target: ProgramTarget::portable("browser-original-test"),
        features: Default::default(),
        compile_options: Default::default(),
        source_objects: vec![],
        expected_artifact_id: None,
    };
    ProgramExecutionRequest {
        schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
        function: envelope.manifest.identity.entrypoint_function.0 as usize,
        artifact: ProgramArtifact::materialize(
            &recipe,
            ExecutableForm::ExecutableUnitV3,
            envelope.canonical_bytes().unwrap(),
        )
        .unwrap(),
        recipe,
        arguments: vec![],
        requested_outputs: outputs,
    }
}

fn values(response: ProgramExecutionResponse) -> Vec<Value> {
    let outputs = match response {
        ProgramExecutionResponse::Success { value } => vec![value],
        ProgramExecutionResponse::ExternalizedSuccess {
            outputs,
            result_objects,
        } => {
            assert!(result_objects.is_empty());
            outputs
        }
        other => panic!("browser original failed: {other:?}"),
    };
    outputs
        .iter()
        .map(|value| runmat_runtime::execution::value_codec::decode_inline_value(value).unwrap())
        .collect()
}

#[wasm_bindgen_test]
async fn browser_original_capture_and_nonprimary_identity_preserve_all_output_ordinals() {
    for outputs in [0, 1, 2] {
        let mut request = original(
            concat!(
                "function y = outer(); y = 999; end\n",
                "function [y,z] = selected(x)\ng = @(t) helper(t) + x;\n",
                "y = g(3); z = helper(x); end\n",
                "function y = helper(x); y = 2*x; end\n",
            ),
            Some("selected"),
            outputs,
        )
        .await;
        assert_ne!(request.function, 0);
        request.arguments =
            vec![
                runmat_runtime::execution::value_codec::encode_inline_value(&Value::Num(10.0))
                    .unwrap(),
            ];
        let cold = serde_json::from_slice(&serde_json::to_vec(&request).unwrap()).unwrap();
        let response = execute(cold).await;
        response.validate_against(&request).unwrap();
        assert_eq!(
            values(response),
            vec![Value::Num(16.0), Value::Num(20.0)][..usize::from(outputs)].to_vec()
        );
    }
}

#[wasm_bindgen_test]
async fn browser_script_requires_actual_ans_and_preserves_empty_outputs() {
    let assigned = original("ans = 42; unrelated = 999;", None, 1).await;
    assert_eq!(values(execute(assigned).await), vec![Value::Num(42.0)]);
    for text in [
        "unrelated = 999;",
        "if false; ans = 42; end; unrelated = 999;",
    ] {
        assert!(matches!(
            execute(original(text, None, 1).await).await,
            ProgramExecutionResponse::Failure { .. }
        ));
        assert!(values(execute(original(text, None, 0).await).await).is_empty());
    }
}

// Exercise each original callable form through the public artifact boundary.
#[wasm_bindgen_test]
async fn original_callable_forms_preserve_nonprimary_outputs() {
    for (name, text) in [
        ("direct", "function y = outer(); y = 999; end\nfunction [y,z] = selected(x); y = x+6; z = 2*x; end\n"),
        ("helper", "function y = outer(); y = 999; end\nfunction [y,z] = selected(x); y = helper(3)+x; z = helper(x); end\nfunction y = helper(x); y = 2*x; end\n"),
        ("creation", "function y = outer(); y = 999; end\nfunction [y,z] = selected(x); g = @(t) 2*t+x; y = x+6; z = 2*x; end\n"),
        ("bound-handle", "function y = outer(); y = 999; end\nfunction [y,z] = selected(x); g = @helper; y = g(3)+x; z = 2*x; end\nfunction y = helper(x); y = 2*x; end\n"),
        ("uncaptured", "function y = outer(); y = 999; end\nfunction [y,z] = selected(x); g = @(t) 2*t; y = g(3)+x; z = 2*x; end\n"),
        ("capture", "function y = outer(); y = 999; end\nfunction [y,z] = selected(x); g = @(t) 2*t+x; y = g(3); z = 2*x; end\n"),
    ] {
        let mut request = original(text, Some("selected"), 2).await;
        request.arguments = vec![runmat_runtime::execution::value_codec::encode_inline_value(&Value::Num(10.0)).unwrap()];
        assert_eq!(values(execute(request).await), vec![Value::Num(16.0), Value::Num(20.0)], "{name}");
    }
}
