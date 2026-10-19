use runmat_execution::identity::ValueId;
use runmat_execution::schema::VALUE_PAYLOAD_SCHEMA_V1;
use runmat_execution::{Digest, OutputContract, ProgramEnvironment, ProgramRevision};

use super::*;
use crate::ExecutableForm;

pub(super) fn request() -> ProgramExecutionRequest {
    let revision = ProgramRevision::new(
        Digest::sha256(b"graph"),
        Digest::sha256(b"source"),
        ProgramEnvironment::new(
            1,
            1,
            Digest::sha256(b"runtime"),
            Digest::sha256(b"catalog"),
            "matlab",
        )
        .unwrap(),
    )
    .unwrap();
    let recipe = ProgramBuildRecipe {
        schema_version: crate::PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
        program_revision: revision,
        entrypoint: "7".into(),
        outputs: OutputContract {
            requested_outputs: 1,
        },
        execution_mode: "interpreter".into(),
        target: crate::ProgramTarget::portable("portable-test"),
        features: Default::default(),
        compile_options: Default::default(),
        source_objects: Vec::new(),
        expected_artifact_id: None,
    };
    let artifact = ProgramArtifact::materialize(
        &recipe,
        ExecutableForm::InterpreterBytecodeV1,
        b"program".to_vec(),
    )
    .unwrap();
    ProgramExecutionRequest {
        console: None,
        environment: EnvironmentSnapshot::default(),
        schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_V3,
        recipe,
        artifact,
        function: 7,
        arguments: Vec::new(),
        requested_outputs: 1,
    }
}

fn result_reference(bytes: &[u8]) -> ValueRef {
    let logical_digest = Digest::sha256(bytes);
    ValueRef {
        schema_version: VALUE_PAYLOAD_SCHEMA_V1,
        id: ValueId::derive(&[b"program-response-test", logical_digest.bytes()]),
        logical_digest,
        encoded_length: bytes.len() as u64,
        media_type: "application/vnd.runmat.test-object".into(),
        value_schema: "runmat.test-object.v1".into(),
        encryption_context: Digest::sha256(b"test-encryption-context"),
        kind: ValueRefKind::ResultObject,
        authorization_scope: "test-scope".into(),
        resident_fence: None,
    }
}

#[test]
fn exact_program_request_validates_every_identity_boundary() {
    request().validate().unwrap();
    let mut mismatched = request();
    mismatched.function = 8;
    assert!(mismatched.validate().is_err());
    let mut tampered = request();
    tampered.artifact.executable_bytes.push(0);
    assert!(tampered.validate().is_err());
}

#[test]
fn exact_program_request_rejects_unknown_schemas_and_output_drift() {
    let mut unknown = request();
    unknown.schema_version += 1;
    assert!(unknown.validate().is_err());
    let mut outputs = request();
    outputs.requested_outputs = 2;
    assert!(outputs.validate().is_err());
}

#[test]
fn canonical_request_digest_is_exact_without_a_small_envelope_limit() {
    let mut large = request();
    large.arguments.push(ValuePayload::Inline(Box::new(
        runmat_execution::value::InlineValue::String("x".repeat(300_000)),
    )));
    let digest = large.canonical_digest().unwrap();
    let mut changed = large;
    changed.arguments[0] = ValuePayload::Inline(Box::new(
        runmat_execution::value::InlineValue::String(format!("{}y", "x".repeat(299_999))),
    ));
    assert_ne!(digest, changed.canonical_digest().unwrap());

    let mut referenced = request();
    let mut reference = result_reference(b"argument");
    reference.kind = ValueRefKind::DriverObject;
    referenced
        .arguments
        .push(ValuePayload::Object(Box::new(reference.clone())));
    let digest = referenced.canonical_digest().unwrap();
    if let ValuePayload::Object(reference) = &mut referenced.arguments[0] {
        reference.authorization_scope = "another-scope".into();
    }
    assert_ne!(digest, referenced.canonical_digest().unwrap());
}

#[test]
fn borrowed_request_preserves_original_payload_wire_identity_and_cold_decode() {
    let mut owned = request();
    owned.arguments.push(ValuePayload::Inline(Box::new(
        runmat_execution::value::InlineValue::String("λ\\\"\n".repeat(25_000)),
    )));
    let borrowed = owned.as_borrowed();
    assert!(std::ptr::eq(borrowed.recipe, &owned.recipe));
    assert!(std::ptr::eq(borrowed.artifact, &owned.artifact));
    assert_eq!(borrowed.arguments.as_ptr(), owned.arguments.as_ptr());
    borrowed.validate_for_portable_host().unwrap();
    let bytes = serde_json::to_vec(&borrowed).unwrap();
    assert_eq!(bytes, serde_json::to_vec(&owned).unwrap());
    assert_eq!(
        borrowed.canonical_digest().unwrap(),
        owned.canonical_digest().unwrap()
    );
    let cold: ProgramExecutionRequest = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(cold, owned);
    cold.validate_for_portable_host().unwrap();
    let response = ProgramExecutionResponse::Success {
        value: ValuePayload::Inline(Box::new(runmat_execution::value::InlineValue::Logical(
            true,
        ))),
    };
    response.validate_against(&borrowed).unwrap();
    response.validate_against(&cold).unwrap();
    let mut foreign = borrowed;
    foreign.function += 1;
    assert!(foreign.validate().is_err());
    assert!(response.validate_against(&foreign).is_err());
}

#[test]
fn script_form_has_an_explicit_argument_free_entrypoint() {
    let mut script = request();
    script.recipe.entrypoint = "script".into();
    script.function = 0;
    script.artifact = ProgramArtifact::materialize(
        &script.recipe,
        ExecutableForm::InterpreterScriptV1,
        b"script-bytecode".to_vec(),
    )
    .unwrap();
    script.validate().unwrap();
    script
        .arguments
        .push(runmat_execution::value::ValuePayload::Inline(Box::new(
            runmat_execution::value::InlineValue::String("unexpected".into()),
        )));
    assert!(script.validate().is_err());
}

#[test]
fn task_success_projection_preserves_inline_delivery_and_request_arity() {
    let request = request();
    let value = ValuePayload::Inline(Box::new(runmat_execution::value::InlineValue::Logical(
        true,
    )));
    let response = ProgramExecutionResponse::from_success_outputs(vec![value.clone()], vec![]);
    assert_eq!(
        response,
        ProgramExecutionResponse::Success {
            value: value.clone()
        }
    );
    response.validate_against(&request).unwrap();
    for outputs in [vec![], vec![value.clone(), value]] {
        assert!(
            ProgramExecutionResponse::from_success_outputs(outputs, vec![])
                .validate_against(&request)
                .is_err()
        );
    }
    let root = result_reference(b"root");
    let response = ProgramExecutionResponse::from_success_outputs(
        vec![ValuePayload::Object(Box::new(root.clone()))],
        vec![root],
    );
    assert!(matches!(
        response,
        ProgramExecutionResponse::ExternalizedSuccess { .. }
    ));
    response.validate_against(&request).unwrap();
}

#[test]
fn inline_output_lists_preserve_exact_arity_through_cold_transport() {
    for count in [0, 2, 3] {
        let mut request = request();
        request.requested_outputs = count;
        request.recipe.outputs.requested_outputs = count;
        request.artifact = ProgramArtifact::materialize(
            &request.recipe,
            ExecutableForm::InterpreterBytecodeV1,
            b"program".to_vec(),
        )
        .unwrap();
        let outputs = (0..count)
            .map(|index| {
                ValuePayload::Inline(Box::new(runmat_execution::value::InlineValue::F64Bits(
                    f64::from(index).to_bits(),
                )))
            })
            .collect();
        let response = ProgramExecutionResponse::from_success_outputs(outputs, vec![]);
        response.validate_against(&request).unwrap();
        let cold: ProgramExecutionResponse =
            serde_json::from_slice(&serde_json::to_vec(&response).unwrap()).unwrap();
        assert_eq!(cold, response);
        cold.validate_against(&request).unwrap();
        request.requested_outputs = 1;
        request.recipe.outputs.requested_outputs = 1;
        request.artifact = ProgramArtifact::materialize(
            &request.recipe,
            ExecutableForm::InterpreterBytecodeV1,
            b"program".to_vec(),
        )
        .unwrap();
        assert!(cold.validate_against(&request).is_err());
    }
    let missing = ProgramExecutionResponse::from_success_outputs(
        vec![ValuePayload::Object(Box::new(result_reference(b"root")))],
        vec![],
    );
    assert!(missing.validate_against(&request()).is_err());
}

#[test]
fn externalized_response_requires_a_complete_unique_result_inventory() {
    let request = request();
    let root = result_reference(b"root");
    let response = ProgramExecutionResponse::ExternalizedSuccess {
        outputs: vec![ValuePayload::Object(Box::new(root.clone()))],
        result_objects: vec![root.clone(), result_reference(b"chunk")],
    };
    response.validate_against(&request).unwrap();

    let missing_root = ProgramExecutionResponse::ExternalizedSuccess {
        outputs: vec![ValuePayload::Object(Box::new(root.clone()))],
        result_objects: vec![result_reference(b"chunk")],
    };
    assert!(missing_root.validate_against(&request).is_err());

    let duplicate = ProgramExecutionResponse::ExternalizedSuccess {
        outputs: vec![ValuePayload::Object(Box::new(root.clone()))],
        result_objects: vec![root.clone(), root],
    };
    assert!(duplicate.validate_against(&request).is_err());
}
