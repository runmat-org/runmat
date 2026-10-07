//! Original durable identity and owned cold schema survive borrowed serialization.
use super::*;

#[test]
fn borrowed_program_identity_preserves_original_wire_and_omits_only_idempotency() {
    let context = super::super::DurableExecutionContext {
        scope_id: runmat_execution::ExecutionScopeId::derive(&[b"wire-scope"]),
        pool_id: runmat_execution::PoolId::derive(&[b"wire-pool"]),
    };
    let original = original_program_submission(context);
    let mut stable = original.clone();
    stable.idempotency_key = None;
    let mut expected = b"runmat-local-program-batch-request-v1\0".to_vec();
    expected.extend(
        runmat_canonical_codec::encode_json_bounded(
            &stable,
            u64::from(super::super::protocol::SUPERVISOR_MAX_MESSAGE_BYTES),
        )
        .unwrap(),
    );
    assert_eq!(
        original.request_digest().unwrap(),
        Digest::sha256(&expected)
    );
    let encoded = serde_json::to_vec(&original).unwrap();
    let cold: ProgramBatchSubmission = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(cold, original);
    let mut different_key = original.clone();
    different_key.idempotency_key = Some("different caller key".into());
    assert_eq!(
        different_key.request_digest().unwrap(),
        original.request_digest().unwrap()
    );
    assert_ne!(
        different_key.idempotency_digest(),
        original.idempotency_digest()
    );
    different_key.retention_millis += 1;
    assert_ne!(
        different_key.request_digest().unwrap(),
        original.request_digest().unwrap()
    );
}

#[test]
fn durable_console_controls_cannot_be_omitted_or_substituted() {
    let context = super::super::DurableExecutionContext {
        scope_id: runmat_execution::ExecutionScopeId::derive(&[b"console-scope"]),
        pool_id: runmat_execution::PoolId::derive(&[b"console-pool"]),
    };
    let original = original_program_submission(context);
    original.validate().unwrap();
    let mut wire = serde_json::to_value(&original).unwrap();
    wire.as_object_mut().unwrap().remove("console");
    assert!(serde_json::from_value::<ProgramBatchSubmission>(wire).is_err());
    let mut substituted = original.clone();
    substituted.task.request.console = Some(runmat_execution::console::ConsoleLimits {
        max_records: 4,
        max_text_bytes: 100,
    });
    let error = substituted.validate().unwrap_err().to_string();
    assert!(
        error.contains("differs from its original program invocation"),
        "{error}"
    );
    assert_ne!(
        original.request_digest().unwrap(),
        substituted.request_digest().unwrap()
    );
}
