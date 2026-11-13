use super::*;
use crate::{console::ConsoleLimits, identity::*, resource::ResourceRequest};

fn request() -> TaskRequest {
    let limits = ConsoleLimits {
        max_records: 4,
        max_text_bytes: 1024,
    };
    let environment =
        crate::EnvironmentSnapshot::new(vec![("TEST".into(), "original".into())]).unwrap();
    TaskRequest {
        schema_version: TASK_REQUEST_SCHEMA_VERSION,
        id: TaskId::derive(&[b"task"]),
        scope_id: ExecutionScopeId::derive(&[b"scope"]),
        pool_id: PoolId::derive(&[b"pool"]),
        executable: TaskExecutableIdentity::ProgramArtifact(ArtifactId::derive(&[b"program"])),
        callable: Callable::for_program(
            "test",
            &ProgramCallable::semantic(runmat_types::ProgramFunctionId(0), Some("entry".into())),
        ),
        // This contract-only fixture exercises invocation accounting, not
        // compiler issuance or admission to an actual execution host.
        host: crate::host::ExecutionHostRequirement::portable(
            crate::ProgramEnvironment::new(
                crate::schema::PROGRAM_SEMANTIC_SCHEMA_V2,
                crate::schema::PROGRAM_COMPILER_SCHEMA_V2,
                Digest::sha256(b"console-contract-test-runtime"),
                Digest::sha256(b"console-contract-test-catalog"),
                "runmat",
            )
            .unwrap(),
            runmat_types::CapabilitySet::default(),
            runmat_types::InteropManifest::empty(),
            std::collections::BTreeSet::from([
                crate::security::ExecutionTrustTier::CustomerTrusted,
            ]),
        )
        .unwrap(),
        invocation_context: ProgramInvocationContext::Direct,
        inputs: Vec::new(),
        resources: ResourceRequest {
            cpu_millicores: 1000,
            memory_bytes: crate::console::invocation_memory_bytes(&environment, Some(limits))
                .unwrap(),
            scratch_bytes: 0,
            max_wall_millis: 1000,
            max_artifact_bytes: limits.storage_profile().unwrap().max_encoded_bytes,
            max_egress_bytes: 0,
            max_relay_bytes: 0,
            accelerators: Vec::new(),
            required_capabilities: Default::default(),
        },
        environment,
        console: Some(limits),
        outputs: OutputContract {
            requested_outputs: 1,
            owned_work: None,
        },
        retry: RetryPolicy::Never,
        deadline_unix_millis: None,
    }
}

#[test]
fn original_console_memory_and_wire_components_reject_understated_reservations() {
    let task = request();
    task.validate_invocation_reservation().unwrap();
    let mut memory = task.clone();
    memory.resources.memory_bytes -= 1;
    assert!(memory.validate_invocation_reservation().is_err());
    let mut wire = task.clone();
    wire.resources.max_artifact_bytes -= 1;
    assert!(wire.validate_invocation_reservation().is_err());
    let cold: TaskRequest = serde_json::from_slice(&serde_json::to_vec(&task).unwrap()).unwrap();
    assert_eq!(cold, task);
    cold.validate_invocation_reservation().unwrap();
}

#[test]
fn missing_controls_do_not_decode_as_disabled_capture() {
    let mut wire = serde_json::to_value(request()).unwrap();
    wire.as_object_mut().unwrap().remove("console");
    assert!(serde_json::from_value::<TaskRequest>(wire).is_err());
}
