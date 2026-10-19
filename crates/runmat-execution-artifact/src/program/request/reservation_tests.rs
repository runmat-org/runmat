use crate::ReservedProgramExecutionRequest;
use runmat_execution::resource::ResourceRequest;
#[test]
fn reservation_handoff_keeps_original_program_identity_and_rejects_missing_or_invalid_capacity() {
    let program = super::tests::request();
    let digest = program.canonical_digest().unwrap();
    let resources = ResourceRequest {
        cpu_millicores: 1000,
        memory_bytes: 1024,
        scratch_bytes: 0,
        max_wall_millis: 1,
        max_artifact_bytes: 0,
        max_egress_bytes: 0,
        max_relay_bytes: 0,
        accelerators: vec![],
        required_capabilities: Default::default(),
    };
    let handoff = ReservedProgramExecutionRequest::new(program.clone(), resources.clone()).unwrap();
    let cold: ReservedProgramExecutionRequest =
        serde_json::from_slice(&serde_json::to_vec(&handoff).unwrap()).unwrap();
    cold.validate().unwrap();
    assert_eq!(cold.request(), &program);
    assert_eq!(cold.request().canonical_digest().unwrap(), digest);
    assert_eq!(cold.resources(), &resources);
    let mut encoded = serde_json::to_value(&handoff).unwrap();
    encoded.as_object_mut().unwrap().remove("resources");
    assert!(serde_json::from_value::<ReservedProgramExecutionRequest>(encoded).is_err());
    let mut invalid = resources;
    invalid.memory_bytes = 0;
    assert!(ReservedProgramExecutionRequest::new(program, invalid).is_err());
}
