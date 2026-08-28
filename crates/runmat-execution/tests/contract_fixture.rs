use runmat_execution::identity::TaskId;
use runmat_execution::protocol::{Envelope, ProtocolLimits, ProtocolVersion};
use runmat_execution::{Digest, DomainContribution, ProgramEnvironment, ProgramRevision};
use serde::Deserialize;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test;

#[derive(Deserialize)]
struct Fixture {
    schema: String,
    program_revision_cbor_hex: String,
    program_revision_identity: String,
    envelope_cbor_hex: String,
    task_id: String,
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn native_and_wasm_match_the_shared_execution_fixture() {
    let fixture: Fixture =
        serde_json::from_str(include_str!("fixtures/execution_contract_v1.json")).unwrap();
    assert_eq!(fixture.schema, "runmat-execution-contract-fixture/v1");

    let revision = ProgramRevision::new(
        Digest::sha256(b"graph"),
        Digest::sha256(b"sources"),
        ProgramEnvironment::new(
            1,
            1,
            Digest::sha256(b"runtime"),
            Digest::sha256(b"catalog"),
            "matlab",
        )
        .unwrap(),
    )
    .unwrap()
    .with_domain_contribution(
        DomainContribution::new("runmat.test.config", Digest::sha256(b"tests")).unwrap(),
    )
    .unwrap();
    assert_eq!(
        hex(&revision.canonical_bytes().unwrap()),
        fixture.program_revision_cbor_hex
    );
    assert_eq!(
        revision.canonical_identity(),
        fixture.program_revision_identity
    );

    let envelope = Envelope {
        version: ProtocolVersion::V1,
        message_kind: 7,
        flags: 0,
        sequence: 42,
        payload: vec![1, 2, 3],
    };
    assert_eq!(
        hex(&envelope.encode(ProtocolLimits::default()).unwrap()),
        fixture.envelope_cbor_hex
    );

    let task = TaskId::derive(&[b"fixture"]);
    assert_eq!(task.to_string(), fixture.task_id);
    assert_eq!(fixture.task_id.parse::<TaskId>().unwrap(), task);
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}
