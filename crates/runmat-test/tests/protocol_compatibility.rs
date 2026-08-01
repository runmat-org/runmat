mod common;

use common::run_id;
use runmat_test::protocol::{
    decode_request, encode_request, negotiate, ProtocolHandshake, ProtocolLimits, WorkerCapability,
    WorkerRequest,
};
use runmat_test::TestDomainError;

#[test]
fn protocol_negotiates_the_stricter_bound_and_round_trips() {
    let local = ProtocolHandshake::current("native", vec![WorkerCapability::StrongIsolation]);
    let mut remote =
        ProtocolHandshake::current("browser", vec![WorkerCapability::SessionIsolation]);
    remote.limits.max_message_bytes = 4096;
    let limits = negotiate(&local, &remote).unwrap();
    assert_eq!(limits.max_message_bytes, 4096);

    let request = WorkerRequest::Cancel {
        run_id: run_id(),
        reason: "stop".into(),
    };
    let bytes = encode_request(&request, limits).unwrap();
    assert_eq!(decode_request(&bytes, limits).unwrap(), request);
}

#[test]
fn protocol_rejects_incompatible_versions_and_oversized_payloads() {
    let local = ProtocolHandshake::current("local", Vec::new());
    let mut future = local.clone();
    future.protocol_version += 1;
    assert!(matches!(
        negotiate(&local, &future),
        Err(TestDomainError::IncompatibleProtocol { .. })
    ));

    let limits = ProtocolLimits {
        max_message_bytes: 8,
        ..ProtocolLimits::default()
    };
    assert!(matches!(
        encode_request(&WorkerRequest::Shutdown, limits),
        Err(TestDomainError::ProtocolPayloadTooLarge { .. })
    ));
}
