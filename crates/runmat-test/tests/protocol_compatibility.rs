mod common;

use common::run_id;
use runmat_test::protocol::{
    decode_request, decode_response, encode_request, encode_response, negotiate, ProtocolHandshake,
    ProtocolLimits, WorkerCapability, WorkerRequest, WorkerResponse,
};
use runmat_test::TestDomainError;
use runmat_test::{
    coverage::{CoverageFragment, CoverageMetric, CoverageSite},
    result::{AttemptResult, ResultState},
};
use std::collections::BTreeMap;

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

#[test]
fn protocol_round_trips_full_width_coverage_counter_keys() {
    let counter_key = 1_026_422_203_302_429_191;
    let response = WorkerResponse::Completed {
        result: AttemptResult {
            test_id: common::test_id("covered"),
            attempt: 1,
            state: ResultState::PASSED,
            diagnostics: Vec::new(),
            artifacts: Vec::new(),
            output: String::new(),
            abort_run: false,
        },
        coverage: vec![CoverageFragment {
            program_revision: "program".into(),
            plan_revision: "executable".into(),
            sites: vec![CoverageSite {
                id: "site".into(),
                counter_key,
                metric: CoverageMetric::Statement,
                owner_identity: "root".into(),
                relative_path: "covered.m".into(),
                semantic_path: "covered".into(),
                source_id: 0,
                start_byte: 0,
                end_byte: 1,
                start_line: 1,
                start_column: 1,
                end_line: 1,
                end_column: 2,
                instrumented: true,
                unsupported_reason: None,
            }],
            counts: BTreeMap::from([(counter_key, 1)]),
        }],
    };

    let encoded = encode_response(&response, ProtocolLimits::default()).unwrap();
    let json = String::from_utf8(encoded.clone()).unwrap();
    assert!(json.contains("\"counter_key\":\"1026422203302429191\""));
    assert_eq!(
        decode_response(&encoded, ProtocolLimits::default()).unwrap(),
        response
    );
}
