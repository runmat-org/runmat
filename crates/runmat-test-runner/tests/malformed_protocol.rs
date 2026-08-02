use runmat_test::protocol::ProtocolLimits;
use runmat_test_runner::worker::{decode_frame, decode_request_frame};

#[test]
fn framing_rejects_missing_mismatched_and_invalid_payloads() {
    let limits = ProtocolLimits::default();
    assert!(decode_frame(&[], limits).is_err());
    assert!(decode_request_frame(&[0, 0, 0, 4, b'{'], limits).is_err());
    assert!(decode_frame(&[0, 0, 0, 1, 0xff], limits).is_err());
}
