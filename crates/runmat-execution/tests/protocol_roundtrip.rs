use runmat_execution::protocol::{
    negotiate, Envelope, ProtocolHello, ProtocolLimits, ProtocolVersion,
};

#[test]
fn envelope_roundtrips_with_stable_bytes() {
    let envelope = Envelope {
        version: ProtocolVersion::V1,
        message_kind: 7,
        flags: 0,
        sequence: 42,
        payload: vec![1, 2, 3],
    };
    let encoded = envelope.encode(ProtocolLimits::default()).unwrap();
    assert_eq!(
        encoded,
        vec![
            0xa6, 0x00, 0x01, 0x01, 0x00, 0x02, 0x07, 0x03, 0x00, 0x04, 0x18, 0x2a, 0x05, 0x43,
            0x01, 0x02, 0x03,
        ]
    );
    assert_eq!(
        Envelope::decode(&encoded, ProtocolLimits::default()).unwrap(),
        envelope
    );
}

#[test]
fn negotiation_selects_the_highest_shared_major_and_lower_minor() {
    let mut left = ProtocolHello::v1("left", []);
    left.supported_majors.push(2);
    left.maximum_minor_by_major.push((2, 3));
    let mut right = ProtocolHello::v1("right", []);
    right.supported_majors.push(2);
    right.maximum_minor_by_major.push((2, 1));
    assert_eq!(
        negotiate(&left, &right).unwrap(),
        ProtocolVersion { major: 2, minor: 1 }
    );
}

#[test]
fn malformed_and_oversized_envelopes_fail_before_payload_use() {
    let limits = ProtocolLimits {
        max_message_bytes: 32,
        max_payload_bytes: 2,
        ..ProtocolLimits::default()
    };
    let envelope = Envelope {
        version: ProtocolVersion::V1,
        message_kind: 1,
        flags: 0,
        sequence: 0,
        payload: vec![1, 2, 3],
    };
    assert!(envelope.encode(limits).is_err());
    assert!(Envelope::decode(&[0xbf, 0xff], limits).is_err());
}
