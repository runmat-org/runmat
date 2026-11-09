use super::*;
use runmat_execution::schema::VALUE_PAYLOAD_SCHEMA_VERSION;

fn reference(label: &[u8], bytes: u64) -> ValueRef {
    let digest = Digest::sha256(label);
    let access = ArtifactAccess {
        authorization_scope: "phase-workspace".into(),
        encryption_context: digest,
    };
    ValueRef {
        schema_version: VALUE_PAYLOAD_SCHEMA_VERSION,
        id: access.value_id(digest),
        logical_digest: digest,
        encoded_length: bytes,
        media_type: "application/octet-stream".into(),
        value_schema: "test/v1".into(),
        encryption_context: digest,
        kind: ValueRefKind::ResultObject,
        authorization_scope: access.authorization_scope,
        resident_fence: None,
    }
}

#[test]
fn sequential_workspace_reuses_phase_peak_without_dropping_cross_phase_authority() {
    let first = vec![reference(b"first", 1_000), reference(b"second", 2_000)];
    let second = vec![reference(b"third", 4_000)];
    let maximum = import_workspace(&first)
        .unwrap()
        .max(import_workspace(&second).unwrap());
    let peak = import_workspace_phases(&[&first, &second, &first]).unwrap();
    assert_eq!(peak, maximum);
    let simultaneous = first.iter().chain(&second).cloned().collect::<Vec<_>>();
    assert!(peak < import_workspace(&simultaneous).unwrap());
    let mut substituted = first.clone();
    substituted[0].encoded_length += 1;
    assert!(import_workspace_phases(&[&first, &substituted]).is_err());
}

#[test]
fn sequential_workspace_rejects_overflow_in_any_phase() {
    let ordinary = vec![reference(b"ordinary", 100)];
    let impossible = vec![reference(b"overflow", u64::MAX)];
    assert!(import_workspace_phases(&[&ordinary, &impossible]).is_err());
}

// Boundary: a study completion larger than the former fixed 256 MiB contract ceiling encodes when
// the run's admitted inventory holds it, and is refused when the admitted inventory does not.
#[test]
fn completion_codec_ceiling_follows_the_admitted_inventory() {
    let payload = vec![7_u8; 4096];
    let encode = |budget: u64| {
        runmat_canonical_codec::encode_contract(
            b"completion-limit-test\0",
            "completion-limit-test",
            &payload,
            completion_codec_limits(ObjectInventoryLimits::admitted(budget)),
        )
    };
    assert!(encode(1024).is_err());
    let bytes = encode(1 << 20).unwrap();
    assert!(bytes.len() > 4096);
    let admitted = ObjectInventoryLimits::admitted(512 * 1024 * 1024);
    assert_eq!(
        completion_codec_limits(admitted).maximum_encoded_bytes,
        512 * 1024 * 1024
    );
}
