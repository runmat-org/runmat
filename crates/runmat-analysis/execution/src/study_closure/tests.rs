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
