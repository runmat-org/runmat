use runmat_execution::identity::ValueId;
use runmat_execution::schema::VALUE_PAYLOAD_SCHEMA_VERSION;
use runmat_execution::value::{ValueRef, ValueRefKind};
use runmat_execution::Digest;

use super::validate_declared_roots;

fn reference(identity: &[u8]) -> ValueRef {
    ValueRef {
        schema_version: VALUE_PAYLOAD_SCHEMA_VERSION,
        id: ValueId::derive(&[b"exact-study-publication-root", identity]),
        logical_digest: Digest::sha256(identity),
        encoded_length: identity.len() as u64,
        media_type: "application/vnd.runmat.test".into(),
        value_schema: "runmat.test".into(),
        encryption_context: Digest::sha256(b"exact-study-publication-root-context"),
        kind: ValueRefKind::ResultObject,
        authorization_scope: "exact-study-publication-test".into(),
        resident_fence: None,
    }
}

#[test]
fn publication_baseline_and_terminal_usage_require_distinct_declared_roots() {
    let baseline = reference(b"publication-baseline-usage");
    let terminal = reference(b"terminal-resource-usage");
    let objects = vec![baseline.clone(), terminal.clone()];

    validate_declared_roots(&[&baseline, &terminal], &objects).unwrap();
    assert!(validate_declared_roots(&[&baseline, &baseline], &objects).is_err());
}

#[test]
fn publication_usage_roots_must_both_be_present_in_the_closure() {
    let baseline = reference(b"publication-baseline-usage");
    let terminal = reference(b"terminal-resource-usage");

    assert!(
        validate_declared_roots(&[&baseline, &terminal], std::slice::from_ref(&baseline),).is_err()
    );
}
