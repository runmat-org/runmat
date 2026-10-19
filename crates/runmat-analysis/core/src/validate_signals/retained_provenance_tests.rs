use super::*;

#[test]
fn retained_provenance_requires_canonical_original_source_identity() {
    for source_digest in ["", "external.csv", &"A".repeat(64), &"a".repeat(63)] {
        assert!(validate_provenance(&SignalProvenance::Retained {
            source_digest: source_digest.into(),
        })
        .is_err());
    }
    let provenance = SignalProvenance::Retained {
        source_digest: format!("sha256:{}", "a1".repeat(32)),
    };
    validate_provenance(&provenance).unwrap();
    let wire = serde_json::to_vec(&provenance).unwrap();
    let recovered: SignalProvenance = serde_json::from_slice(&wire).unwrap();
    assert_eq!(recovered, provenance);
    // Ordinary imports remain distinct; their source names are not retained producer authority.
    validate_provenance(&SignalProvenance::Imported {
        source_digest: "external.csv".into(),
    })
    .unwrap();
}
