use runmat_execution::{Digest, DomainContribution, ProgramEnvironment, ProgramRevision};

fn revision() -> ProgramRevision {
    ProgramRevision::new(
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
    .unwrap()
}

#[test]
fn program_revision_vector_is_stable() {
    let revision = revision();
    assert_eq!(
        hex(&revision.canonical_bytes().unwrap()),
        "a90001015820eef93e1d14482804277fca0172464032d1a4fdbcc338524059fa1e861454ad4d025820878a52fc5ff6a57d50b7b870aa51637a3dfd38fc22352a39f95a3c292eb976d503010401055820d92c6a81b2ff50096bcda80885427d1f59a25b5f483f7055523504925d16ab23065820652f55016243bf1b9f1bbea46d5749ef892dbe394e46de9d66ab1aacf0b4af5707666d61746c61620881827272756e6d61742e746573742e636f6e666967582059830ebc3a4184110566bf1a290d08473dfdcbd492ce498b14cd1a5e2fa2e441"
    );
    assert_eq!(
        revision.canonical_identity(),
        "sha256:251faff061b8572ab58377fd7ef16d29d949af253b251ff9a80db15d106900bd"
    );
}

#[test]
fn deserialization_cannot_bypass_revision_validation() {
    let mut value = serde_json::to_value(revision()).unwrap();
    value["semantic_schema"] = serde_json::json!(0);
    assert!(serde_json::from_value::<ProgramRevision>(value).is_err());

    let mut value = serde_json::to_value(revision()).unwrap();
    value["domain_contributions"] = serde_json::json!([
        {
            "name": "z",
            "digest": Digest::sha256(b"z").to_string()
        },
        {
            "name": "a",
            "digest": Digest::sha256(b"a").to_string()
        }
    ]);
    assert!(serde_json::from_value::<ProgramRevision>(value).is_err());
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}
