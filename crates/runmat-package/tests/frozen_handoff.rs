use runmat_package::{
    build_frozen_project, build_frozen_project_async, FrozenProjectHandoff,
    FrozenProjectHandoffError, FROZEN_PROJECT_HANDOFF_SCHEMA_VERSION,
};
use std::collections::BTreeSet;
use std::fs;
use tempfile::TempDir;

fn fixture() -> (TempDir, FrozenProjectHandoff) {
    let temp = TempDir::new().unwrap();
    fs::create_dir_all(temp.path().join("src")).unwrap();
    fs::write(
        temp.path().join("runmat.toml"),
        r#"
[package]
name = "handoff"

[sources]
roots = ["src"]
"#,
    )
    .unwrap();
    fs::write(temp.path().join("src/main.m"), "value = 1;\n").unwrap();
    let project = build_frozen_project(&temp.path().join("runmat.toml"), BTreeSet::new()).unwrap();
    (temp, FrozenProjectHandoff::new(project))
}

#[test]
fn frozen_handoff_round_trips_and_preserves_revision() {
    let (_temp, handoff) = fixture();
    handoff.validate().unwrap();
    let revision = handoff.revision();
    let json = serde_json::to_vec(&handoff).unwrap();
    let decoded: FrozenProjectHandoff = serde_json::from_slice(&json).unwrap();
    decoded.validate().unwrap();
    assert_eq!(decoded, handoff);
    assert_eq!(decoded.revision(), revision);
    assert_eq!(
        decoded.schema_version,
        FROZEN_PROJECT_HANDOFF_SCHEMA_VERSION
    );
}

#[test]
fn frozen_handoff_rejects_schema_and_digest_tampering() {
    let (_temp, mut handoff) = fixture();
    handoff.schema_version += 1;
    assert!(matches!(
        handoff.validate(),
        Err(FrozenProjectHandoffError::UnsupportedSchema { .. })
    ));

    let (_temp, mut handoff) = fixture();
    handoff.project.graph.graph_digest =
        runmat_package::ContentDigest::sha256("tampered graph digest");
    assert!(matches!(
        handoff.validate(),
        Err(FrozenProjectHandoffError::Graph(_))
    ));

    let (_temp, mut handoff) = fixture();
    handoff.project.sources.revision =
        runmat_package::ContentDigest::sha256("tampered source revision");
    assert!(matches!(
        handoff.validate(),
        Err(FrozenProjectHandoffError::SourceCatalog(_))
    ));
}

#[test]
fn native_and_async_hosts_emit_identical_handoffs() {
    let (temp, native) = fixture();
    let asynchronous = futures::executor::block_on(build_frozen_project_async(
        &temp.path().join("runmat.toml"),
        BTreeSet::new(),
    ))
    .map(FrozenProjectHandoff::new)
    .unwrap();

    assert_eq!(asynchronous.revision(), native.revision());
    assert_eq!(
        serde_json::to_vec(&asynchronous).unwrap(),
        serde_json::to_vec(&native).unwrap()
    );
}
