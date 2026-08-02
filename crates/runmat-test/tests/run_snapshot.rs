use runmat_test::discovery::{
    FrozenTestRunSnapshot, RunSourceOrigin, SavedRunSource, UnsavedRunBuffer,
};

fn saved(path: &str, content: &str) -> SavedRunSource {
    SavedRunSource {
        owner_identity: "registry:acme/pkg@1.0.0#tree".into(),
        relative_path: path.into(),
        content: content.into(),
    }
}

#[test]
fn unsaved_buffers_form_an_explicit_immutable_overlay() {
    let snapshot = FrozenTestRunSnapshot::freeze(
        "sha256:graph",
        "sha256:base-sources",
        1,
        2,
        "sha256:config",
        vec![saved("tests/b.m", "b = 1;"), saved("tests/a.m", "a = 1;")],
        vec![UnsavedRunBuffer {
            owner_identity: "registry:acme/pkg@1.0.0#tree".into(),
            relative_path: "tests/a.m".into(),
            content: "a = 2;".into(),
        }],
    )
    .unwrap();

    snapshot.validate().unwrap();
    assert_eq!(snapshot.sources[0].relative_path, "tests/a.m");
    assert_eq!(snapshot.sources[0].content, "a = 2;");
    assert_eq!(snapshot.sources[0].origin, RunSourceOrigin::UnsavedBuffer);
    assert_eq!(snapshot.sources[1].origin, RunSourceOrigin::Saved);

    let encoded = serde_json::to_vec(&snapshot).unwrap();
    let decoded: FrozenTestRunSnapshot = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(decoded, snapshot);
}

#[test]
fn source_revision_is_mount_and_input_order_independent_but_content_bound() {
    let first = FrozenTestRunSnapshot::freeze(
        "sha256:graph",
        "sha256:base-sources",
        1,
        2,
        "sha256:config",
        vec![saved("tests/a.m", "a = 1;"), saved("tests/b.m", "b = 1;")],
        Vec::new(),
    )
    .unwrap();
    let reordered = FrozenTestRunSnapshot::freeze(
        "sha256:graph",
        "sha256:base-sources",
        1,
        2,
        "sha256:config",
        vec![saved("tests/b.m", "b = 1;"), saved("tests/a.m", "a = 1;")],
        Vec::new(),
    )
    .unwrap();
    let changed = FrozenTestRunSnapshot::freeze(
        "sha256:graph",
        "sha256:base-sources",
        1,
        2,
        "sha256:config",
        vec![saved("tests/a.m", "a = 2;"), saved("tests/b.m", "b = 1;")],
        Vec::new(),
    )
    .unwrap();

    assert_eq!(first, reordered);
    assert_ne!(
        first.program_revision.source_digest,
        changed.program_revision.source_digest
    );
}

#[test]
fn validation_rejects_mutated_or_non_relative_sources() {
    assert!(FrozenTestRunSnapshot::freeze(
        "sha256:graph",
        "sha256:base-sources",
        1,
        2,
        "sha256:config",
        vec![saved("../outside.m", "")],
        Vec::new(),
    )
    .is_err());

    let mut snapshot = FrozenTestRunSnapshot::freeze(
        "sha256:graph",
        "sha256:base-sources",
        1,
        2,
        "sha256:config",
        vec![saved("tests/a.m", "a = 1;")],
        Vec::new(),
    )
    .unwrap();
    snapshot.sources[0].content = "mutated".into();
    assert!(snapshot.validate().is_err());
}
