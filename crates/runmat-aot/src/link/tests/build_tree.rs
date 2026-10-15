use super::*;

#[test]
fn exact_build_tree_staging_preserves_layout_and_private_lease_lifetime() {
    let tree = NativeBuildTree::new(vec![
        TreeInventoryEntry::directory("Frameworks"),
        TreeInventoryEntry::directory("Frameworks/Foo.framework"),
        TreeInventoryEntry::file(
            "Frameworks/Foo.framework/Foo",
            b"link-image".to_vec(),
            false,
        ),
        TreeInventoryEntry::directory("Frameworks/Foo.framework/Resources"),
        TreeInventoryEntry::file(
            "Frameworks/Foo.framework/Resources/data",
            b"resource".to_vec(),
            false,
        ),
    ]);
    let bytes = tree.canonical_bytes().unwrap();
    let identity = Digest::sha256(&bytes);
    assert!(NativeBuildTree::decode(&bytes, Digest::sha256("other tree")).is_err());
    let verified = NativeBuildTree::decode(&bytes, identity).unwrap();
    let lease = tempfile::tempdir().unwrap();
    let root = lease.path().join("sysroot");
    verified.stage(&root).unwrap();
    assert_eq!(
        std::fs::read(root.join("Frameworks/Foo.framework/Resources/data")).unwrap(),
        b"resource"
    );
    verified.verify_staged(&root).unwrap();
    assert!(
        verified.stage(&root).is_err(),
        "staging must not overwrite a leased tree"
    );
    let path = lease.path().to_path_buf();
    drop(lease);
    assert!(
        !path.exists(),
        "the entire private closure is retired on drop"
    );
}

#[test]
fn build_tree_reuses_archive_path_collision_and_link_rejection() {
    for entries in [
        vec![TreeInventoryEntry::file("../escape", vec![1], false)],
        vec![
            TreeInventoryEntry::file("A", vec![1], false),
            TreeInventoryEntry::file("a", vec![1], false),
        ],
        vec![
            TreeInventoryEntry::file("a", vec![1], false),
            TreeInventoryEntry::file("a/b", vec![1], false),
        ],
        vec![TreeInventoryEntry::symlink("a", "../../outside")],
    ] {
        assert!(NativeBuildTree::new(entries).canonical_bytes().is_err());
    }
    let old = br#"{"schema_version":0,"entries":"old shape"}"#;
    assert!(NativeBuildTree::decode(old, Digest::sha256(old))
        .unwrap_err()
        .to_string()
        .contains("unsupported build tree schema"));
}

#[cfg(unix)]
#[test]
fn staged_build_links_cannot_redirect_writes_and_mutated_bytes_fail_revalidation() {
    let tree = NativeBuildTree::new(vec![
        TreeInventoryEntry::symlink("alias", "data"),
        TreeInventoryEntry::file("data", b"exact".to_vec(), false),
    ]);
    let lease = tempfile::tempdir().unwrap();
    let root = lease.path().join("tree");
    tree.stage(&root).unwrap();
    assert_eq!(std::fs::read(root.join("alias")).unwrap(), b"exact");
    use std::os::unix::fs::PermissionsExt;
    let data = root.join("data");
    std::fs::set_permissions(&data, std::fs::Permissions::from_mode(0o600)).unwrap();
    std::fs::write(data, b"changed").unwrap();
    assert!(tree.verify_staged(&root).is_err());
}
