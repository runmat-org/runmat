use runmat_package::{ContentDigest, NormalizedRelativePath};
use runmat_package_cache::{MountDescriptor, TreeEntry, TreeManifest};

fn path(value: &str) -> NormalizedRelativePath {
    NormalizedRelativePath::new(value).unwrap()
}

#[test]
fn tree_identity_is_canonical_and_mounts_are_immutable() {
    let digest = ContentDigest::sha256(b"file");
    let left = TreeManifest::new(vec![
        TreeEntry::file(path("src/b.m"), digest.clone(), 4, false),
        TreeEntry::file(path("src/a.m"), digest, 4, false),
        TreeEntry::directory(path("src")),
    ])
    .unwrap();
    let right = TreeManifest::new(left.entries.iter().rev().cloned().collect()).unwrap();
    assert_eq!(left, right);
    left.validate().unwrap();

    let mount = MountDescriptor::immutable(left.digest, path("packages/example"));
    assert!(mount.read_only);
}

#[test]
fn files_and_links_cannot_shadow_descendants() {
    let digest = ContentDigest::sha256(b"file");
    assert!(TreeManifest::new(vec![
        TreeEntry::file(path("src"), digest, 4, false),
        TreeEntry::directory(path("src/nested")),
    ])
    .is_err());
    assert!(TreeManifest::new(vec![
        TreeEntry::symlink(path("src"), path("elsewhere")),
        TreeEntry::directory(path("src/nested")),
    ])
    .is_err());
}
