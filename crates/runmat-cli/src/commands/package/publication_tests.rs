use super::publication_artifact::{collect_entries, normalized_link_target};
use super::publication_manifest::PreparedPublication;
use super::publication_retry::{load_or_create_encrypted, private_state_path};
use crate::cli::PackageInspectArgs;
use runmat_package_cache::PublicationPolicy;
use runmat_package_cache_native::registry::RecipientKeyPair;
use std::path::Path;

#[test]
fn inspection_is_deterministic_and_excludes_private_retry_state() {
    let project = fixture();
    let args = PackageInspectArgs {
        manifest_path: project.path().join("runmat.toml"),
        allow_native: false,
        json: false,
    };
    let first = PreparedPublication::build(&args).unwrap();
    let second = PreparedPublication::build(&args).unwrap();
    assert_eq!(first.bundle.artifact_bytes, second.bundle.artifact_bytes);
    assert_eq!(first.release_manifest, second.release_manifest);
    assert!(first
        .bundle
        .inventory
        .entries
        .iter()
        .all(|entry| !entry.path.as_str().starts_with(".runmat")));
}

#[cfg(unix)]
#[test]
fn inspection_resolves_relative_symlinks_against_their_parent() {
    use std::os::unix::fs::symlink;

    let project = fixture();
    std::fs::create_dir_all(project.path().join("src/nested")).unwrap();
    symlink("../tool.m", project.path().join("src/nested/entry.m")).unwrap();
    let prepared = PreparedPublication::build(&PackageInspectArgs {
        manifest_path: project.path().join("runmat.toml"),
        allow_native: false,
        json: false,
    })
    .unwrap();
    let entry = prepared
        .bundle
        .inventory
        .entries
        .iter()
        .find(|entry| entry.path.as_str() == "src/nested/entry.m")
        .unwrap();
    assert_eq!(
        entry.link_target.as_ref().map(ToString::to_string),
        Some("src/tool.m".to_string())
    );
}

#[test]
fn publication_symlinks_cannot_escape_the_package_root() {
    assert!(normalized_link_target("src/entry.m", Path::new("../../secret")).is_err());
    assert!(normalized_link_target("entry.m", Path::new("/secret")).is_err());
}

#[cfg(unix)]
#[test]
fn collector_prunes_builtin_trees_and_does_not_read_excluded_files() {
    use std::os::unix::fs::PermissionsExt as _;

    let root = tempfile::tempdir().unwrap();
    std::fs::create_dir(root.path().join(".git")).unwrap();
    std::fs::write(root.path().join(".git/config"), b"secret").unwrap();
    std::fs::write(root.path().join("ignored.secret"), b"secret").unwrap();
    std::fs::write(root.path().join("source.m"), b"x = 1;\n").unwrap();
    std::fs::set_permissions(
        root.path().join("ignored.secret"),
        std::fs::Permissions::from_mode(0o000),
    )
    .unwrap();
    let policy = PublicationPolicy::new(&[], &["ignored.secret".to_string()], false).unwrap();
    let entries = collect_entries(root.path(), &policy).unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].path.as_str(), "source.m");
}

#[test]
fn encrypted_retry_state_reuses_the_exact_ciphertext_and_rejects_input_drift() {
    let project = fixture();
    let args = PackageInspectArgs {
        manifest_path: project.path().join("runmat.toml"),
        allow_native: false,
        json: false,
    };
    let prepared = PreparedPublication::build(&args).unwrap();
    let recipient = RecipientKeyPair::from_secret_bytes("pkr_test", [7; 32])
        .unwrap()
        .public_key()
        .unwrap();
    let path = private_state_path(
        project.path(),
        "pkg_test",
        &prepared.release_manifest.version,
    )
    .unwrap();
    let first = load_or_create_encrypted(
        &path,
        "https://packages.runmat.test",
        &prepared,
        1,
        std::slice::from_ref(&recipient),
    )
    .unwrap();
    let second = load_or_create_encrypted(
        &path,
        "https://packages.runmat.test",
        &prepared,
        1,
        std::slice::from_ref(&recipient),
    )
    .unwrap();
    assert_eq!(first.ciphertext().unwrap(), second.ciphertext().unwrap());
    assert_eq!(first.artifact_digest, second.artifact_digest);
    assert!(load_or_create_encrypted(
        &path,
        "https://packages.runmat.test",
        &prepared,
        2,
        &[recipient],
    )
    .is_err());
}

#[test]
fn encrypted_retry_state_path_rejects_path_like_package_ids() {
    let version = "1.0.0".parse().unwrap();
    for package_id in ["", "../other", "nested/package", "package key"] {
        assert!(private_state_path(Path::new("/project"), package_id, &version).is_err());
    }
}

fn fixture() -> tempfile::TempDir {
    let project = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(project.path().join("src")).unwrap();
    std::fs::create_dir_all(project.path().join(".runmat/publications")).unwrap();
    std::fs::write(
        project.path().join("runmat.toml"),
        r#"[package]
name = "tools"
organization = "acme"
version = "1.2.3"

[sources]
roots = ["src"]

[publish]
license = "MIT"
"#,
    )
    .unwrap();
    std::fs::write(
        project.path().join("src/tool.m"),
        b"function y = tool(x)\ny = x;\nend\n",
    )
    .unwrap();
    std::fs::write(
        project.path().join(".runmat/publications/stale.json"),
        b"ciphertext",
    )
    .unwrap();
    project
}
