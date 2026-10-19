use super::*;
use std::fs::{self, File};
use tempfile::tempdir;

#[test]
fn identical_path_and_containing_directory_are_successful_noops() {
    let _lock = runmat_filesystem::provider_override_lock();
    let temp = tempdir().unwrap();
    let directory = temp.path().join("docs");
    fs::create_dir(&directory).unwrap();
    let source = directory.join("readme.txt");
    File::create(&source).unwrap();
    assert_eq!(
        evaluate(&[text(&source), text(&source)]).unwrap().status(),
        1.0
    );
    assert_eq!(
        evaluate(&[text(&source), text(&directory)])
            .unwrap()
            .status(),
        1.0
    );
    assert!(source.exists());
}

#[test]
fn force_preserves_self_through_canonical_path_aliases() {
    let _lock = runmat_filesystem::provider_override_lock();
    let temp = tempdir().unwrap();
    let source = temp.path().join("stable.txt");
    fs::write(&source, b"preserved").unwrap();
    let alias = temp.path().join(".").join("stable.txt");
    assert_eq!(
        evaluate(&[text(&source), text(&alias), Value::from("f")])
            .unwrap()
            .status(),
        1.0
    );
    assert_eq!(fs::read(&source).unwrap(), b"preserved");
    // Wildcard planning must protect aliases before any replacement is attempted.
    assert_eq!(
        evaluate(&[
            text(&temp.path().join("*.txt")),
            text(&temp.path().join(".")),
            Value::from("f")
        ])
        .unwrap()
        .status(),
        1.0
    );
    assert_eq!(fs::read(&source).unwrap(), b"preserved");
}

#[cfg(unix)]
#[test]
fn force_preserves_symbolic_alias_and_moves_dangling_source_link() {
    use std::os::unix::fs::symlink;
    let _lock = runmat_filesystem::provider_override_lock();
    let temp = tempdir().unwrap();
    let source = temp.path().join("source");
    let alias = temp.path().join("alias");
    fs::write(&source, b"preserved").unwrap();
    symlink(&source, &alias).unwrap();
    assert_eq!(
        evaluate(&[text(&source), text(&alias), Value::from("f")])
            .unwrap()
            .status(),
        1.0
    );
    assert_eq!(fs::read(&source).unwrap(), b"preserved");
    assert!(fs::symlink_metadata(&alias)
        .unwrap()
        .file_type()
        .is_symlink());
    let dangling = temp.path().join("dangling");
    let moved = temp.path().join("moved");
    let absent = temp.path().join("absent");
    symlink(&absent, &dangling).unwrap();
    assert_eq!(
        evaluate(&[text(&dangling), text(&moved)]).unwrap().status(),
        1.0
    );
    assert!(fs::symlink_metadata(&dangling).is_err());
    assert_eq!(fs::read_link(&moved).unwrap(), absent);
}

#[test]
fn rejects_descendant_target_before_force_removal() {
    let _lock = runmat_filesystem::provider_override_lock();
    let temp = tempdir().unwrap();
    let source = temp.path().join("tree");
    let destination = source.join("nested");
    let collision = destination.join("tree");
    fs::create_dir_all(&collision).unwrap();
    fs::write(collision.join("preserve"), b"payload").unwrap();
    assert_eq!(
        evaluate(&[text(&source), text(&destination), Value::from("f")])
            .unwrap()
            .status(),
        0.0
    );
    assert_eq!(fs::read(collision.join("preserve")).unwrap(), b"payload");
}
