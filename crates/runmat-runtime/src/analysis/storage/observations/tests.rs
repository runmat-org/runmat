use super::*;

#[test]
fn pruning_observation_attachment_preserves_other_runs_and_shared_objects() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    std::fs::create_dir(root.join("observations")).unwrap();
    std::fs::create_dir(root.join("objects")).unwrap();
    let selected = root.join("observations/selected.json");
    let retained = root.join("observations/retained.json");
    let shared = root.join("objects/shared");
    std::fs::write(&selected, b"selected attachment").unwrap();
    std::fs::write(&retained, b"retained attachment").unwrap();
    std::fs::write(&shared, b"shared numerical bytes").unwrap();

    prune_attachment(root, "selected").unwrap();
    assert!(!selected.exists());
    assert_eq!(std::fs::read(retained).unwrap(), b"retained attachment");
    assert_eq!(std::fs::read(shared).unwrap(), b"shared numerical bytes");
    // Repeated cleanup is harmless after the parent run has already been pruned.
    prune_attachment(root, "selected").unwrap();
}

#[test]
fn pruning_observation_attachment_rejects_unowned_targets() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    std::fs::create_dir(root.join("observations")).unwrap();
    let non_file = root.join("observations/selected.json");
    std::fs::create_dir(&non_file).unwrap();
    let marker = non_file.join("keep");
    std::fs::write(&marker, b"not an attachment").unwrap();

    assert!(prune_attachment(root, "selected").is_err());
    assert_eq!(std::fs::read(marker).unwrap(), b"not an attachment");
    assert!(prune_attachment(root, "../selected").is_err());
}
