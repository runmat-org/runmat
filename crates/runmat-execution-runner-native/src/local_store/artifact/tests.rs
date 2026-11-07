use super::*;
#[cfg(unix)]
use std::fs;
use std::sync::{Arc, Barrier};

#[test]
fn concurrent_different_publishers_preserve_one_complete_program() {
    let directory = tempfile::tempdir().unwrap();
    let store = Arc::new(ArtifactStore::new(directory.path().join("artifacts")).unwrap());
    let id = ArtifactId::derive(&[b"competing-programs"]);
    let start = Arc::new(Barrier::new(3));
    let first = vec![0x37; 256 * 1024];
    let second = vec![0xb2; 256 * 1024];
    let handles = [first.clone(), second.clone()].map(|bytes| {
        let store = Arc::clone(&store);
        let start = Arc::clone(&start);
        std::thread::spawn(move || {
            start.wait();
            let outcome = store.put(id, &bytes);
            (bytes, outcome)
        })
    });
    start.wait();
    let outcomes = handles.map(|handle| handle.join().unwrap());
    assert_eq!(
        outcomes.iter().filter(|(_, result)| result.is_ok()).count(),
        1
    );
    let winner = outcomes.iter().find(|(_, result)| result.is_ok()).unwrap();
    assert_eq!(store.get(id, 1024 * 1024).unwrap(), winner.0);
    let loser = outcomes.iter().find(|(_, result)| result.is_err()).unwrap();
    assert!(loser
        .1
        .as_ref()
        .unwrap_err()
        .to_string()
        .contains("identity collision"));
    store.put(id, &winner.0).unwrap();
    assert!(store.put(id, &loser.0).is_err());
    assert_eq!(store.get(id, 1024 * 1024).unwrap(), winner.0);
}

#[test]
fn concurrent_identical_publishers_are_idempotent() {
    let directory = tempfile::tempdir().unwrap();
    let store = Arc::new(ArtifactStore::new(directory.path().join("artifacts")).unwrap());
    let id = ArtifactId::derive(&[b"same-program"]);
    let start = Arc::new(Barrier::new(3));
    let handles = [0, 1].map(|_| {
        let store = Arc::clone(&store);
        let start = Arc::clone(&start);
        std::thread::spawn(move || {
            start.wait();
            store.put(id, b"one immutable program")
        })
    });
    start.wait();
    for handle in handles {
        handle.join().unwrap().unwrap();
    }
    assert_eq!(
        store.get(id, 1024 * 1024).unwrap(),
        b"one immutable program"
    );
}

#[cfg(unix)]
#[test]
fn publication_rejects_a_symlink_without_modifying_its_target() {
    let directory = tempfile::tempdir().unwrap();
    let store = ArtifactStore::new(directory.path().join("artifacts")).unwrap();
    let id = ArtifactId::derive(&[b"symlink-program"]);
    let target = directory.path().join("other");
    fs::write(&target, b"original").unwrap();
    std::os::unix::fs::symlink(&target, store.root.join(id.to_string())).unwrap();
    assert!(store.put(id, b"replacement").is_err());
    assert_eq!(fs::read(&target).unwrap(), b"original");
}

#[test]
fn artifact_reads_reject_original_oversize_before_loading_and_preserve_content() {
    let directory = tempfile::tempdir().unwrap();
    let store = ArtifactStore::new(directory.path().join("artifacts")).unwrap();
    let id = ArtifactId::derive(&[b"bounded-native-artifact"]);
    store.put(id, b"original").unwrap();
    assert!(store.get(id, 7).is_err());
    assert_eq!(store.get(id, 8).unwrap(), b"original");
    let path = store.root.join(id.to_string());
    std::fs::File::create(&path)
        .unwrap()
        .set_len(1 << 30)
        .unwrap();
    assert!(store.get(id, 1024).is_err());
    assert_eq!(std::fs::metadata(path).unwrap().len(), 1 << 30);
}
