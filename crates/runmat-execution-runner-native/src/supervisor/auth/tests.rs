use super::*;
use std::sync::{Arc, Barrier};

#[test]
fn concurrent_token_creation_returns_one_complete_original_credential() {
    let directory = tempfile::tempdir().unwrap();
    let paths = SupervisorPaths::new(directory.path().join("authority")).unwrap();
    let barrier = Arc::new(Barrier::new(8));
    let threads: Vec<_> = (0..8)
        .map(|_| {
            let paths = paths.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                barrier.wait();
                load_or_create_token(&paths).unwrap()
            })
        })
        .collect();
    let tokens: Vec<_> = threads.into_iter().map(|t| t.join().unwrap()).collect();
    assert_eq!(tokens[0].len(), 64);
    assert!(tokens.iter().all(|token| token == &tokens[0]));
    assert_eq!(read_token(&paths.token).unwrap(), tokens[0]);
    assert_eq!(load_or_create_token(&paths).unwrap(), tokens[0]);
    assert_eq!(fs::read_dir(&paths.root).unwrap().count(), 1);
}

#[test]
fn oversized_or_malformed_original_token_is_never_replaced() {
    let directory = tempfile::tempdir().unwrap();
    let paths = SupervisorPaths::new(directory.path().join("authority")).unwrap();
    crate::supervisor::filesystem::secure_directory(&paths.root).unwrap();
    for bytes in [b"partial".to_vec(), vec![b'a'; 65]] {
        runmat_filesystem::NativeFsProvider
            .atomic_create_private_new(&paths.token, &bytes, 65)
            .unwrap();
        assert!(load_or_create_token(&paths).is_err());
        assert_eq!(fs::read(&paths.token).unwrap(), bytes);
        fs::remove_file(&paths.token).unwrap();
    }
}
