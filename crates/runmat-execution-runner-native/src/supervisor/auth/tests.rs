use super::*;
use std::sync::{Arc, Barrier};

#[test]
fn concurrent_token_creation_returns_one_complete_original_credential() {
    let directory = tempfile::tempdir().unwrap();
    let paths = SupervisorPaths::new(directory.path().to_path_buf()).unwrap();
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
    assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 1);
}

#[test]
fn oversized_or_malformed_original_token_is_never_replaced() {
    let directory = tempfile::tempdir().unwrap();
    let paths = SupervisorPaths::new(directory.path().to_path_buf()).unwrap();
    for bytes in [b"partial".to_vec(), vec![b'a'; 65]] {
        fs::write(&paths.token, &bytes).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            fs::set_permissions(&paths.token, fs::Permissions::from_mode(0o600)).unwrap();
        }
        assert!(load_or_create_token(&paths).is_err());
        assert_eq!(fs::read(&paths.token).unwrap(), bytes);
    }
}
