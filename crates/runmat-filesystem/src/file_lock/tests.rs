use super::*;

#[test]
fn ownership_is_exclusive_and_drop_reuses_the_original_file() {
    let temporary = tempfile::tempdir().unwrap();
    let path = temporary.path().join("authority.lock");
    // A stale/live-looking PID is merely legacy content, never ownership evidence.
    std::fs::write(&path, std::process::id().to_string()).unwrap();
    let original = std::fs::read(&path).unwrap();
    let first = ExclusiveFileLock::try_acquire(&path).unwrap().unwrap();
    assert!(ExclusiveFileLock::try_acquire(&path).unwrap().is_none());
    drop(first);
    assert!(path.exists());
    let second = ExclusiveFileLock::try_acquire(&path).unwrap().unwrap();
    assert!(ExclusiveFileLock::try_acquire(&path).unwrap().is_none());
    drop(second);
    // Windows byte-range locks also exclude ordinary readers. Inspect the
    // conserved legacy bytes after releasing ownership, not through its lock.
    assert_eq!(std::fs::read(&path).unwrap(), original);
    assert!(ExclusiveFileLock::try_acquire(&path).unwrap().is_some());
}

#[test]
fn independent_contenders_cannot_both_mutate_under_one_lease() {
    use std::sync::{Arc, Barrier};
    let temporary = tempfile::tempdir().unwrap();
    let path = temporary.path().join("authority.lock");
    let enter = Arc::new(Barrier::new(2));
    let leave = Arc::new(Barrier::new(2));
    let contenders = (0..2)
        .map(|_| {
            let (path, enter, leave) = (path.clone(), enter.clone(), leave.clone());
            std::thread::spawn(move || {
                enter.wait();
                let lease = ExclusiveFileLock::try_acquire(&path).unwrap();
                let acquired = lease.is_some();
                leave.wait(); // Keep the winner alive through the loser's acquisition.
                drop(lease);
                acquired
            })
        })
        .collect::<Vec<_>>();
    assert_eq!(
        contenders
            .into_iter()
            .filter_map(|c| c.join().unwrap().then_some(()))
            .count(),
        1
    );
    assert!(ExclusiveFileLock::try_acquire(&path).unwrap().is_some());
}

#[cfg(unix)]
#[test]
fn symlinks_are_not_accepted_as_authority_files() {
    let temporary = tempfile::tempdir().unwrap();
    let target = temporary.path().join("target");
    std::fs::write(&target, b"unrelated").unwrap();
    let path = temporary.path().join("authority.lock");
    std::os::unix::fs::symlink(&target, &path).unwrap();
    assert!(ExclusiveFileLock::try_acquire(&path).is_err());
    assert_eq!(std::fs::read(&target).unwrap(), b"unrelated");
}

#[test]
fn another_process_cannot_enter_and_abrupt_owner_exit_releases_the_lease() {
    use std::io::BufRead as _;
    use std::process::{Command, Stdio};
    let temporary = tempfile::tempdir().unwrap();
    let path = temporary.path().join("authority.lock");
    let current = std::env::current_exe().unwrap();
    let fixture = "file_lock::tests::process_authority_fixture";
    let lease = ExclusiveFileLock::try_acquire(&path).unwrap().unwrap();
    let status = Command::new(&current)
        .args(["--ignored", "--exact", fixture, "--nocapture"])
        .env("RUNMAT_FILE_AUTHORITY_FIXTURE", &path)
        .env("RUNMAT_FILE_AUTHORITY_EXPECT_CONTENDED", "1")
        .stdout(Stdio::null())
        .status()
        .unwrap();
    assert!(status.success());
    drop(lease);
    let mut child = Command::new(current)
        .args(["--ignored", "--exact", fixture, "--nocapture"])
        .env("RUNMAT_FILE_AUTHORITY_FIXTURE", &path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut output = std::io::BufReader::new(child.stdout.take().unwrap());
    loop {
        let mut line = String::new();
        assert_ne!(
            output.read_line(&mut line).unwrap(),
            0,
            "owner exited before handshake"
        );
        if line.contains("authority-owner-ready") {
            break;
        }
    }
    assert!(ExclusiveFileLock::try_acquire(&path).unwrap().is_none());
    child.kill().unwrap();
    assert!(!child.wait().unwrap().success());
    assert!(path.exists());
    assert!(ExclusiveFileLock::try_acquire(&path).unwrap().is_some());
}

#[test]
#[ignore = "subprocess entry for the native authority lifecycle witness"]
fn process_authority_fixture() {
    use std::io::{Read as _, Write as _};
    let path = std::env::var_os("RUNMAT_FILE_AUTHORITY_FIXTURE").unwrap();
    let lease = ExclusiveFileLock::try_acquire(Path::new(&path)).unwrap();
    if std::env::var_os("RUNMAT_FILE_AUTHORITY_EXPECT_CONTENDED").is_some() {
        assert!(lease.is_none());
        return;
    }
    assert!(lease.is_some());
    println!("authority-owner-ready");
    std::io::stdout().flush().unwrap();
    let mut byte = [0];
    let _ = std::io::stdin().read(&mut byte).unwrap();
    drop(lease);
}
