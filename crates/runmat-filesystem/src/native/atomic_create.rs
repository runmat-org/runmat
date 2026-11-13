//! Shared no-replace file publication, including private credentials.
use crate::{AtomicCreateResult, PublicationDurability};
use std::{fs, io, path::Path};
use uuid::Uuid;

pub(super) fn publish(
    path: &Path,
    bytes: &[u8],
    maximum_existing_bytes: Option<u64>,
    private: bool,
) -> io::Result<AtomicCreateResult> {
    let parent = path.parent().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "atomic publication path has no parent directory",
        )
    })?;
    #[cfg(windows)]
    let _directory_guard = if private {
        Some(super::private_windows::directory::open(
            &std::path::absolute(parent)?,
            super::private_windows::directory::Mode::CreateMissingDurable,
        )?)
    } else {
        fs::create_dir_all(parent)?;
        None
    };
    #[cfg(not(windows))]
    fs::create_dir_all(parent)?;

    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("artifact");
    let mut staged = None;
    for _ in 0..16 {
        let candidate = parent.join(format!(
            ".{file_name}.publish-{}-{}",
            std::process::id(),
            Uuid::new_v4()
        ));
        match create_staged(&candidate, private) {
            Ok(mut file) => {
                if let Err(error) =
                    std::io::Write::write_all(&mut file, bytes).and_then(|()| file.sync_all())
                {
                    let _ = fs::remove_file(&candidate);
                    return Err(error);
                }
                staged = Some(candidate);
                break;
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    }
    let staged = staged.ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::AlreadyExists,
            "could not allocate a unique atomic publication staging path",
        )
    })?;

    let publication = match fs::hard_link(&staged, path) {
        Ok(()) => {
            let durability = match super::directory_sync::sync(parent) {
                Ok(()) => PublicationDurability::Durable,
                Err(error)
                    if matches!(
                        error.kind(),
                        io::ErrorKind::Unsupported | io::ErrorKind::PermissionDenied
                    ) =>
                {
                    PublicationDurability::AtomicVisibility
                }
                Err(_) => PublicationDurability::AtomicVisibility,
            };
            Ok(AtomicCreateResult::Published { durability })
        }
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => (|| {
            let metadata = fs::symlink_metadata(path)?;
            if metadata.file_type().is_symlink() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "atomic publication target must not be a symbolic link",
                ));
            }
            if !metadata.is_file() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "atomic publication target is not a regular file",
                ));
            }
            let mut input = open_existing(path, private)?;
            let existing = match maximum_existing_bytes {
                Some(maximum) => crate::workspace_access::file::read_opened(input, maximum)?,
                None => {
                    let mut bytes = Vec::new();
                    std::io::Read::read_to_end(&mut input, &mut bytes)?;
                    bytes
                }
            };
            Ok(AtomicCreateResult::Existing(existing))
        })(),
        Err(error) => Err(error),
    };
    let _ = fs::remove_file(&staged);
    publication
}

fn create_staged(path: &Path, private: bool) -> io::Result<fs::File> {
    #[cfg(windows)]
    if private {
        return super::private_windows::create_file_new(path);
    }
    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    if private {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.mode(0o600);
    }
    #[cfg(not(any(unix, windows)))]
    if private {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "native private publication is unavailable on this platform",
        ));
    }
    options.open(path)
}

fn open_existing(path: &Path, private: bool) -> io::Result<fs::File> {
    #[cfg(windows)]
    if private {
        return super::private_windows::open_file(path);
    }
    let mut options = fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.custom_flags(libc::O_NOFOLLOW);
    }
    let file = options.open(path)?;
    let metadata = file.metadata()?;
    if !metadata.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "atomic publication target is not a regular file",
        ));
    }
    #[cfg(unix)]
    if private {
        use std::os::unix::fs::PermissionsExt as _;
        if metadata.permissions().mode() & 0o777 != 0o600 {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "private atomic publication target must be mode0600",
            ));
        }
    }
    #[cfg(not(any(unix, windows)))]
    if private {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "native private publication is unavailable on this platform",
        ));
    }
    Ok(file)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FsProvider, NativeFsProvider};
    use futures::executor;
    use std::sync::{Arc, Barrier};

    #[test]
    fn concurrent_native_publishers_choose_exactly_one_complete_payload() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("artifact.json");
        let barrier = Arc::new(Barrier::new(2));
        let handles = [b"first".to_vec(), b"second".to_vec()].map(|payload| {
            let barrier = Arc::clone(&barrier);
            let path = path.clone();
            std::thread::spawn(move || {
                barrier.wait();
                executor::block_on(NativeFsProvider.atomic_create_new(&path, &payload))
            })
        });
        let outcomes = handles.map(|handle| handle.join().unwrap().unwrap());
        assert_eq!(
            outcomes
                .iter()
                .filter(|outcome| matches!(outcome, AtomicCreateResult::Published { .. }))
                .count(),
            1
        );
        let winning = fs::read(&path).unwrap();
        assert!(winning == b"first" || winning == b"second");
        assert!(outcomes.iter().any(
            |outcome| matches!(outcome, AtomicCreateResult::Existing(bytes) if bytes == &winning)
        ));
        assert!(fs::read_dir(directory.path()).unwrap().all(|entry| {
            !entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .contains(".publish-")
        }));
    }

    #[cfg(unix)]
    #[test]
    fn native_publication_rejects_an_existing_symlink() {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir().unwrap();
        let outside = directory.path().join("outside.json");
        let target = directory.path().join("artifact.json");
        fs::write(&outside, b"outside").unwrap();
        symlink(&outside, &target).unwrap();
        let error = executor::block_on(NativeFsProvider.atomic_create_new(&target, b"candidate"))
            .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert_eq!(fs::read(outside).unwrap(), b"outside");
    }
    #[cfg(unix)]
    #[test]
    fn private_publication_sets_permissions_before_exposing_one_complete_winner() {
        use std::os::unix::fs::PermissionsExt as _;
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("credential");
        let barrier = Arc::new(Barrier::new(2));
        let handles = [b"first".to_vec(), b"second".to_vec()].map(|payload| {
            let path = path.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                barrier.wait();
                NativeFsProvider
                    .atomic_create_private_new(&path, &payload, 6)
                    .unwrap()
            })
        });
        let outcomes = handles.map(|h| h.join().unwrap());
        assert_eq!(
            outcomes
                .iter()
                .filter(|v| matches!(v, AtomicCreateResult::Published { .. }))
                .count(),
            1
        );
        let bytes = fs::read(&path).unwrap();
        assert!(bytes == b"first" || bytes == b"second");
        assert!(outcomes
            .iter()
            .any(|v| matches!(v, AtomicCreateResult::Existing(existing) if existing == &bytes)));
        assert_eq!(
            fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
        assert!(NativeFsProvider
            .atomic_create_private_new(&path, b"replace", 1)
            .is_err());
        assert_eq!(fs::read(&path).unwrap(), bytes);
        fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();
        assert_eq!(
            NativeFsProvider
                .atomic_create_private_new(&path, b"replace", 6)
                .unwrap_err()
                .kind(),
            io::ErrorKind::PermissionDenied
        );
        assert_eq!(fs::read(&path).unwrap(), bytes);
    }
}
