//! Consumer-requested private directory creation with explicit ancestor retention.
#[cfg(not(windows))]
use std::fs;
use std::{io, path::Path};

#[cfg(windows)]
pub(super) fn create_durable(path: &Path) -> io::Result<()> {
    let path = std::path::absolute(path)?;
    super::private_windows::directory::open(
        &path,
        super::private_windows::directory::Mode::CreateMissingDurable,
    )?;
    Ok(())
}

#[cfg(not(windows))]
pub(super) fn create_durable(path: &Path) -> io::Result<()> {
    let mut missing = Vec::new();
    let mut ancestor = path;
    loop {
        match fs::symlink_metadata(ancestor) {
            Ok(metadata) if metadata.file_type().is_dir() => break,
            Ok(_) => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "private directory is not a regular directory",
                ))
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                missing.push(ancestor.to_path_buf());
                ancestor = ancestor
                    .parent()
                    .filter(|p| !p.as_os_str().is_empty())
                    .ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::InvalidInput,
                            "private directory requires an existing ancestor",
                        )
                    })?;
            }
            Err(error) => return Err(error),
        }
    }
    for directory in missing.into_iter().rev() {
        match fs::create_dir(&directory) {
            Ok(()) => {}
            Err(error)
                if error.kind() == io::ErrorKind::AlreadyExists
                    && fs::symlink_metadata(&directory)?.file_type().is_dir() => {}
            Err(error) => return Err(error),
        }
        make_private(&directory)?;
        super::directory_sync::sync(&directory)?;
        super::directory_sync::sync(directory.parent().expect("validated ancestor"))?;
    }
    make_private(path)?;
    super::directory_sync::sync(path)
}

#[cfg(not(windows))]
fn make_private(path: &Path) -> io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o700))?;
    }
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    #[test]
    fn native_publication_works_inside_an_existing_executor() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("original.json");
        futures::executor::block_on(async {
            let first = crate::NativeFsProvider
                .atomic_create_new_sync(&path, b"original")
                .unwrap();
            assert!(matches!(first, crate::AtomicCreateResult::Published { .. }));
            let replaced = crate::NativeFsProvider
                .atomic_replace_sync(&path, b"replacement")
                .unwrap();
            assert_eq!(replaced.durability, crate::PublicationDurability::Durable);
        });
        assert_eq!(fs::read(path).unwrap(), b"replacement");
    }

    #[test]
    fn nested_creation_retains_private_ancestors_and_rejects_foreign_leaf() {
        let root = tempfile::tempdir().unwrap();
        let leaf = root.path().join("authority/jobs/original");
        create_durable(&leaf).unwrap();
        for directory in [
            root.path().join("authority"),
            root.path().join("authority/jobs"),
            leaf.clone(),
        ] {
            assert_eq!(
                fs::metadata(directory).unwrap().permissions().mode() & 0o777,
                0o700
            );
        }
        create_durable(&leaf).unwrap();
        let foreign = root.path().join("foreign");
        fs::write(&foreign, b"original").unwrap();
        assert!(create_durable(&foreign).is_err());
        assert_eq!(fs::read(foreign).unwrap(), b"original");
        let link = root.path().join("alias");
        std::os::unix::fs::symlink(&leaf, &link).unwrap();
        assert!(create_durable(&link).is_err());
    }
}
