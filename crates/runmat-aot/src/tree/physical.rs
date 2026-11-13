//! Actual native filesystem I/O shared by build inputs and installations.
use crate::{AotError, AotResult};
use runmat_package_cache::{relative_link_target, TreeEntryKind, TreeManifest};
use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::Path;
use std::path::PathBuf;

/// Revalidate exact physical contents, including implicit ancestor directories.
/// A manifest or previous successful check alone is not an installed lease.
pub(crate) fn verify(root: &Path, tree: &TreeManifest) -> AotResult<()> {
    tree.validate().map_err(invalid)?;
    let expected = super::storage::physical_inventory(tree, u64::MAX)?;
    let metadata = std::fs::symlink_metadata(root)
        .map_err(|error| AotError::io("inspect native tree root", root, error))?;
    if !metadata.is_dir() || metadata.is_symlink() {
        return Err(invalid("native tree root is not a physical directory"));
    }
    verify_inventory(root, root, &expected)?;
    for (relative, kind) in &expected {
        let path = root.join(relative);
        let metadata = std::fs::symlink_metadata(&path)
            .map_err(|error| AotError::io("inspect retained native entry", &path, error))?;
        if !matches_kind(&metadata, *kind) {
            return Err(invalid("native tree retained entry is missing or changed"));
        }
    }
    for entry in &tree.entries {
        let path = root.join(entry.path.as_str());
        if entry.kind == TreeEntryKind::File {
            let metadata = std::fs::metadata(&path)
                .map_err(|error| AotError::io("inspect native tree file", &path, error))?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                if entry.executable != (metadata.permissions().mode() & 0o100 != 0) {
                    return Err(invalid("native file execute intent changed"));
                }
            }
            if metadata.len() != entry.byte_len {
                return Err(invalid("native tree file length changed"));
            }
            let file = std::fs::File::open(&path)
                .map_err(|error| AotError::io("open native tree file", &path, error))?;
            let mut digest = runmat_execution::DigestStream::new();
            let count = std::io::copy(
                &mut file.take(entry.byte_len.saturating_add(1)),
                &mut digest,
            )
            .map_err(|error| AotError::io("verify native tree file", &path, error))?;
            if count != entry.byte_len
                || digest.finish().bytes() != entry.digest.as_ref().expect("validated file").bytes()
            {
                return Err(invalid("native tree file identity changed"));
            }
        } else if entry.kind == TreeEntryKind::Symlink {
            let target = std::fs::read_link(&path)
                .map_err(|error| AotError::io("verify native tree link", &path, error))?;
            if target
                != relative_link_target(
                    &entry.path,
                    entry.link_target.as_ref().expect("validated link"),
                )
            {
                return Err(invalid("native tree link identity changed"));
            }
        }
    }
    Ok(())
}

fn verify_inventory(
    root: &Path,
    directory: &Path,
    expected: &BTreeMap<PathBuf, TreeEntryKind>,
) -> AotResult<()> {
    for entry in std::fs::read_dir(directory)
        .map_err(|error| AotError::io("inspect native tree inventory", directory, error))?
    {
        let path = entry
            .map_err(|error| AotError::io("inspect native tree entry", directory, error))?
            .path();
        let relative = path.strip_prefix(root).map_err(invalid)?;
        let kind = expected
            .get(relative)
            .ok_or_else(|| invalid("native tree has an unretained entry"))?;
        let metadata = std::fs::symlink_metadata(&path)
            .map_err(|error| AotError::io("inspect native tree entry", &path, error))?;
        if !matches_kind(&metadata, *kind) {
            return Err(invalid("native tree physical entry kind changed"));
        }
        if *kind == TreeEntryKind::Directory {
            verify_inventory(root, &path, expected)?;
        }
    }
    Ok(())
}

fn matches_kind(metadata: &std::fs::Metadata, kind: TreeEntryKind) -> bool {
    match kind {
        TreeEntryKind::File => metadata.is_file() && !metadata.is_symlink(),
        TreeEntryKind::Directory => metadata.is_dir() && !metadata.is_symlink(),
        TreeEntryKind::Symlink => metadata.is_symlink(),
    }
}

pub(crate) fn write_new(path: &Path, bytes: &[u8]) -> AotResult<()> {
    write_file(path, bytes, None)
}

fn write_file(path: &Path, bytes: &[u8], executable: Option<bool>) -> AotResult<()> {
    let mut file = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)
        .map_err(|error| AotError::io("create private native input", path, error))?;
    file.write_all(bytes)
        .map_err(|error| AotError::io("write private native input", path, error))?;
    synchronize(file, path, executable)
}

pub(crate) fn write_blob(
    path: &Path,
    mut reader: impl Read,
    expected: &runmat_package_cache::TreeEntry,
) -> AotResult<()> {
    let mut file = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)
        .map_err(|error| AotError::io("create streamed native file", path, error))?;
    let mut remaining = expected.byte_len;
    let mut digest = runmat_execution::DigestStream::new();
    let mut buffer = [0u8; 64 * 1024];
    while remaining != 0 {
        let maximum = remaining.min(buffer.len() as u64) as usize;
        let count = reader
            .read(&mut buffer[..maximum])
            .map_err(|error| AotError::io("read streamed native file", path, error))?;
        if count == 0 {
            return Err(invalid("native file source ended before its exact extent"));
        }
        file.write_all(&buffer[..count])
            .map_err(|error| AotError::io("write streamed native file", path, error))?;
        digest.update(&buffer[..count]);
        remaining -= count as u64;
    }
    let mut trailing = [0u8; 1];
    if reader
        .read(&mut trailing)
        .map_err(|error| AotError::io("verify native file source extent", path, error))?
        != 0
        || digest.finish().bytes()
            != expected
                .digest
                .as_ref()
                .ok_or_else(|| invalid("native file digest is missing"))?
                .bytes()
    {
        return Err(invalid(
            "native file source differs from its exact digest or extent",
        ));
    }
    synchronize(file, path, Some(expected.executable))
}

fn synchronize(file: std::fs::File, path: &Path, executable: Option<bool>) -> AotResult<()> {
    #[cfg(unix)]
    if let Some(executable) = executable {
        use std::os::unix::fs::PermissionsExt;
        file.set_permissions(std::fs::Permissions::from_mode(if executable {
            0o555
        } else {
            0o444
        }))
        .map_err(|error| AotError::io("seal private native input", path, error))?;
    }
    #[cfg(not(unix))]
    let _ = executable;
    // Synchronize the same file after BOTH payload and mode mutations. A
    // subsequent directory fsync alone cannot persist this inode's mode.
    file.sync_all()
        .map_err(|error| AotError::io("synchronize sealed native input", path, error))
}

pub(crate) fn read_regular(path: &Path, max_bytes: u64) -> AotResult<Vec<u8>> {
    let (file, _) = open_bounded_regular(path, max_bytes)?;
    let mut bytes = Vec::new();
    file.take(max_bytes.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|error| AotError::io("read bounded native file", path, error))?;
    if bytes.len() as u64 > max_bytes {
        return Err(invalid("native file exceeds its admitted byte bound"));
    }
    Ok(bytes)
}

/// Retain the exact regular file currently named by `path` and independently
/// validate its opened extent. The path identity comparison prevents a
/// pre-open inspection from silently authorizing a different opened inode.
pub(crate) fn open_bounded_regular(path: &Path, max_bytes: u64) -> AotResult<(std::fs::File, u64)> {
    let named = std::fs::symlink_metadata(path)
        .map_err(|error| AotError::io("inspect bounded native file", path, error))?;
    if !named.is_file() || named.file_type().is_symlink() || named.len() > max_bytes {
        return Err(invalid("native file is not a bounded physical file"));
    }
    let file = std::fs::File::open(path)
        .map_err(|error| AotError::io("open bounded native file", path, error))?;
    let opened = file
        .metadata()
        .map_err(|error| AotError::io("inspect opened native file", path, error))?;
    if !opened.is_file() || opened.len() != named.len() || opened.len() > max_bytes {
        return Err(invalid(
            "opened native file differs from its admitted extent",
        ));
    }
    let retained = same_file::Handle::from_file(
        file.try_clone()
            .map_err(|error| AotError::io("retain opened native file", path, error))?,
    )
    .map_err(|error| AotError::io("identify opened native file", path, error))?;
    let current = same_file::Handle::from_path(path)
        .map_err(|error| AotError::io("identify named native file", path, error))?;
    let still_named = std::fs::symlink_metadata(path)
        .map_err(|error| AotError::io("reinspect named native file", path, error))?;
    if !still_named.is_file()
        || still_named.file_type().is_symlink()
        || still_named.len() != opened.len()
        || retained != current
    {
        return Err(invalid("native file identity changed while it was opened"));
    }
    Ok((file, opened.len()))
}

pub(crate) fn hash_open_file(
    file: &std::fs::File,
    length: u64,
    path: &Path,
) -> AotResult<runmat_execution::Digest> {
    use std::io::Seek;
    let mut reader = file
        .try_clone()
        .map_err(|error| AotError::io("retain native source file", path, error))?;
    reader
        .rewind()
        .map_err(|error| AotError::io("rewind native source file", path, error))?;
    let mut digest = runmat_execution::DigestStream::new();
    let count = std::io::copy(&mut reader.take(length.saturating_add(1)), &mut digest)
        .map_err(|error| AotError::io("hash bounded native source", path, error))?;
    if count != length {
        return Err(invalid("native source file extent changed"));
    }
    Ok(digest.finish())
}

fn invalid(error: impl std::fmt::Display) -> AotError {
    AotError::contract("aot.tree.physical", error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_regular_file_uses_opened_extent() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("image");
        std::fs::write(&path, b"native-image").unwrap();
        let (file, length) = open_bounded_regular(&path, 12).unwrap();
        assert_eq!(length, 12);
        assert_eq!(file.metadata().unwrap().len(), length);
        assert!(open_bounded_regular(&path, 11).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn bounded_regular_file_rejects_symbolic_links() {
        let root = tempfile::tempdir().unwrap();
        let image = root.path().join("image");
        let alias = root.path().join("alias");
        std::fs::write(&image, b"native-image").unwrap();
        std::os::unix::fs::symlink(&image, &alias).unwrap();
        assert!(open_bounded_regular(&alias, 12).is_err());
    }
}
