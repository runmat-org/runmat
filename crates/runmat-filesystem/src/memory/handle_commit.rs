//! Snapshot handles may flush only to the entry that was opened. In particular,
//! a dropped old handle cannot overwrite a newly created mapped backing.

use super::{bump_identity, ensure_parent_dirs, normalize_path, MemoryEntry, MemoryFsProvider};
use std::io::{self, ErrorKind};
use std::path::Path;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

pub(super) fn commit(
    provider: &MemoryFsProvider,
    path: &Path,
    expected_identity: &Arc<AtomicU64>,
    expected_version: u64,
    originally_present: bool,
    bytes: &[u8],
) -> io::Result<u64> {
    let path = normalize_path(path)?;
    let mut tree = provider
        .inner
        .lock()
        .map_err(|_| io::Error::other("memory filesystem lock poisoned"))?;
    match tree.entries.get(&path) {
        Some(MemoryEntry::File {
            identity,
            readonly: false,
            ..
        }) if originally_present
            && Arc::ptr_eq(identity, expected_identity)
            && identity.load(Ordering::SeqCst) == expected_version => {}
        Some(MemoryEntry::File { readonly: true, .. }) => {
            return Err(io::Error::new(
                ErrorKind::PermissionDenied,
                "opened memory file became read-only",
            ));
        }
        None if !originally_present => ensure_parent_dirs(&mut tree, &path)?,
        Some(MemoryEntry::File { .. }) | Some(MemoryEntry::Directory) | None => {
            return Err(io::Error::new(
                ErrorKind::PermissionDenied,
                "opened memory file identity was removed or replaced",
            ));
        }
    }
    let replacement = bytes.to_vec();
    let version = if originally_present {
        bump_identity(expected_identity)?
    } else {
        expected_version
    };
    tree.entries.insert(
        path,
        MemoryEntry::File {
            bytes: Arc::new(replacement),
            readonly: false,
            revision: uuid::Uuid::new_v4().to_string(),
            identity: expected_identity.clone(),
        },
    );
    Ok(version)
}
