use super::*;
use crate::workspace::*;
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::{BufWriter, Read, Write};

pub(crate) struct Guard {
    _file: fs::File,
}

pub(crate) fn protected(root: &Path, target: &Path) -> io::Result<()> {
    let journal = root.join(JOURNAL);
    if target.starts_with(&journal) || journal.starts_with(target) {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "workspace publication journal is reserved",
        ));
    }
    ensure_no_symlink(root, target)
}

pub(crate) fn guard(root: &Path, create: bool) -> io::Result<Option<Guard>> {
    guard_admitted(root, create, u64::MAX)
}

pub(crate) fn guard_admitted(
    root: &Path,
    create: bool,
    maximum_resident_bytes: u64,
) -> io::Result<Option<Guard>> {
    let directory = root.join(JOURNAL);
    ensure_no_symlink(root, &directory)?;
    if !create && !directory.exists() {
        return Ok(None);
    }
    fs::create_dir_all(&directory)?;
    ensure_no_symlink(root, &directory.join("lock"))?;
    let lock = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(directory.join("lock"))?;
    // Guards cover a synchronous filesystem operation and are dropped before
    // returning. Writable handles acquire a new guard per operation; they never
    // retain it between calls. Serialize short competing operations here rather
    // than dropping accepted decisions because another host briefly owns the lock.
    // Native hosts call these operations on their filesystem/runtime workers.
    lock.lock_exclusive()?;
    let guard = Guard { _file: lock };
    recover_admitted(root, maximum_resident_bytes)?;
    Ok(Some(guard))
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) enum FinalEdit {
    Write {
        path: PathBuf,
        bytes: Vec<u8>,
        mode: Option<u32>,
    },
    Remove {
        path: PathBuf,
    },
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Record {
    pub digest: [u8; 32],
    pub plan_digest: [u8; 32],
    pub receipt: WorkspacePublicationReceipt,
}

pub(super) fn record_directory(root: &Path, id: WorkspacePublicationId) -> PathBuf {
    root.join(JOURNAL)
        .join("records")
        .join(uuid::Uuid::from_bytes(id.0).to_string())
}

pub(super) fn read_record(directory: &Path) -> io::Result<Option<Record>> {
    read_record_admitted(directory, u64::MAX)
}

pub(super) fn read_record_admitted(directory: &Path, maximum: u64) -> io::Result<Option<Record>> {
    recovery::read_metadata(&directory.join("commit.json"), 8192, maximum)
}

pub(super) fn retained_recovery_metadata_bytes() -> u64 {
    (std::mem::size_of::<Record>() + std::mem::size_of::<WorkspacePublicationId>()) as u64
}

#[cfg(test)]
pub(super) fn write_record(
    directory: &Path,
    plan: &[FinalEdit],
    record: &mut Record,
) -> io::Result<CommitDurability> {
    write_record_checked(directory, plan, record, || Ok(()))
}

pub(super) fn write_record_checked(
    directory: &Path,
    plan: &[FinalEdit],
    record: &mut Record,
    before_commit: impl FnOnce() -> io::Result<()>,
) -> io::Result<CommitDurability> {
    let journal = directory
        .parent()
        .and_then(Path::parent)
        .ok_or_else(|| invalid("journal record has no mount"))?;
    let pending = serde_json::to_vec(&record.receipt.id).map_err(io::Error::other)?;
    match crate::native::atomic_create_new_native(&journal.join("pending.json"), &pending)? {
        crate::AtomicCreateResult::Published {
            durability: crate::PublicationDurability::Durable,
        } => {}
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::WouldBlock,
                "another workspace publication is pending",
            ))
        }
    }
    fs::create_dir_all(directory)?;
    let file = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(directory.join("plan.json"))?;
    let mut file = BufWriter::with_capacity(64 * 1024, file);
    serde_json::to_writer(&mut file, plan).map_err(io::Error::other)?;
    file.flush()?;
    file.get_ref().sync_all()?;
    // Persist the plan and its containing directory entry before making the
    // commit durable. No fallible synchronization may masquerade as a rejected
    // commit after atomic_create_new has established Durable.
    sync_journal_directories(directory)?;
    record.plan_digest = digest_file(&directory.join("plan.json"))?;
    let bytes = serde_json::to_vec(&record).map_err(io::Error::other)?;
    // Preparation may be expensive. Fence again after it, immediately before
    // allocating the durable commit, while still holding the publication lock.
    before_commit()?;
    let outcome = crate::native::atomic_create_new_native(&directory.join("commit.json"), &bytes)?;
    match outcome {
        crate::AtomicCreateResult::Published {
            durability: crate::PublicationDurability::Durable,
        } => Ok(CommitDurability::Durable),
        crate::AtomicCreateResult::Published {
            durability: crate::PublicationDurability::AtomicVisibility,
        } => Ok(CommitDurability::Indeterminate),
        crate::AtomicCreateResult::Existing(_) => {
            Err(invalid("workspace commit identity already exists"))
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(super) enum CommitDurability {
    Durable,
    Indeterminate,
}

pub(super) fn finish_committed(
    root: &Path,
    directory: &Path,
    plan: &[FinalEdit],
    receipt: WorkspacePublicationReceipt,
) -> WorkspacePublicationOutcome {
    match materialize(root, directory, plan) {
        Ok(()) => WorkspacePublicationOutcome::Published { receipt },
        Err(error) => {
            log::warn!(
                "durable workspace publication {:?} awaits recovery: {error}",
                receipt.id
            );
            WorkspacePublicationOutcome::CommittedPendingRecovery { receipt }
        }
    }
}

fn digest_file(path: &Path) -> io::Result<[u8; 32]> {
    let mut file = fs::File::open(path)?;
    let mut hash = Sha256::new();
    let mut bytes = [0u8; 64 * 1024];
    loop {
        let count = file.read(&mut bytes)?;
        if count == 0 {
            break;
        }
        hash.update(&bytes[..count]);
    }
    Ok(hash.finalize().into())
}

pub(super) fn materialize(root: &Path, directory: &Path, plan: &[FinalEdit]) -> io::Result<()> {
    for (index, edit) in plan.iter().enumerate() {
        match edit {
            FinalEdit::Write { path, bytes, mode } => {
                let target = confined(root, path)?;
                protected(root, &target)?;
                let parent = target
                    .parent()
                    .ok_or_else(|| invalid("publication target has no parent"))?;
                fs::create_dir_all(parent)?;
                let temporary = parent.join(format!(
                    ".runmat-publication-{}-{index}",
                    directory
                        .file_name()
                        .and_then(|value| value.to_str())
                        .ok_or_else(|| invalid("invalid journal id"))?
                ));
                ensure_no_symlink(root, &temporary)?;
                let mut file = fs::OpenOptions::new()
                    .create(true)
                    .truncate(true)
                    .write(true)
                    .open(&temporary)?;
                file.write_all(bytes)?;
                #[cfg(unix)]
                if let Some(mode) = mode {
                    use std::os::unix::fs::PermissionsExt;
                    file.set_permissions(fs::Permissions::from_mode(*mode))?;
                }
                #[cfg(not(unix))]
                let _ = mode;
                file.sync_all()?;
                fs::rename(&temporary, &target)?;
                sync_ancestors(root, parent)?;
            }
            FinalEdit::Remove { path } => {
                let target = confined(root, path)?;
                protected(root, &target)?;
                match fs::remove_file(&target) {
                    Ok(()) => {}
                    Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                    Err(error) => return Err(error),
                }
                if let Some(parent) = target.parent() {
                    sync_directory(parent)?;
                }
            }
        }
    }
    let complete = fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(directory.join("complete"))?;
    complete.sync_all()?;
    sync_directory(directory)?;
    // Receipts remain durable while large prepared content is reclaimed.
    fs::remove_file(directory.join("plan.json"))?;
    sync_directory(directory)?;
    clear_pending(root)
}

pub(crate) fn recover_admitted(root: &Path, maximum_resident_bytes: u64) -> io::Result<()> {
    recovery::recover(root, maximum_resident_bytes)
}

mod certainty;
pub(crate) use certainty::uncertain_receipt;
pub(super) use certainty::{confirm, uncertain_outcome};
mod recovery;
pub(super) use recovery::limits as recovery_limits;

fn clear_pending(root: &Path) -> io::Result<()> {
    match fs::remove_file(root.join(JOURNAL).join("pending.json")) {
        Ok(()) => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(error),
    }
    sync_directory(&root.join(JOURNAL))
}

/// Persist each journal directory entry through the admitted mount root. Syncing
/// only a newly created records directory does not persist that directory itself.
fn sync_journal_directories(directory: &Path) -> io::Result<()> {
    let mut current = directory;
    for depth in 0..5 {
        sync_directory(current)?;
        if depth < 4 {
            current = current
                .parent()
                .ok_or_else(|| invalid("journal has no mount ancestor"))?;
        }
    }
    Ok(())
}
