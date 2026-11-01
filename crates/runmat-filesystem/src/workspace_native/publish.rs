use super::*;
use crate::{workspace::*, FsProvider, MemoryFsProvider};
use std::collections::{BTreeMap, BTreeSet};

pub(crate) fn lookup(
    root: &Path,
    request: &WorkspacePublicationRequest,
) -> io::Result<Option<WorkspacePublicationOutcome>> {
    if !matches!(request.root.as_path(), path if path == Path::new("/") || path == Path::new(".")) {
        return Err(invalid("workspace receipt belongs to a different mount"));
    }
    let _guard = match guard(root, false) {
        Ok(guard) => guard,
        Err(error) => {
            return match journal::uncertain_outcome(&error, request) {
                Some(outcome) => Ok(Some(outcome)),
                None => Err(error),
            }
        }
    };
    let directory = journal::record_directory(root, request.id);
    ensure_no_symlink(root, &directory.join("commit.json"))?;
    let digest = publication_digest(request)?;
    Ok(journal::read_record(&directory)?.map(|record| {
        if record.digest == digest {
            WorkspacePublicationOutcome::Published {
                receipt: record.receipt,
            }
        } else {
            WorkspacePublicationOutcome::IdReused
        }
    }))
}

pub(crate) async fn publish(
    root: &Path,
    request: WorkspacePublicationRequest,
) -> io::Result<WorkspacePublicationOutcome> {
    publish_with_lease(root, request, None).await
}

pub(crate) async fn publish_with_lease(
    root: &Path,
    request: WorkspacePublicationRequest,
    lease: Option<&crate::execution::ExecutionLease>,
) -> io::Result<WorkspacePublicationOutcome> {
    validate_budget(&request)?;
    let _guard = match guard(root, true) {
        Ok(guard) => guard,
        Err(error) => {
            return match journal::uncertain_outcome(&error, &request) {
                Some(outcome) => Ok(outcome),
                None => Err(error),
            }
        }
    };
    publish_locked(root, request, lease, None).await
}

/// Caller owns the journal/recovery guard for the entire operation.
pub(crate) async fn publish_locked(
    root: &Path,
    request: WorkspacePublicationRequest,
    lease: Option<&crate::execution::ExecutionLease>,
    maximum_resident_bytes: Option<u64>,
) -> io::Result<WorkspacePublicationOutcome> {
    validate_budget(&request)?;
    if let Some(maximum) = maximum_resident_bytes {
        super::admission::request(&request, maximum)?;
    }
    publish_validated(root, request, lease, maximum_resident_bytes).await
}

fn validate_budget(request: &WorkspacePublicationRequest) -> io::Result<()> {
    if !cfg!(any(unix, windows)) {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "native durable workspace publication requires directory synchronization",
        ));
    }
    if request.edits.is_empty()
        || request.edits.len() > MAX_WORKSPACE_PUBLICATION_EDITS
        || request
            .edits
            .iter()
            .try_fold(0usize, |total, edit| {
                total.checked_add(match edit {
                    WorkspaceEdit::Write { bytes, .. } => bytes.len(),
                    _ => 0,
                })
            })
            .is_none_or(|size| size > MAX_WORKSPACE_PUBLICATION_BYTES)
    {
        return Err(invalid("workspace publication exceeds its bounded budget"));
    }
    Ok(())
}

async fn publish_validated(
    root: &Path,
    request: WorkspacePublicationRequest,
    lease: Option<&crate::execution::ExecutionLease>,
    maximum_resident_bytes: Option<u64>,
) -> io::Result<WorkspacePublicationOutcome> {
    let digest = publication_digest(&request)?;
    let directory = journal::record_directory(root, request.id);
    if let Some(record) =
        journal::read_record_admitted(&directory, maximum_resident_bytes.unwrap_or(u64::MAX))?
    {
        return Ok(if record.digest == digest {
            WorkspacePublicationOutcome::Published {
                receipt: record.receipt,
            }
        } else {
            WorkspacePublicationOutcome::IdReused
        });
    }
    if let Some(lease) = lease {
        execution::require_current(root, lease)?;
    }
    let capture_request = WorkspaceCaptureRequest {
        root: request.root.clone(),
        excluded_subtrees: request.excluded_subtrees.clone(),
        max_files: 10_000,
        max_bytes: MAX_WORKSPACE_PUBLICATION_BYTES as u64,
    };
    let inventory = match maximum_resident_bytes {
        Some(maximum) => capture::publication_inventory(root, capture_request, &request, maximum)?,
        None => capture::inventory(root, capture_request)?,
    };
    if inventory.snapshot.revision != request.expected_revision {
        return Ok(WorkspacePublicationOutcome::Conflict {
            current: inventory.snapshot.revision,
        });
    }
    let mut touched = BTreeSet::new();
    let mut modes = BTreeMap::new();
    for file in &inventory.snapshot.files {
        modes.insert(file.path.clone(), mode(&root.join(&file.path))?);
    }
    for edit in &request.edits {
        let paths = match edit {
            WorkspaceEdit::Write { path, .. } | WorkspaceEdit::Remove { path } => vec![path],
            WorkspaceEdit::Rename { from, to } => vec![from, to],
        };
        for path in paths {
            protected(root, &confined(root, path)?)?;
            for (parent, readonly) in &inventory.directories {
                if *readonly && path.starts_with(parent) {
                    return Err(io::Error::new(
                        io::ErrorKind::PermissionDenied,
                        "publication would mutate a read-only directory",
                    ));
                }
            }
            touched.insert(path.clone());
        }
        if let WorkspaceEdit::Rename { from, to } = edit {
            let permission = modes.remove(from).flatten();
            modes.insert(to.clone(), permission);
        }
    }
    // Reuse the common transaction validator on a private memory view. No
    // ordinary native writes occur while checking the complete edit sequence.
    let memory = MemoryFsProvider::with_current_dir("/");
    {
        for (directory, _) in &inventory.directories {
            memory
                .create_dir_all(&Path::new("/").join(directory))
                .await?;
        }
        for file in &inventory.snapshot.files {
            let path = Path::new("/").join(&file.path);
            memory.write(&path, &file.bytes).await?;
            memory.set_readonly(&path, file.readonly).await?;
        }
        // The private provider now owns its validation view. Release the original
        // capture buffers before it creates either pre- or post-edit snapshots.
        drop(inventory.snapshot.files);
        let capture_request = WorkspaceCaptureRequest {
            root: "/".into(),
            excluded_subtrees: inventory.snapshot.excluded_subtrees.clone(),
            max_files: 10_000,
            max_bytes: MAX_WORKSPACE_PUBLICATION_BYTES as u64,
        };
        let before = memory.capture_workspace(capture_request.clone()).await?;
        // Only the revision and exclusions are needed for conditional validation.
        // Release the copied file buffers before creating the post-edit inventory.
        drop(before.files);
        let outcome = memory
            .publish_workspace(WorkspacePublicationRequest {
                id: request.id,
                root: "/".into(),
                excluded_subtrees: before.excluded_subtrees,
                expected_revision: before.revision,
                edits: request.edits,
            })
            .await?;
        if !matches!(outcome, WorkspacePublicationOutcome::Published { .. }) {
            return Err(invalid("private transaction validation did not publish"));
        }
        let mut after = memory.capture_workspace(capture_request).await?;
        for file in &mut after.files {
            file.version = capture::hex(&sha2::Sha256::digest(&file.bytes));
        }
        after
            .files
            .sort_by(|left, right| left.path.cmp(&right.path));
        let mut directories = inventory
            .directories
            .into_iter()
            .collect::<BTreeMap<_, _>>();
        for file in &after.files {
            let mut parent = file.path.parent();
            while let Some(path) = parent {
                directories.entry(path.to_path_buf()).or_insert(false);
                parent = path.parent();
            }
        }
        let revision = capture::revision(
            &inventory.snapshot.excluded_subtrees,
            &directories.into_iter().collect::<Vec<_>>(),
            &after.files,
        )?;
        // Validation is complete; the original memory tree is no longer an input
        // to revision hashing or journal construction.
        drop(memory);
        let mut after = after
            .files
            .into_iter()
            .map(|file| (file.path.clone(), file))
            .collect::<BTreeMap<_, _>>();
        let mut plan = Vec::new();
        for path in touched {
            match after.remove(&path) {
                Some(file) => plan.push(journal::FinalEdit::Write {
                    path: path.clone(),
                    bytes: file.bytes,
                    mode: modes.get(&path).copied().flatten(),
                }),
                None => plan.push(journal::FinalEdit::Remove { path }),
            }
        }
        // Untouched originals are not journal payloads.
        drop(after);
        if let Some(maximum) = maximum_resident_bytes {
            super::admission::journal(&plan, maximum)?;
        }
        let records = directory
            .parent()
            .ok_or_else(|| invalid("journal record has no parent"))?;
        fs::create_dir_all(records)?;
        sync_ancestors(root, records)?;
        if fs::read_dir(records)?.take(10_000).count() >= 10_000 {
            return Err(invalid("workspace publication receipt budget exhausted"));
        }
        if directory.exists() {
            fs::remove_dir_all(&directory)?;
        }
        let mut record = journal::Record {
            digest,
            plan_digest: [0; 32],
            receipt: WorkspacePublicationReceipt {
                id: request.id,
                revision,
            },
        };
        let durability = journal::write_record_checked(&directory, &plan, &mut record, || {
            if let Some(lease) = lease {
                execution::require_current(root, lease)?;
            }
            Ok(())
        })?;
        if durability == journal::CommitDurability::Indeterminate {
            return Ok(WorkspacePublicationOutcome::DurabilityIndeterminate {
                receipt: record.receipt,
            });
        }
        Ok(journal::finish_committed(
            root,
            &directory,
            &plan,
            record.receipt,
        ))
    }
}

fn mode(path: &Path) -> io::Result<Option<u32>> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        Ok(Some(fs::metadata(path)?.permissions().mode()))
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        Ok(None)
    }
}

use sha2::Digest;
