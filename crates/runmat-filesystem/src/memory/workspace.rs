use super::*;
use crate::workspace::*;
use sha2::{Digest, Sha256};

#[derive(Debug, Clone)]
pub(super) struct Published {
    digest: [u8; 32],
    receipt: WorkspacePublicationReceipt,
}

fn invalid(message: &str) -> io::Error {
    io::Error::new(ErrorKind::InvalidInput, message)
}

fn root(tree: &MemoryTree, path: &Path) -> io::Result<PathBuf> {
    let path = normalize_path(path)?;
    if !matches!(tree.entries.get(&path), Some(MemoryEntry::Directory)) {
        return Err(not_found(&path));
    }
    Ok(path)
}

fn excluded(root: &Path, paths: &[PathBuf]) -> io::Result<Vec<PathBuf>> {
    let mut result = paths
        .iter()
        .map(|path| confined(root, path))
        .collect::<io::Result<Vec<_>>>()?;
    result.sort();
    result.dedup();
    Ok(result)
}

fn revision(tree: &MemoryTree, root: &Path, excluded: &[PathBuf]) -> io::Result<WorkspaceRevision> {
    let mut hash = Sha256::new();
    hash.update(b"runmat-workspace-memory-v1");
    hash.update(
        serde_json::to_vec(
            &excluded
                .iter()
                .map(|path| path.strip_prefix(root).unwrap_or(path))
                .collect::<Vec<_>>(),
        )
        .map_err(|error| io::Error::new(ErrorKind::InvalidData, error))?,
    );
    for (path, entry) in tree.entries.iter().filter(|(path, _)| {
        path.starts_with(root) && !excluded.iter().any(|ignored| path.starts_with(ignored))
    }) {
        let relative = path
            .strip_prefix(root)
            .map_err(|_| invalid("invalid workspace path"))?;
        let relative = relative
            .to_str()
            .ok_or_else(|| invalid("workspace paths must be UTF-8"))?;
        let value = match entry {
            MemoryEntry::Directory => serde_json::to_vec(&(relative, None::<(&str, bool)>)),
            MemoryEntry::File {
                revision, readonly, ..
            } => serde_json::to_vec(&(relative, Some((revision.as_str(), *readonly)))),
        }
        .map_err(|error| io::Error::new(ErrorKind::InvalidData, error))?;
        hash.update((value.len() as u64).to_le_bytes());
        hash.update(value);
    }
    Ok(WorkspaceRevision(hash.finalize().into()))
}

pub(super) fn capture(
    provider: &MemoryFsProvider,
    request: WorkspaceCaptureRequest,
) -> io::Result<WorkspaceSnapshot> {
    let tree = provider
        .inner
        .lock()
        .expect("memory filesystem lock poisoned");
    let root = root(&tree, &request.root)?;
    let excluded = excluded(&root, &request.excluded_subtrees)?;
    let mut files = Vec::new();
    let mut size = 0u64;
    for (path, entry) in tree.entries.iter().filter(|(path, _)| {
        path.starts_with(&root) && !excluded.iter().any(|ignored| path.starts_with(ignored))
    }) {
        if let MemoryEntry::File {
            bytes,
            revision,
            readonly,
            ..
        } = entry
        {
            size = size
                .checked_add(bytes.len() as u64)
                .ok_or_else(|| invalid("workspace capture size overflow"))?;
            if size > request.max_bytes || files.len() >= request.max_files as usize {
                return Err(invalid("workspace capture exceeds its bounded budget"));
            }
            files.push(WorkspaceFile {
                path: path
                    .strip_prefix(&root)
                    .map_err(|_| invalid("invalid workspace path"))?
                    .to_path_buf(),
                version: revision.clone(),
                bytes: bytes.as_ref().clone(),
                readonly: *readonly,
            });
        }
    }
    Ok(WorkspaceSnapshot {
        revision: revision(&tree, &root, &excluded)?,
        root,
        excluded_subtrees: request.excluded_subtrees,
        files,
    })
}

fn confined(root: &Path, relative: &Path) -> io::Result<PathBuf> {
    if relative.as_os_str().is_empty()
        || relative
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(invalid(
            "workspace edit path must be a relative project file",
        ));
    }
    let text = relative
        .to_str()
        .ok_or_else(|| invalid("workspace paths must be UTF-8"))?;
    if text.len() > 1024 {
        return Err(invalid("workspace edit path exceeds its budget"));
    }
    Ok(root.join(relative))
}

fn writable_file(tree: &MemoryTree, path: &Path, required: bool) -> io::Result<()> {
    match tree.entries.get(path) {
        Some(MemoryEntry::File {
            readonly: false, ..
        }) => Ok(()),
        None if !required => Ok(()),
        None => Err(not_found(path)),
        Some(MemoryEntry::File { readonly: true, .. }) => Err(io::Error::new(
            ErrorKind::PermissionDenied,
            "workspace file is read-only",
        )),
        Some(MemoryEntry::Directory) => Err(invalid("workspace source edit targets a directory")),
    }
}

pub(super) fn lookup(
    provider: &MemoryFsProvider,
    request: &WorkspacePublicationRequest,
) -> io::Result<Option<WorkspacePublicationOutcome>> {
    let digest = publication_digest(request)?;
    let tree = provider
        .inner
        .lock()
        .expect("memory filesystem lock poisoned");
    let _root = root(&tree, &request.root)?;
    Ok(tree.publications.get(&request.id).map(|prior| {
        if prior.digest == digest {
            WorkspacePublicationOutcome::Published {
                receipt: prior.receipt.clone(),
            }
        } else {
            WorkspacePublicationOutcome::IdReused
        }
    }))
}

pub(super) fn publish(
    provider: &MemoryFsProvider,
    request: WorkspacePublicationRequest,
) -> io::Result<WorkspacePublicationOutcome> {
    publish_with_lease(provider, request, None)
}

pub(super) fn publish_with_lease(
    provider: &MemoryFsProvider,
    request: WorkspacePublicationRequest,
    lease: Option<&crate::execution::ExecutionLease>,
) -> io::Result<WorkspacePublicationOutcome> {
    if request.edits.is_empty() || request.edits.len() > MAX_WORKSPACE_PUBLICATION_EDITS {
        return Err(invalid(
            "workspace publication must contain between 1 and 10000 edits",
        ));
    }
    let size = request
        .edits
        .iter()
        .try_fold(0usize, |size, edit| {
            size.checked_add(match edit {
                WorkspaceEdit::Write { bytes, .. } => bytes.len(),
                _ => 0,
            })
        })
        .ok_or_else(|| invalid("workspace publication size overflow"))?;
    if size > MAX_WORKSPACE_PUBLICATION_BYTES {
        return Err(invalid("workspace publication exceeds its byte budget"));
    }
    let digest = publication_digest(&request)?;
    let mut tree = provider
        .inner
        .lock()
        .expect("memory filesystem lock poisoned");
    if let Some(prior) = tree.publications.get(&request.id) {
        return Ok(if prior.digest == digest {
            WorkspacePublicationOutcome::Published {
                receipt: prior.receipt.clone(),
            }
        } else {
            WorkspacePublicationOutcome::IdReused
        });
    }
    let root = root(&tree, &request.root)?;
    if let Some(lease) = lease {
        crate::execution::require_current(
            tree.executions
                .get(&crate::execution::work_key(&lease.work)?),
            lease,
        )?;
    }
    let excluded = excluded(&root, &request.excluded_subtrees)?;
    let current = revision(&tree, &root, &excluded)?;
    if current != request.expected_revision {
        return Ok(WorkspacePublicationOutcome::Conflict { current });
    }
    if tree.publications.len() >= MAX_WORKSPACE_PUBLICATION_EDITS {
        return Err(invalid("workspace publication receipt budget is exhausted"));
    }
    // All paths, rights and parent transitions are validated on a private tree.
    // One replacement publishes files and the idempotent receipt together.
    let mut next = tree.clone();
    for edit in request.edits {
        let paths = match &edit {
            WorkspaceEdit::Write { path, .. } | WorkspaceEdit::Remove { path } => vec![path],
            WorkspaceEdit::Rename { from, to } => vec![from, to],
        };
        for path in paths {
            let path = confined(&root, path)?;
            if excluded.iter().any(|ignored| path.starts_with(ignored)) {
                return Err(invalid("workspace edit targets an excluded subtree"));
            }
        }
        match edit {
            WorkspaceEdit::Write { path, bytes } => {
                let path = confined(&root, &path)?;
                writable_file(&next, &path, false)?;
                ensure_parent_dirs(&mut next, &path)?;
                let identity = successor_identity(next.entries.get(&path))?;
                next.entries.insert(
                    path,
                    MemoryEntry::File {
                        bytes: Arc::new(bytes),
                        readonly: false,
                        revision: uuid::Uuid::new_v4().to_string(),
                        identity,
                    },
                );
            }
            WorkspaceEdit::Remove { path } => {
                let path = confined(&root, &path)?;
                writable_file(&next, &path, true)?;
                next.entries.remove(&path);
            }
            WorkspaceEdit::Rename { from, to } => {
                let from = confined(&root, &from)?;
                let to = confined(&root, &to)?;
                writable_file(&next, &from, true)?;
                if next.entries.contains_key(&to) {
                    return Err(io::Error::new(
                        ErrorKind::AlreadyExists,
                        "workspace rename destination exists",
                    ));
                }
                ensure_parent_dirs(&mut next, &to)?;
                let entry = next.entries.remove(&from).ok_or_else(|| not_found(&from))?;
                next.entries.insert(to, entry);
            }
        }
    }
    let receipt = WorkspacePublicationReceipt {
        id: request.id,
        revision: revision(&next, &root, &excluded)?,
    };
    next.publications.insert(
        request.id,
        Published {
            digest,
            receipt: receipt.clone(),
        },
    );
    if let Some(lease) = lease {
        crate::execution::require_current(
            tree.executions
                .get(&crate::execution::work_key(&lease.work)?),
            lease,
        )?;
    }
    *tree = next;
    Ok(WorkspacePublicationOutcome::Published { receipt })
}
