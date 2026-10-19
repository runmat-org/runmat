use super::*;
use crate::workspace::*;
use sha2::{Digest, Sha256};
use std::io::Read;

pub(crate) fn capture(
    root: &Path,
    request: WorkspaceCaptureRequest,
) -> io::Result<WorkspaceSnapshot> {
    let _guard = guard(root, true)?;
    capture_locked(root, request)
}

pub(crate) fn capture_locked(
    root: &Path,
    request: WorkspaceCaptureRequest,
) -> io::Result<WorkspaceSnapshot> {
    Ok(inventory(root, request)?.snapshot)
}

pub(crate) fn capture_locked_admitted(
    root: &Path,
    request: WorkspaceCaptureRequest,
    maximum_resident_bytes: u64,
) -> io::Result<WorkspaceSnapshot> {
    let plan = inspection::inspect_admitted(root, request, maximum_resident_bytes)?;
    super::admission::capture(&plan, maximum_resident_bytes)?;
    Ok(materialize(root, plan)?.snapshot)
}

pub(super) fn publication_inventory(
    root: &Path,
    request: WorkspaceCaptureRequest,
    publication: &WorkspacePublicationRequest,
    maximum_resident_bytes: u64,
) -> io::Result<Inventory> {
    let plan = inspection::inspect_admitted(root, request, maximum_resident_bytes)?;
    super::admission::publication(&plan, publication, maximum_resident_bytes)?;
    materialize(root, plan)
}

pub(super) struct Inventory {
    pub snapshot: WorkspaceSnapshot,
    pub directories: Vec<(PathBuf, bool)>,
}

pub(super) fn inventory(root: &Path, request: WorkspaceCaptureRequest) -> io::Result<Inventory> {
    materialize(root, inspection::inspect(root, request)?)
}

fn materialize(root: &Path, plan: inspection::CapturePlan) -> io::Result<Inventory> {
    let inspection::CapturePlan {
        excluded,
        directories,
        files: planned,
    } = plan;
    let mut files = Vec::new();
    files
        .try_reserve_exact(planned.len())
        .map_err(io::Error::other)?;
    for file in planned {
        let path = root.join(&file.path);
        ensure_no_symlink(root, &path)?;
        let mut input = fs::File::open(&path)?;
        let metadata = input.metadata()?;
        if !metadata.is_file()
            || metadata.len() != file.length
            || metadata.permissions().readonly() != file.readonly
        {
            return Err(invalid("workspace input changed after capture admission"));
        }
        let length = usize::try_from(file.length)
            .map_err(|_| invalid("workspace capture length exceeds address space"))?;
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(length).map_err(io::Error::other)?;
        bytes.resize(length, 0);
        input.read_exact(&mut bytes)?;
        let mut trailing = [0u8; 1];
        if input.read(&mut trailing)? != 0 {
            return Err(invalid("workspace input grew after capture admission"));
        }
        let version = hex(&Sha256::digest(&bytes));
        files.push(WorkspaceFile {
            path: file.path,
            version,
            bytes,
            readonly: file.readonly,
        });
    }
    let revision = revision(&excluded, &directories, &files)?;
    Ok(Inventory {
        snapshot: WorkspaceSnapshot {
            root: "/".into(),
            excluded_subtrees: excluded,
            revision,
            files,
        },
        directories,
    })
}

pub(super) fn revision(
    excluded: &[PathBuf],
    directories: &[(PathBuf, bool)],
    files: &[WorkspaceFile],
) -> io::Result<WorkspaceRevision> {
    let mut hash = Sha256::new();
    hash.update(b"runmat-workspace-native-v1");
    hash.update(serde_json::to_vec(&(&excluded, directories)).map_err(io::Error::other)?);
    for file in files {
        let value = serde_json::to_vec(&(&file.path, &file.version, file.readonly))
            .map_err(io::Error::other)?;
        hash.update((value.len() as u64).to_le_bytes());
        hash.update(value);
    }
    Ok(WorkspaceRevision(hash.finalize().into()))
}

pub(super) fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

pub(super) mod inspection;

#[cfg(test)]
mod tests;
