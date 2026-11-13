//! Filesystem publication adapters for analysis evidence.

use std::io::ErrorKind;
use std::path::PathBuf;

/// Stage mutable evidence in the already captured result transaction. The outer
/// adapter owns the original provider's commit; this scope never escapes this
/// synchronous call. Native callers without a captured view keep their direct
/// filesystem route.
pub(super) fn write_captured_evidence(
    path: &std::path::Path,
    bytes: Vec<u8>,
) -> Result<(), super::storage::AnalysisPublicationError> {
    use super::storage::AnalysisPublicationError;
    use runmat_filesystem::workspace::{
        WorkspaceCaptureRequest, WorkspaceEdit, WorkspacePublicationId,
        WorkspacePublicationOutcome, WorkspacePublicationRequest,
    };
    let root = path
        .parent()
        .ok_or("Study evidence has no parent directory.")?;
    let name = path.file_name().ok_or("Study evidence has no file name.")?;
    let Some(provider) = super::storage::scoped_artifact_provider() else {
        fs_create_dir_all(root).map_err(AnalysisPublicationError::from_io)?;
        return atomic_write_bytes(&path.to_path_buf(), &bytes).map_err(Into::into);
    };
    let maximum = super::configuration::publication_staging_budget()?;
    let access = provider
        .workspace_access(root, true, maximum)
        .map_err(AnalysisPublicationError::from_io)?;
    access
        .create_dir_all(std::path::Path::new("."))
        .map_err(AnalysisPublicationError::from_io)?;
    let revision = access
        .capture(
            WorkspaceCaptureRequest {
                root: ".".into(),
                excluded_subtrees: Vec::new(),
                max_files: runmat_filesystem::workspace::MAX_WORKSPACE_PUBLICATION_EDITS as u32,
                max_bytes: maximum,
            },
            maximum,
        )
        .map_err(AnalysisPublicationError::from_io)?
        .revision;
    let outcome = access
        .publish(
            WorkspacePublicationRequest {
                id: WorkspacePublicationId::new(),
                root: ".".into(),
                excluded_subtrees: Vec::new(),
                expected_revision: revision,
                edits: vec![WorkspaceEdit::Write {
                    path: name.into(),
                    bytes,
                }],
            },
            maximum,
        )
        .map_err(AnalysisPublicationError::from_io)?;
    match outcome {
        WorkspacePublicationOutcome::Published { .. }
        | WorkspacePublicationOutcome::CommittedPendingRecovery { .. } => Ok(()),
        WorkspacePublicationOutcome::DurabilityIndeterminate { receipt } => {
            Err(AnalysisPublicationError::DurabilityIndeterminate {
                workspace: receipt,
                publication: None,
            })
        }
        WorkspacePublicationOutcome::Conflict { .. } => {
            Err("Study evidence changed during publication.".into())
        }
        WorkspacePublicationOutcome::IdReused => {
            Err("Study evidence publication identity was reused.".into())
        }
    }
}

pub(super) fn atomic_write_bytes(path: &PathBuf, bytes: &[u8]) -> Result<(), String> {
    // Publication also runs in browser workers, which have no process ID.
    // Independent executions need independent staging names, not a wall clock
    // or a counter shared only within one worker.
    let mut nonce = [0_u8; 16];
    getrandom::getrandom(&mut nonce)
        .map_err(|err| format!("failed to name temporary study evidence file: {err}"))?;
    let tmp = path.with_extension(format!("tmp-{:032x}", u128::from_le_bytes(nonce)));
    if let Err(err) = fs_write(&tmp, bytes) {
        let _ = fs_remove_file(&tmp);
        return Err(format!(
            "failed to write temporary study evidence file: {err}"
        ));
    }
    fs_rename(&tmp, path).map_err(|err| {
        let _ = fs_remove_file(&tmp);
        format!("failed to atomically persist study evidence file: {err}")
    })
}

pub(super) fn fs_create_dir_all(path: impl Into<PathBuf>) -> std::io::Result<()> {
    runmat_filesystem::create_dir_all(path.into())
}

fn fs_write(path: impl Into<PathBuf>, bytes: &[u8]) -> std::io::Result<()> {
    runmat_filesystem::write(path.into(), bytes)
}

fn fs_rename(from: impl Into<PathBuf>, to: impl Into<PathBuf>) -> std::io::Result<()> {
    runmat_filesystem::rename(from.into(), to.into())
}

fn fs_remove_file(path: impl Into<PathBuf>) -> std::io::Result<()> {
    match runmat_filesystem::remove_file(path.into()) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == ErrorKind::NotFound => Ok(()),
        Err(err) => Err(err),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evidence_publication_uses_the_provider_and_cleans_refused_staging() {
        let _guard = crate::analysis::tests::analysis_test_guard();
        let _provider_lock = runmat_filesystem::provider_override_lock();
        let _provider = runmat_filesystem::replace_provider(std::sync::Arc::new(
            runmat_filesystem::MemoryFsProvider::new(),
        ));
        let root = PathBuf::from("/evidence");
        fs_create_dir_all(root.clone()).unwrap();
        let path = root.join("study.json");
        atomic_write_bytes(&path, b"first").unwrap();
        atomic_write_bytes(&path, b"replacement").unwrap();
        assert_eq!(runmat_filesystem::read(&path).unwrap(), b"replacement");

        let directory = root.join("directory");
        fs_create_dir_all(directory.clone()).unwrap();
        assert!(atomic_write_bytes(&directory, b"refused")
            .unwrap_err()
            .contains("failed to atomically persist study evidence file"));
        assert!(runmat_filesystem::metadata(&directory).unwrap().is_dir());
        assert_eq!(runmat_filesystem::read_dir(&root).unwrap().len(), 2);
        assert_eq!(runmat_filesystem::read(&path).unwrap(), b"replacement");
    }
}
