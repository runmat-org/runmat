use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

pub fn write_canonical_artifact_pair(
    first_path: &Path,
    first_bytes: &[u8],
    second_path: &Path,
    second_bytes: &[u8],
    force: bool,
) -> Result<(), String> {
    if first_path == second_path {
        return Err("canonical artifact destinations must be distinct".to_string());
    }
    let first = StagedArtifact::prepare(first_path, first_bytes, force)?;
    let second = match StagedArtifact::prepare(second_path, second_bytes, force) {
        Ok(staged) => staged,
        Err(error) => {
            return match first.remove_temporary() {
                Ok(()) => Err(error),
                Err(cleanup) => Err(format!("{error}; cleanup also failed: {cleanup}")),
            };
        }
    };
    let mut staged = [first, second];
    for index in 0..staged.len() {
        if let Err(error) = staged[index].publish() {
            let mut rollback_errors = Vec::new();
            if let Err(rollback) = staged[index].restore() {
                rollback_errors.push(rollback);
            }
            for published in staged[..index].iter_mut().rev() {
                if let Err(rollback) = published.restore() {
                    rollback_errors.push(rollback);
                }
            }
            for pending in &staged[index + 1..] {
                if let Err(rollback) = pending.remove_temporary() {
                    rollback_errors.push(rollback);
                }
            }
            return if rollback_errors.is_empty() {
                Err(error)
            } else {
                Err(format!(
                    "{error}; rollback also failed: {}",
                    rollback_errors.join("; ")
                ))
            };
        }
    }
    let mut cleanup_errors = Vec::new();
    for artifact in &staged {
        if let Err(error) = artifact.remove_backup() {
            cleanup_errors.push(error);
        }
    }
    if !cleanup_errors.is_empty() {
        return Err(format!(
            "canonical artifacts were published but backup cleanup failed: {}",
            cleanup_errors.join("; ")
        ));
    }
    Ok(())
}

struct StagedArtifact {
    target: PathBuf,
    temporary: PathBuf,
    backup: Option<PathBuf>,
    published: bool,
}

impl StagedArtifact {
    fn prepare(path: &Path, bytes: &[u8], force: bool) -> Result<Self, String> {
        let parent = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        runmat_filesystem::create_dir_all(parent)
            .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
        let exists = target_exists(path)?;
        if exists && !force {
            return Err(format!(
                "{} already exists; pass Force, true to replace it",
                path.display()
            ));
        }
        if exists {
            let metadata = runmat_filesystem::metadata(path)
                .map_err(|error| format!("failed to inspect {}: {error}", path.display()))?;
            if metadata.file_type() != runmat_filesystem::FsFileType::File {
                return Err(format!(
                    "{} exists but is not a regular file",
                    path.display()
                ));
            }
        }
        let temporary = sibling_work_path(path, "tmp");
        runmat_filesystem::write(&temporary, bytes).map_err(|error| {
            format!(
                "failed to write temporary artifact {}: {error}",
                temporary.display()
            )
        })?;
        Ok(Self {
            target: path.to_path_buf(),
            temporary,
            backup: exists.then(|| sibling_work_path(path, "backup")),
            published: false,
        })
    }

    fn publish(&mut self) -> Result<(), String> {
        if let Some(backup) = &self.backup {
            runmat_filesystem::rename(&self.target, backup).map_err(|error| {
                format!(
                    "failed to stage replacement of {}: {error}",
                    self.target.display()
                )
            })?;
        }
        if let Err(error) = runmat_filesystem::rename(&self.temporary, &self.target) {
            return Err(format!(
                "failed to publish {}: {error}",
                self.target.display()
            ));
        }
        self.published = true;
        Ok(())
    }

    fn restore(&mut self) -> Result<(), String> {
        let mut errors = Vec::new();
        if self.published {
            if let Err(error) = remove_file_if_present(&self.target) {
                errors.push(error);
            }
            self.published = false;
        } else if let Err(error) = self.remove_temporary() {
            errors.push(error);
        }
        if let Some(backup) = &self.backup {
            if target_exists(backup)? {
                if let Err(error) = runmat_filesystem::rename(backup, &self.target) {
                    errors.push(format!(
                        "failed to restore {} from {}: {error}",
                        self.target.display(),
                        backup.display()
                    ));
                }
            }
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors.join("; "))
        }
    }

    fn remove_temporary(&self) -> Result<(), String> {
        remove_file_if_present(&self.temporary)
    }

    fn remove_backup(&self) -> Result<(), String> {
        let Some(backup) = &self.backup else {
            return Ok(());
        };
        remove_file_if_present(backup)
    }
}

fn sibling_work_path(path: &Path, kind: &str) -> PathBuf {
    static NEXT_TEMPORARY: AtomicU64 = AtomicU64::new(1);
    let sequence = NEXT_TEMPORARY.fetch_add(1, Ordering::Relaxed);
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    parent.join(format!(
        ".runmat-meshing-{}-{sequence}.{kind}",
        std::process::id(),
    ))
}

fn target_exists(path: &Path) -> Result<bool, String> {
    match runmat_filesystem::metadata(path) {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(format!("failed to inspect {}: {error}", path.display())),
    }
}

fn remove_file_if_present(path: &Path) -> Result<(), String> {
    match runmat_filesystem::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(format!("failed to remove {}: {error}", path.display())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pair_write_protects_and_replaces_existing_destinations() {
        let directory = tempfile::TempDir::new().expect("temporary directory");
        let mesh_path = directory.path().join("mesh.cbor");
        let evidence_path = directory.path().join("evidence.cbor");

        write_canonical_artifact_pair(
            &mesh_path,
            b"first mesh",
            &evidence_path,
            b"first evidence",
            false,
        )
        .expect("first publication");
        let error = write_canonical_artifact_pair(
            &mesh_path,
            b"second mesh",
            &evidence_path,
            b"second evidence",
            false,
        )
        .expect_err("replacement must require Force");
        assert!(error.contains("Force, true"));
        assert_eq!(std::fs::read(&mesh_path).unwrap(), b"first mesh");
        assert_eq!(std::fs::read(&evidence_path).unwrap(), b"first evidence");

        write_canonical_artifact_pair(
            &mesh_path,
            b"second mesh",
            &evidence_path,
            b"second evidence",
            true,
        )
        .expect("forced replacement");
        assert_eq!(std::fs::read(&mesh_path).unwrap(), b"second mesh");
        assert_eq!(std::fs::read(&evidence_path).unwrap(), b"second evidence");
    }
}
