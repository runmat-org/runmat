//! An immutable candidate view. Applying source edits here never writes a provider.
use super::{source_digest, FeaSourceSnapshotBuilder, PreparedFeaSourceSnapshot};
use runmat_analysis_language::{LogicalSourcePath, SourceDigest};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::Arc;

pub use runmat_fea_source::{FeaWorkspaceEdit, FeaWorkspaceFile, FeaWorkspaceSnapshot};

#[derive(Debug, thiserror::Error)]
pub enum FeaWorkspaceError {
    #[error("candidate path is outside the captured project: {0}")]
    OutOfScope(String),
    #[error("duplicate captured path: {0}")]
    DuplicatePath(String),
    #[error("candidate source does not exist: {0}")]
    MissingPath(String),
    #[error("rename destination already exists: {0}")]
    DestinationExists(String),
    #[error("captured source has no immutable version: {0}")]
    MissingVersion(String),
    #[error("candidate workspace exceeds its source capture budget")]
    CaptureLimit,
    #[error("candidate source checking failed: {0}")]
    Check(String),
}

pub struct FeaWorkspaceCandidate {
    source_root: PathBuf,
    base: BTreeMap<LogicalSourcePath, Arc<FeaWorkspaceFile>>,
    files: BTreeMap<LogicalSourcePath, Arc<FeaWorkspaceFile>>,
    changed: BTreeSet<LogicalSourcePath>,
}

fn confined(path: &LogicalSourcePath) -> Result<(), FeaWorkspaceError> {
    if path.as_str().len() > 1024 {
        return Err(FeaWorkspaceError::CaptureLimit);
    }
    let normalized = LogicalSourcePath::new(path.as_str())
        .map_err(|_| FeaWorkspaceError::OutOfScope(path.to_string()))?;
    let root = LogicalSourcePath::root();
    if normalized == root || !runmat_analysis_language::logical_path_is_within(&root, &normalized) {
        return Err(FeaWorkspaceError::OutOfScope(path.to_string()));
    }
    Ok(())
}

impl FeaWorkspaceCandidate {
    pub fn new(
        snapshot: FeaWorkspaceSnapshot,
        edits: Vec<FeaWorkspaceEdit>,
    ) -> Result<Self, FeaWorkspaceError> {
        if snapshot.files.len() > runmat_fea_source::MAX_FEA_WORKSPACE_FILES
            || edits.len() > runmat_fea_source::MAX_FEA_WORKSPACE_FILES
        {
            return Err(FeaWorkspaceError::CaptureLimit);
        }
        let mut bytes = 0usize;
        let mut files = BTreeMap::new();
        for file in snapshot.files {
            bytes = bytes
                .checked_add(file.bytes.len())
                .ok_or(FeaWorkspaceError::CaptureLimit)?;
            if bytes > runmat_fea_source::MAX_FEA_CAPTURE_BYTES {
                return Err(FeaWorkspaceError::CaptureLimit);
            }
            confined(&file.path)?;
            if file.version.is_empty() || file.version.len() > 256 {
                return Err(FeaWorkspaceError::MissingVersion(file.path.to_string()));
            }
            if files.contains_key(&file.path) {
                return Err(FeaWorkspaceError::DuplicatePath(file.path.to_string()));
            }
            files.insert(file.path.clone(), Arc::new(file));
        }
        let mut candidate = Self {
            source_root: snapshot.source_root,
            base: files.clone(),
            files,
            changed: BTreeSet::new(),
        };
        for edit in edits {
            match edit {
                FeaWorkspaceEdit::Write { path, bytes } => {
                    confined(&path)?;
                    candidate.changed.insert(path.clone());
                    candidate.files.insert(
                        path.clone(),
                        Arc::new(FeaWorkspaceFile {
                            path,
                            version: String::new(),
                            bytes,
                        }),
                    );
                }
                FeaWorkspaceEdit::Remove { path } => {
                    confined(&path)?;
                    if candidate.files.remove(&path).is_none() {
                        return Err(FeaWorkspaceError::MissingPath(path.to_string()));
                    }
                    candidate.changed.insert(path);
                }
                FeaWorkspaceEdit::Rename { from, to } => {
                    confined(&from)?;
                    confined(&to)?;
                    if candidate.files.contains_key(&to) {
                        return Err(FeaWorkspaceError::DestinationExists(to.to_string()));
                    }
                    let source = candidate
                        .files
                        .remove(&from)
                        .ok_or_else(|| FeaWorkspaceError::MissingPath(from.to_string()))?;
                    candidate.files.insert(
                        to.clone(),
                        Arc::new(FeaWorkspaceFile {
                            path: to.clone(),
                            version: String::new(),
                            bytes: source.bytes.clone(),
                        }),
                    );
                    candidate.changed.insert(from);
                    candidate.changed.insert(to);
                }
            }
        }
        let (source_bytes, geometry_bytes) = candidate
            .files
            .values()
            .try_fold((0usize, 0usize), |(source, geometry), file| {
                let is_source = matches!(
                    std::path::Path::new(file.path.as_str())
                        .extension()
                        .and_then(|extension| extension.to_str()),
                    Some("fea" | "m" | "toml")
                );
                if is_source {
                    Some((source.checked_add(file.bytes.len())?, geometry))
                } else {
                    Some((source, geometry.checked_add(file.bytes.len())?))
                }
            })
            .ok_or(FeaWorkspaceError::CaptureLimit)?;
        if candidate.files.len() > runmat_fea_source::MAX_FEA_WORKSPACE_FILES
            || source_bytes > runmat_fea_source::MAX_FEA_SOURCE_BYTES
            || geometry_bytes > runmat_fea_source::MAX_FEA_GEOMETRY_BYTES
        {
            return Err(FeaWorkspaceError::CaptureLimit);
        }
        Ok(candidate)
    }

    pub fn read(&self, path: &LogicalSourcePath) -> Option<&[u8]> {
        self.files.get(path).map(|file| file.bytes.as_slice())
    }
    pub fn base_version(&self, path: &LogicalSourcePath) -> Option<&str> {
        self.base.get(path).map(|file| file.version.as_str())
    }
    pub fn changed_paths(&self) -> impl Iterator<Item = &LogicalSourcePath> {
        self.changed.iter()
    }
    pub fn files(&self) -> impl Iterator<Item = (&LogicalSourcePath, &[u8])> {
        self.files
            .iter()
            .map(|(path, file)| (path, file.bytes.as_slice()))
    }
    pub fn source_root(&self) -> &std::path::Path {
        &self.source_root
    }

    /// Includes file identity, immutable base version and bytes. Reordering a
    /// capture cannot alter identity; deleting/replacing an import always does.
    pub fn digest(&self) -> SourceDigest {
        let entries = self
            .files
            .iter()
            .map(|(path, file)| (path, &file.version, source_digest(&file.bytes)))
            .collect::<Vec<_>>();
        let base = self
            .base
            .iter()
            .map(|(path, file)| (path, &file.version, source_digest(&file.bytes)))
            .collect::<Vec<_>>();
        source_digest(
            &serde_json::to_vec(&("runmat-fea-candidate-v2", base, &self.changed, entries))
                .expect("source identities are serializable"),
        )
    }

    pub fn prepare(
        &self,
        root: &LogicalSourcePath,
    ) -> Result<PreparedFeaSourceSnapshot, FeaWorkspaceError> {
        confined(root)?;
        let bytes = self
            .read(root)
            .ok_or_else(|| FeaWorkspaceError::MissingPath(root.to_string()))?;
        let mut snapshot = FeaSourceSnapshotBuilder::new(
            &self.source_root,
            FeaSourceSnapshotBuilder::source_module(root.clone(), bytes),
        )
        .map_err(FeaWorkspaceError::Check)?;
        let mut attempted = BTreeSet::new();
        loop {
            let missing = snapshot
                .missing_module_paths()
                .map_err(FeaWorkspaceError::Check)?;
            let pending = missing
                .into_iter()
                .filter(|path| attempted.insert(path.clone()))
                .collect::<Vec<_>>();
            if pending.is_empty() {
                break;
            }
            for path in pending {
                if let Some(bytes) = self.read(&path) {
                    snapshot
                        .add_module(FeaSourceSnapshotBuilder::source_module(path, bytes))
                        .map_err(FeaWorkspaceError::Check)?;
                }
            }
        }
        snapshot.prepare().map_err(FeaWorkspaceError::Check)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn path(value: &str) -> LogicalSourcePath {
        LogicalSourcePath::new(value).unwrap()
    }
    fn capture() -> FeaWorkspaceSnapshot {
        FeaWorkspaceSnapshot { source_root: PathBuf::from("/project"), files: vec![
        FeaWorkspaceFile { path: path("main.fea"), version: "v1".into(), bytes: b"import material steel from \"./materials.fea\"\n".to_vec() },
        FeaWorkspaceFile { path: path("materials.fea"), version: "v2".into(), bytes: b"material steel {\n elastic { youngs_modulus = 200 GPa\n poisson_ratio = 0.3 }\n}\n".to_vec() },
    ] }
    }
    #[test]
    fn tombstones_do_not_fall_back_to_base_imports() {
        let candidate = FeaWorkspaceCandidate::new(
            capture(),
            vec![FeaWorkspaceEdit::Remove {
                path: path("materials.fea"),
            }],
        )
        .unwrap();
        assert!(candidate.read(&path("materials.fea")).is_none());
        assert_eq!(candidate.base_version(&path("materials.fea")), Some("v2"));
        let checked = candidate
            .prepare(&path("main.fea"))
            .unwrap()
            .check(Vec::new(), Vec::new())
            .unwrap();
        assert_eq!(
            checked.check_result().status,
            super::super::FeaCheckStatus::Incomplete
        );
    }
    #[test]
    fn splitting_an_import_is_checked_as_one_candidate() {
        let candidate = FeaWorkspaceCandidate::new(
            capture(),
            vec![
                FeaWorkspaceEdit::Rename {
                    from: path("materials.fea"),
                    to: path("shared.fea"),
                },
                FeaWorkspaceEdit::Write {
                    path: path("main.fea"),
                    bytes: b"import material steel from \"./shared.fea\"\n".to_vec(),
                },
            ],
        )
        .unwrap();
        let checked = candidate
            .prepare(&path("main.fea"))
            .unwrap()
            .check(Vec::new(), Vec::new())
            .unwrap();
        assert_eq!(
            checked.check_result().status,
            super::super::FeaCheckStatus::Valid,
            "{:?}",
            checked.check_result().diagnostics
        );
        assert!(candidate.read(&path("materials.fea")).is_none());
    }
    #[test]
    fn capture_order_is_irrelevant_but_dependency_bytes_are_bound() {
        let first = FeaWorkspaceCandidate::new(capture(), vec![]).unwrap();
        let mut reversed = capture();
        reversed.files.reverse();
        assert_eq!(
            first.digest(),
            FeaWorkspaceCandidate::new(reversed, vec![])
                .unwrap()
                .digest()
        );
        let second = FeaWorkspaceCandidate::new(
            capture(),
            vec![FeaWorkspaceEdit::Write {
                path: path("materials.fea"),
                bytes: b"material steel { nonsense }".to_vec(),
            }],
        )
        .unwrap();
        assert_ne!(first.digest(), second.digest());
    }
    #[test]
    fn host_mount_paths_do_not_change_portable_candidate_identity() {
        let first = FeaWorkspaceCandidate::new(capture(), vec![]).unwrap();
        let mut other = capture();
        other.source_root = PathBuf::from("/Users/alex/Projects/bracket");
        assert_eq!(
            first.digest(),
            FeaWorkspaceCandidate::new(other, vec![]).unwrap().digest()
        );
    }
    #[test]
    fn excessive_capture_and_edit_counts_are_rejected_before_compilation() {
        let mut excessive = capture();
        excessive.files = (0..=runmat_fea_source::MAX_FEA_WORKSPACE_FILES)
            .map(|i| FeaWorkspaceFile {
                path: path(&format!("{i}.fea")),
                version: "v1".into(),
                bytes: vec![],
            })
            .collect();
        assert!(matches!(
            FeaWorkspaceCandidate::new(excessive, vec![]),
            Err(FeaWorkspaceError::CaptureLimit)
        ));
        let edits = vec![
            FeaWorkspaceEdit::Write {
                path: path("source.m"),
                bytes: vec![]
            };
            runmat_fea_source::MAX_FEA_WORKSPACE_FILES + 1
        ];
        assert!(matches!(
            FeaWorkspaceCandidate::new(capture(), edits),
            Err(FeaWorkspaceError::CaptureLimit)
        ));
    }
    #[test]
    fn deleted_and_overwritten_base_versions_remain_in_the_attestation_identity() {
        for edit in [
            FeaWorkspaceEdit::Remove {
                path: path("materials.fea"),
            },
            FeaWorkspaceEdit::Write {
                path: path("materials.fea"),
                bytes: b"material steel {}".to_vec(),
            },
        ] {
            let first = FeaWorkspaceCandidate::new(capture(), vec![edit.clone()]).unwrap();
            let mut different_base = capture();
            different_base.files[1].version = "v3".into();
            let second = FeaWorkspaceCandidate::new(different_base, vec![edit]).unwrap();
            assert_ne!(first.digest(), second.digest());
        }
    }
    #[test]
    fn candidate_rejects_duplicate_capture_and_escape_without_any_provider_operations() {
        let mut duplicate = capture();
        duplicate.files.push(duplicate.files[0].clone());
        assert!(matches!(
            FeaWorkspaceCandidate::new(duplicate, vec![]),
            Err(FeaWorkspaceError::DuplicatePath(_))
        ));
        assert!(matches!(
            FeaWorkspaceCandidate::new(
                capture(),
                vec![FeaWorkspaceEdit::Write {
                    // Logical paths are now host-root-relative, not a reserved
                    // `project/` versus `packages/` namespace. The root itself
                    // is never a writable file in the captured workspace.
                    path: LogicalSourcePath::root(),
                    bytes: vec![]
                }]
            ),
            Err(FeaWorkspaceError::OutOfScope(_))
        ));
        assert!(LogicalSourcePath::new("../outside/source.m").is_err());
    }
}
