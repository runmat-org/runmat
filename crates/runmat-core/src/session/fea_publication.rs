//! A checked publication handle binds captured source, compiler facts and the
//! provider which arbitrates publication. It cannot be constructed from model
//! output or replaced with a check against live files after writing.
use super::{
    FeaCapturedDependency, FeaWorkspaceCapture, FeaWorkspaceCheckRequest, FeaWorkspaceCheckResult,
    RunMatSession,
};
use runmat_filesystem::{workspace::*, FsProvider};
use runmat_runtime::analysis::{FeaWorkspaceEdit, FeaWorkspaceFile, LogicalSourcePath};
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

pub struct PreparedFeaWorkspacePublication {
    pub(super) provider: Arc<dyn FsProvider>,
    project_revision: Option<runmat_package::ProjectRevision>,
    pub(super) request: WorkspacePublicationRequest,
    check: FeaWorkspaceCheckResult,
    pub(super) original_sources: std::collections::BTreeMap<LogicalSourcePath, String>,
}

impl PreparedFeaWorkspacePublication {
    pub fn check(&self) -> &FeaWorkspaceCheckResult {
        &self.check
    }
    pub fn publication_id(&self) -> WorkspacePublicationId {
        self.request.id
    }
    pub fn base_revision(&self) -> WorkspaceRevision {
        self.request.expected_revision
    }
}

fn relative(path: &LogicalSourcePath) -> Result<PathBuf, String> {
    if path.is_root() {
        return Err("source edit path must name a workspace-relative file".into());
    }
    let path = Path::new(path.as_str());
    if !matches!(
        path.extension().and_then(|value| value.to_str()),
        Some("fea" | "m")
    ) {
        return Err("checked source publication accepts FEA and MATLAB sources".into());
    }
    Ok(path.to_path_buf())
}

impl RunMatSession {
    pub async fn prepare_fea_workspace_publication(
        &mut self,
        edits: Vec<FeaWorkspaceEdit>,
    ) -> Result<PreparedFeaWorkspacePublication, String> {
        self.prepare_fea_workspace_publication_at(edits, None)
            .await?
            .map_err(|_| "unexpected source publication conflict".into())
    }

    pub(super) async fn prepare_fea_workspace_publication_at(
        &mut self,
        edits: Vec<FeaWorkspaceEdit>,
        expected: Option<WorkspaceRevision>,
    ) -> Result<Result<PreparedFeaWorkspacePublication, WorkspaceRevision>, String> {
        if edits.is_empty() || edits.len() > MAX_WORKSPACE_PUBLICATION_EDITS {
            return Err("a source publication must contain between 1 and 10000 edits".into());
        }
        let bytes = edits
            .iter()
            .try_fold(0usize, |total, edit| {
                total.checked_add(match edit {
                    FeaWorkspaceEdit::Write { bytes, .. } => bytes.len(),
                    _ => 0,
                })
            })
            .ok_or("source publication size overflow")?;
        if bytes > runmat_fea_source::MAX_FEA_SOURCE_BYTES {
            return Err("source publication exceeds 32 MiB".into());
        }
        // Validate mutation scope before capture or expensive compilation.
        let publication_edits = edits
            .iter()
            .map(|edit| match edit {
                FeaWorkspaceEdit::Write { path, bytes } => Ok(WorkspaceEdit::Write {
                    path: relative(path)?,
                    bytes: bytes.clone(),
                }),
                FeaWorkspaceEdit::Remove { path } => Ok(WorkspaceEdit::Remove {
                    path: relative(path)?,
                }),
                FeaWorkspaceEdit::Rename { from, to } => Ok(WorkspaceEdit::Rename {
                    from: relative(from)?,
                    to: relative(to)?,
                }),
            })
            .collect::<Result<Vec<_>, String>>()?;
        let provider = runmat_filesystem::current_provider();
        let root = self.project_handoff.as_ref().map_or_else(
            || {
                provider
                    .current_dir_override()
                    .unwrap_or_else(|| PathBuf::from("/"))
            },
            |handoff| handoff.project.workspace_root.clone(),
        );
        let project_revision = self
            .project_handoff
            .as_ref()
            .map(|handoff| handoff.project.revision());
        let capture = provider
            .capture_workspace(WorkspaceCaptureRequest {
                root,
                excluded_subtrees: vec![
                    ".artifacts".into(),
                    ".git".into(),
                    "node_modules".into(),
                    "target".into(),
                ],
                max_files: runmat_fea_source::MAX_FEA_WORKSPACE_FILES as u32,
                max_bytes: runmat_fea_source::MAX_FEA_CAPTURE_BYTES as u64,
            })
            .await
            .map_err(|error| error.to_string())?;
        // Detect stale proposals before applying a rename/remove or compiling
        // against a changed graph. The provider still arbitrates at publication.
        if expected.is_some_and(|revision| revision != capture.revision) {
            return Ok(Err(capture.revision));
        }
        let files = capture
            .files
            .into_iter()
            .map(|file| {
                let path = file
                    .path
                    .to_str()
                    .ok_or("captured source path is not UTF-8")?;
                Ok(FeaWorkspaceFile {
                    path: LogicalSourcePath::new(path.to_owned())
                        .map_err(|error| error.to_string())?,
                    version: file.version,
                    bytes: file.bytes,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        // Retain only changed source originals for bounded review, from the exact
        // capture used by the compiler. Reading live files later would mislead review.
        let changed = edits
            .iter()
            .flat_map(|edit| match edit {
                FeaWorkspaceEdit::Write { path, .. } | FeaWorkspaceEdit::Remove { path } => {
                    vec![path]
                }
                FeaWorkspaceEdit::Rename { from, to } => vec![from, to],
            })
            .collect::<std::collections::BTreeSet<_>>();
        let original_sources = files
            .iter()
            .filter(|file| changed.contains(&file.path))
            .map(|file| {
                Ok((
                    file.path.clone(),
                    std::str::from_utf8(&file.bytes)
                        .map_err(|_| "FEA source review requires UTF-8 source")?
                        .to_owned(),
                ))
            })
            .collect::<Result<std::collections::BTreeMap<_, _>, String>>()?;
        let mut dependencies = Vec::new();
        if let Some(handoff) = &self.project_handoff {
            let mut budget = 0usize;
            for (source, path) in &handoff.project.access_paths {
                if source.package_instance == handoff.project.graph.root {
                    continue;
                }
                let metadata = provider
                    .metadata(path)
                    .await
                    .map_err(|error| error.to_string())?;
                if metadata.len() > (runmat_fea_source::MAX_FEA_SOURCE_BYTES - budget) as u64 {
                    return Err("captured dependencies exceed the source budget".into());
                }
                let bytes = provider
                    .read(path)
                    .await
                    .map_err(|error| error.to_string())?;
                budget = budget
                    .checked_add(bytes.len())
                    .ok_or("dependency capture size overflow")?;
                if budget > runmat_fea_source::MAX_FEA_SOURCE_BYTES {
                    return Err("captured dependencies exceed the source budget".into());
                }
                dependencies.push(FeaCapturedDependency {
                    source: source.clone(),
                    bytes,
                });
            }
        }
        let check = self
            .check_fea_workspace(FeaWorkspaceCheckRequest {
                snapshot: FeaWorkspaceCapture { files },
                edits,
                dependencies,
            })
            .await?;
        Ok(Ok(PreparedFeaWorkspacePublication {
            provider,
            project_revision,
            check,
            original_sources,
            request: WorkspacePublicationRequest {
                id: WorkspacePublicationId::new(),
                root: capture.root,
                excluded_subtrees: capture.excluded_subtrees,
                expected_revision: capture.revision,
                edits: publication_edits,
            },
        }))
    }

    pub async fn publish_fea_workspace(
        &self,
        prepared: &PreparedFeaWorkspacePublication,
    ) -> Result<WorkspacePublicationOutcome, String> {
        if !prepared.check.publication_valid {
            return Err("source candidate did not pass checking; nothing was published".into());
        }
        if prepared.check.environment != crate::program_environment(self.compat_mode()) {
            return Err("the compiler environment changed after checking".into());
        }
        if !Arc::ptr_eq(&prepared.provider, &runmat_filesystem::current_provider())
            || prepared.project_revision
                != self
                    .project_handoff
                    .as_ref()
                    .map(|handoff| handoff.project.revision())
        {
            return Err("the admitted project or provider changed after checking".into());
        }
        prepared
            .provider
            .publish_workspace(prepared.request.clone())
            .await
            .map_err(|error| error.to_string())
    }
}
