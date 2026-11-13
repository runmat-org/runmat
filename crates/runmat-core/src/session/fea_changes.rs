//! Durable proposals are source edits, not serialized compiler attestations.
//! Applying rechecks the proposal through the same captured compiler and provider
//! CAS boundary. A recovered record cannot authorize unchecked publication.
use super::RunMatSession;
use runmat_fea_source::{
    FeaChangeSetCheck, FeaChangeSetId, FeaChangeSetPublication, FeaChangeSetRequest,
    FeaChangeSetReview, FeaChangeSetReviewRequest, FeaSourceChangeSummary, FeaSourceExcerpt,
    FeaWorkspaceEdit,
};
use runmat_filesystem::{workspace::*, AtomicCreateResult, FsProvider};
use serde::{Deserialize, Serialize};
use std::{path::PathBuf, sync::Arc};

const DIRECTORY: &str = ".artifacts/fea-change-sets";
const MAX_RECORD_BYTES: u64 = 192 * 1024 * 1024;
const MAX_STAGES: usize = 1_000;
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Stage {
    #[serde(default = "legacy_path_format")]
    path_format_version: u16,
    id: FeaChangeSetId,
    publication: WorkspacePublicationRequest,
    project: Option<runmat_package::ProjectRevision>,
    environment: runmat_execution::ProgramEnvironment,
    original_sources:
        std::collections::BTreeMap<runmat_runtime::analysis::LogicalSourcePath, String>,
}
impl Stage {
    fn source_edits(&self) -> Result<Vec<FeaWorkspaceEdit>, String> {
        self.publication
            .edits
            .iter()
            .map(|edit| {
                let logical = |path: &std::path::Path| {
                    runmat_runtime::analysis::LogicalSourcePath::new(
                        path.to_str()
                            .ok_or("change set path is not UTF-8")?
                            .to_owned(),
                    )
                    .map_err(|error| error.to_string())
                };
                Ok(match edit {
                    WorkspaceEdit::Write { path, bytes } => FeaWorkspaceEdit::Write {
                        path: logical(path)?,
                        bytes: bytes.clone(),
                    },
                    WorkspaceEdit::Remove { path } => FeaWorkspaceEdit::Remove {
                        path: logical(path)?,
                    },
                    WorkspaceEdit::Rename { from, to } => FeaWorkspaceEdit::Rename {
                        from: logical(from)?,
                        to: logical(to)?,
                    },
                })
            })
            .collect::<Result<Vec<_>, String>>()
    }
}

fn path(root: &std::path::Path, id: FeaChangeSetId) -> PathBuf {
    root.join(DIRECTORY)
        .join(format!("{}.json", uuid::Uuid::from_bytes(id.0)))
}
impl RunMatSession {
    fn change_set_scope(
        &self,
    ) -> (
        Arc<dyn FsProvider>,
        PathBuf,
        Option<runmat_package::ProjectRevision>,
    ) {
        let provider = runmat_filesystem::current_provider();
        let root = self.project_handoff.as_ref().map_or_else(
            || {
                provider
                    .current_dir_override()
                    .unwrap_or_else(|| "/".into())
            },
            |handoff| handoff.project.workspace_root.clone(),
        );
        let revision = self
            .project_handoff
            .as_ref()
            .map(|handoff| handoff.project.revision());
        (provider, root, revision)
    }
    async fn read_change_set(&self, id: FeaChangeSetId) -> Result<Stage, String> {
        let (provider, root, _) = self.change_set_scope();
        let target = path(&root, id);
        let metadata = provider
            .metadata(&target)
            .await
            .map_err(|error| error.to_string())?;
        if !metadata.is_file() || metadata.len() > MAX_RECORD_BYTES {
            return Err("source change set exceeds its persisted record budget".into());
        }
        let bytes = provider
            .read(&target)
            .await
            .map_err(|error| error.to_string())?;
        if bytes.len() as u64 > MAX_RECORD_BYTES {
            return Err("source change set exceeds its persisted record budget".into());
        }
        let mut stage: Stage = serde_json::from_slice(&bytes).map_err(|error| error.to_string())?;
        stage.migrate_paths()?;
        if stage.id != id || stage.publication.id.0 != id.0 || stage.publication.root != root {
            return Err(
                "source change set belongs to a different project or compiler environment".into(),
            );
        }
        Ok(stage)
    }
    fn validate_change_set_environment(&self, stage: &Stage) -> Result<(), String> {
        let (_, _, project) = self.change_set_scope();
        if stage.project != project
            || stage.environment != crate::program_environment(self.compat_mode())
        {
            return Err(
                "source change set belongs to a different admitted project or compiler environment"
                    .into(),
            );
        }
        Ok(())
    }
    pub async fn stage_fea_changes(
        &mut self,
        request: FeaChangeSetRequest,
    ) -> Result<FeaChangeSetCheck, String> {
        let base = match request.base {
            Some(id) => {
                let stage = self.read_change_set(id).await?;
                self.validate_change_set_environment(&stage)?;
                Some(stage)
            }
            None => None,
        };
        let edits = request
            .changes
            .into_iter()
            .map(FeaWorkspaceEdit::from)
            .collect::<Vec<_>>();
        let prepared = self
            .prepare_fea_workspace_publication_at(edits, base.as_ref().map(|prior| prior.publication.expected_revision))
            .await?
            .map_err(|_| "the workspace changed after this proposal; inspect the changes and create a new proposal")?;
        let (provider, root, project) = self.change_set_scope();
        let directory = root.join(DIRECTORY);
        provider
            .create_dir_all(&directory)
            .await
            .map_err(|error| error.to_string())?;
        if provider
            .read_dir(&directory)
            .await
            .map_err(|error| error.to_string())?
            .len()
            >= MAX_STAGES
        {
            return Err(
                "pending source change set budget exhausted; discard proposals no longer needed"
                    .into(),
            );
        }
        let id = FeaChangeSetId(prepared.publication_id().0);
        let stage = Stage {
            path_format_version: 2,
            id,
            publication: prepared.request.clone(),
            original_sources: prepared.original_sources.clone(),
            project,
            environment: crate::program_environment(self.compat_mode()),
        };
        let bytes = serde_json::to_vec(&stage).map_err(|error| error.to_string())?;
        if bytes.len() as u64 > MAX_RECORD_BYTES {
            return Err("source proposal exceeds its persisted record budget".into());
        }
        match provider
            .atomic_create_new(&path(&root, id), &bytes)
            .await
            .map_err(|error| error.to_string())?
        {
            AtomicCreateResult::Published { .. } => {}
            AtomicCreateResult::Existing(_) => {
                return Err("source proposal identity already exists".into())
            }
        }
        Ok(FeaChangeSetCheck {
            id,
            roots: prepared.check().roots.clone(),
            publication_valid: prepared.check().publication_valid,
        })
    }
    pub async fn publish_fea_changes(
        &mut self,
        id: FeaChangeSetId,
    ) -> Result<FeaChangeSetPublication, String> {
        let stage = self.read_change_set(id).await?;
        // Recovered stage data is never an attestation. Check the current complete
        // candidate, then fence publication on the original captured workspace.
        let (provider, _, _) = self.change_set_scope();
        if let Some(outcome) = provider
            .lookup_workspace_publication(&stage.publication)
            .await
            .map_err(|error| error.to_string())?
        {
            return Ok(publication(outcome));
        }
        self.validate_change_set_environment(&stage)?;
        let edits = stage.source_edits()?;
        let mut prepared = match self
            .prepare_fea_workspace_publication_at(edits, Some(stage.publication.expected_revision))
            .await?
        {
            Ok(prepared) => prepared,
            Err(current) => {
                return Ok(FeaChangeSetPublication::Conflict {
                    current: runmat_runtime::analysis::SourceDigest(current.0),
                })
            }
        };
        if !prepared.check().publication_valid {
            return Ok(FeaChangeSetPublication::Invalid {
                roots: prepared.check().roots.clone(),
            });
        }
        prepared.request.id = WorkspacePublicationId(id.0);
        prepared.request.expected_revision = stage.publication.expected_revision;
        Ok(publication(self.publish_fea_workspace(&prepared).await?))
    }

    /// Rebuild a retained draft against its original workspace fence. This is
    /// read-only inspection, never publication or execution authority.
    pub async fn inspect_fea_changes(
        &mut self,
        id: FeaChangeSetId,
    ) -> Result<FeaChangeSetCheck, String> {
        let stage = self.read_change_set(id).await?;
        self.validate_change_set_environment(&stage)?;
        let prepared = self.prepare_fea_workspace_publication_at(
            stage.source_edits()?, Some(stage.publication.expected_revision),
        ).await?.map_err(|_| "The workspace changed since this draft was captured. Refresh the draft before previewing or running it.")?;
        Ok(FeaChangeSetCheck {
            id,
            roots: prepared.check().roots.clone(),
            publication_valid: prepared.check().publication_valid,
        })
    }

    pub async fn review_fea_changes(
        &self,
        request: FeaChangeSetReviewRequest,
    ) -> Result<FeaChangeSetReview, String> {
        let stage = self.read_change_set(request.id).await?;
        self.validate_change_set_environment(&stage)?;
        let edit = stage
            .publication
            .edits
            .get(request.change_index as usize)
            .ok_or("source change index is out of range")?;
        let logical = |path: &std::path::Path| {
            runmat_runtime::analysis::LogicalSourcePath::new(
                path.to_str().ok_or("source path is not UTF-8")?.to_owned(),
            )
            .map_err(|error| error.to_string())
        };
        let (change, before_path, after) = match edit {
            WorkspaceEdit::Write { path, bytes } => {
                let path = logical(path)?;
                let after =
                    std::str::from_utf8(bytes).map_err(|_| "proposed source is not UTF-8")?;
                (
                    FeaSourceChangeSummary::Write { path: path.clone() },
                    path,
                    Some(after),
                )
            }
            WorkspaceEdit::Remove { path } => {
                let path = logical(path)?;
                (
                    FeaSourceChangeSummary::Remove { path: path.clone() },
                    path,
                    None,
                )
            }
            WorkspaceEdit::Rename { from, to } => {
                let from = logical(from)?;
                (
                    FeaSourceChangeSummary::Rename {
                        from: from.clone(),
                        to: logical(to)?,
                    },
                    from.clone(),
                    stage.original_sources.get(&from).map(String::as_str),
                )
            }
        };
        Ok(FeaChangeSetReview {
            id: stage.id,
            base_revision: runmat_runtime::analysis::SourceDigest(
                stage.publication.expected_revision.0,
            ),
            change_index: request.change_index,
            change_count: stage.publication.edits.len() as u32,
            change,
            before: stage
                .original_sources
                .get(&before_path)
                .map(|text| excerpt(text, request.before_offset))
                .transpose()?,
            after: after
                .map(|text| excerpt(text, request.after_offset))
                .transpose()?,
        })
    }

    pub async fn discard_fea_changes(&self, id: FeaChangeSetId) -> Result<(), String> {
        self.read_change_set(id).await?;
        let (provider, root, _) = self.change_set_scope();
        provider
            .remove_file(&path(&root, id))
            .await
            .map_err(|error| error.to_string())
    }
}

fn publication(outcome: WorkspacePublicationOutcome) -> FeaChangeSetPublication {
    match outcome {
        WorkspacePublicationOutcome::DurabilityIndeterminate { receipt } => {
            FeaChangeSetPublication::DurabilityIndeterminate {
                id: FeaChangeSetId(receipt.id.0),
                revision: runmat_runtime::analysis::SourceDigest(receipt.revision.0),
            }
        }
        WorkspacePublicationOutcome::CommittedPendingRecovery { receipt } => {
            FeaChangeSetPublication::CommittedPendingRecovery {
                id: FeaChangeSetId(receipt.id.0),
                revision: runmat_runtime::analysis::SourceDigest(receipt.revision.0),
            }
        }
        WorkspacePublicationOutcome::Published { receipt } => FeaChangeSetPublication::Published {
            id: FeaChangeSetId(receipt.id.0),
            revision: runmat_runtime::analysis::SourceDigest(receipt.revision.0),
        },
        WorkspacePublicationOutcome::Conflict { current } => FeaChangeSetPublication::Conflict {
            current: runmat_runtime::analysis::SourceDigest(current.0),
        },
        WorkspacePublicationOutcome::IdReused => FeaChangeSetPublication::IdReused,
    }
}

fn excerpt(text: &str, offset: u32) -> Result<FeaSourceExcerpt, String> {
    let start = offset as usize;
    if start > text.len() || !text.is_char_boundary(start) {
        return Err("source review offset is outside a UTF-8 character boundary".into());
    }
    let mut end = (start + 32 * 1024).min(text.len());
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    Ok(FeaSourceExcerpt {
        offset,
        total_bytes: text.len() as u32,
        text: text[start..end].to_owned(),
        next_offset: (end < text.len()).then_some(end as u32),
    })
}

fn legacy_path_format() -> u16 {
    1
}
impl Stage {
    fn migrate_paths(&mut self) -> Result<(), String> {
        match self.path_format_version {
            2 => Ok(()),
            1 => {
                self.original_sources = std::mem::take(&mut self.original_sources)
                    .into_iter()
                    .map(|(path, bytes)| {
                        let relative = path
                            .as_str()
                            .strip_prefix("project/")
                            .ok_or("legacy source record has an invalid source namespace")?;
                        let path =
                            runmat_runtime::analysis::LogicalSourcePath::new(relative.to_owned())
                                .map_err(|error| error.to_string())?;
                        if path.is_root() {
                            return Err("legacy source record must name a file".into());
                        }
                        Ok((path, bytes))
                    })
                    .collect::<Result<_, String>>()?;
                self.path_format_version = 2;
                Ok(())
            }
            _ => Err("unsupported source proposal path format".into()),
        }
    }
}

#[cfg(test)]
mod publication_tests {
    use super::*;

    #[test]
    fn source_change_retains_uncertain_original_receipt_as_a_distinct_disposition() {
        let id = WorkspacePublicationId([31; 16]);
        let revision = runmat_filesystem::workspace::WorkspaceRevision([79; 32]);
        let outcome = publication(WorkspacePublicationOutcome::DurabilityIndeterminate {
            receipt: runmat_filesystem::workspace::WorkspacePublicationReceipt { id, revision },
        });
        assert!(
            matches!(outcome, FeaChangeSetPublication::DurabilityIndeterminate {
            id: original_id, revision: original_revision,
        } if original_id.0 == id.0 && original_revision.0 == revision.0)
        );
    }
}
