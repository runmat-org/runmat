//! One explicit provider-owned access scope covers an artifact operation and its nested reads.
use super::*;
use std::ops::Deref;
mod admission;
pub(in crate::analysis::storage) use admission::Reservation;
use admission::ScopeBudget;

pub(super) struct FilesystemAccess<'a> {
    store: &'a FilesystemAnalysisArtifactStore,
    budget: ScopeBudget,
    workspace: Box<dyn runmat_filesystem::workspace_access::WorkspaceAccess>,
}
impl Deref for FilesystemAccess<'_> {
    type Target = FilesystemAnalysisArtifactStore;
    fn deref(&self) -> &Self::Target {
        self.store
    }
}
impl FilesystemAnalysisArtifactStore {
    pub(super) fn access(&self, writable: bool) -> Result<FilesystemAccess<'_>, String> {
        self.access_with_budget(
            writable,
            crate::analysis::configuration::publication_staging_budget()?,
        )
    }
    pub(super) fn access_with_budget(
        &self,
        writable: bool,
        maximum_inventory_bytes: u64,
    ) -> Result<FilesystemAccess<'_>, String> {
        self.access_with_budget_io(writable, maximum_inventory_bytes)
            .map_err(|e| e.to_string())
    }
    fn access_with_budget_io(
        &self,
        writable: bool,
        maximum_inventory_bytes: u64,
    ) -> std::io::Result<FilesystemAccess<'_>> {
        if maximum_inventory_bytes == 0 {
            return Err(std::io::Error::other(
                "artifact directory budget must be positive",
            ));
        }
        Ok(FilesystemAccess {
            store: self,
            budget: ScopeBudget::new(maximum_inventory_bytes),
            workspace: self.provider.workspace_access(
                &self.root,
                writable,
                maximum_inventory_bytes,
            )?,
        })
    }
}
impl FilesystemAccess<'_> {
    pub(in crate::analysis::storage) fn capture_workspace(
        &self,
        request: runmat_filesystem::workspace::WorkspaceCaptureRequest,
    ) -> Result<runmat_filesystem::workspace::WorkspaceSnapshot, AnalysisPublicationError> {
        self.workspace
            .capture(request, self.remaining_bytes())
            .map_err(AnalysisPublicationError::from_io)
    }
    pub(in crate::analysis::storage) fn publish_workspace(
        &self,
        request: runmat_filesystem::workspace::WorkspacePublicationRequest,
    ) -> Result<runmat_filesystem::workspace::WorkspacePublicationOutcome, AnalysisPublicationError>
    {
        self.workspace
            .publish(request, self.remaining_bytes())
            .map_err(AnalysisPublicationError::from_io)
    }
    fn relative<'p>(&self, path: &'p Path) -> std::io::Result<&'p Path> {
        path.strip_prefix(&self.root).map_err(|_| {
            std::io::Error::new(
                ErrorKind::PermissionDenied,
                "artifact access escapes its original root",
            )
        })
    }
    /// One group commit for the immutable files of an operation: every file is staged, then
    /// one durability barrier and one commit record make all of them visible together.
    pub(super) fn immutable_objects(&self) -> std::io::Result<ImmutableObjects<'_, '_>> {
        Ok(ImmutableObjects {
            access: self,
            batch: self.workspace.immutable_batch()?,
        })
    }
    pub(super) fn fs_read(&self, path: impl AsRef<Path>) -> std::io::Result<Vec<u8>> {
        self.workspace
            .read(self.relative(path.as_ref())?, self.remaining_bytes())
    }
    pub(super) fn fs_symlink_metadata(
        &self,
        path: impl AsRef<Path>,
    ) -> std::io::Result<FsMetadata> {
        self.workspace.metadata(self.relative(path.as_ref())?)
    }
    pub(super) fn fs_create_dir_all(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        self.workspace.create_dir_all(self.relative(path.as_ref())?)
    }
    pub(super) fn fs_remove_file(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        match self.workspace.remove_file(self.relative(path.as_ref())?) {
            Err(e) if e.kind() == ErrorKind::NotFound => Ok(()),
            result => result,
        }
    }
    pub(super) fn fs_remove_dir(&self, path: &Path) -> std::io::Result<()> {
        self.workspace.remove_dir(self.relative(path)?)
    }
    pub(super) fn fs_read_dir(&self, path: impl AsRef<Path>) -> std::io::Result<Vec<DirEntry>> {
        self.workspace.read_dir(
            self.relative(path.as_ref())?,
            &self.root,
            self.remaining_bytes(),
        )
    }
    pub(super) fn atomic_create_new(
        &self,
        path: impl AsRef<Path>,
        bytes: &[u8],
    ) -> std::io::Result<AtomicCreateResult> {
        self.workspace.atomic_create_new(
            self.relative(path.as_ref())?,
            bytes,
            self.remaining_bytes(),
        )
    }
    pub(super) fn owned_directory_exists(
        &self,
        path: &Path,
        label: &str,
    ) -> Result<bool, AnalysisPublicationError> {
        match self.fs_symlink_metadata(path) {
            Ok(metadata) => {
                validate_owned_directory_metadata(metadata, path, label)?;
                Ok(true)
            }
            Err(e) if e.kind() == ErrorKind::NotFound => Ok(false),
            Err(e) => Err(AnalysisPublicationError::from_io(e).context(label)),
        }
    }
    pub(super) fn ensure_owned_directory(
        &self,
        path: &Path,
        label: &str,
    ) -> Result<(), AnalysisPublicationError> {
        if !self.owned_directory_exists(path, label)? {
            self.fs_create_dir_all(path)
                .map_err(AnalysisPublicationError::from_io)?;
        }
        validate_owned_directory_metadata(
            self.fs_symlink_metadata(path)
                .map_err(AnalysisPublicationError::from_io)?,
            path,
            label,
        )
        .map_err(Into::into)
    }
    pub(super) fn read_bounded_artifact(
        &self,
        path: &Path,
        maximum: u64,
    ) -> Result<Option<Vec<u8>>, AssessmentStoreError> {
        assessment::read_bounded_artifact(self, path, maximum)
    }
}

impl AnalysisArtifactStore for FilesystemAnalysisArtifactStore {
    fn filesystem_scope(&self) -> Option<(PathBuf, Arc<dyn runmat_filesystem::FsProvider>)> {
        Some((self.root.clone(), self.provider.clone()))
    }
    fn export_initial_field(
        &self,
        run_id: &str,
        field_id: &str,
        maximum: u64,
    ) -> Result<Option<runmat_analysis_fea::execution::InitialFieldBinding>, AnalysisPublicationError>
    {
        self.access_with_budget(false, maximum)
            .map_err(AnalysisPublicationError::from)?
            .export_original_field(run_id, field_id, maximum)
    }

    fn commit_publication(
        &self,
        publication: PreparedAnalysisPublication,
    ) -> Result<AnalysisPublicationReceipt, AnalysisPublicationError> {
        publication.validate()?;
        let remaining = publication
            .maximum_bytes()
            .checked_sub(publication::resources::admission_bytes(&publication)?)
            .ok_or_else(|| AnalysisPublicationError::ResourceLimit {
                message: "publication exceeds its cumulative resident byte budget".into(),
            })?;
        self.access_with_budget_io(true, remaining)
            .map_err(AnalysisPublicationError::from_io)?
            .commit_publication(publication)
    }
    fn load_publication(
        &self,
        identity: Digest,
        maximum_bytes: u64,
    ) -> Result<Option<serde_json::Value>, String> {
        self.access(false).and_then(|scope| {
            scope
                .load_publication(identity, maximum_bytes)
                .map_err(|error| error.to_string())
        })
    }
    fn load_publication_manifest_path(
        &self,
        path: &Path,
        maximum_bytes: u64,
    ) -> Result<serde_json::Value, AssessmentStoreError> {
        self.access(false)
            .map_err(AssessmentStoreError::Invalid)?
            .load_publication_manifest_path(path, maximum_bytes)
    }
    fn load_observation_publication(
        &self,
        run_id: &str,
    ) -> Result<Option<crate::analysis::ThermalObservationPublication>, String> {
        self.access(false).and_then(|scope| {
            scope
                .load_observation_publication(run_id)
                .map_err(|error| error.to_string())
        })
    }
    fn persist_thermal_observations(
        &self,
        run_id: &str,
        completed: &crate::analysis::ThermalStudyCompletion,
        source: &crate::analysis::ThermalStudyObjectSource,
    ) -> Result<(), String> {
        let operation = || -> Result<_, AnalysisPublicationError> {
            let (attachment, objects) =
                observations::prepare_filesystem_observations(self, run_id, completed, source)?;
            self.access(true)?
                .install_observations(run_id, attachment, objects)
        };
        operation().map_err(|error| error.to_string())
    }
    fn load_observation_history_sample(
        &self,
        run_id: &str,
        observation_id: &runmat_analysis_core::ObservationId,
        sample: runmat_analysis_fea::results::field::RecoveredFieldSample,
    ) -> Result<ObservationHistoryLookup, String> {
        self.access(false).and_then(|scope| {
            scope
                .load_observation_history_sample(run_id, observation_id, sample)
                .map_err(|error| error.to_string())
        })
    }
    fn persist_run(&self, run: &AnalysisRunResult) -> Result<AnalysisArtifactRecord, String> {
        let operation = || -> Result<_, AnalysisPublicationError> {
            validate_run_identity(run)?;
            let prepared = prepare_run_artifact(run, Utc::now().to_rfc3339())?;
            self.access(true)?.persist_prepared_run(run, prepared)
        };
        operation().map_err(|error| error.to_string())
    }
    fn load_run(&self, run_id: &str) -> Result<Option<AnalysisRunResult>, String> {
        self.access(false)
            .and_then(|scope| scope.load_run(run_id).map_err(|error| error.to_string()))
    }
    fn resolve_run_attempt(&self, run_id: &str) -> Result<String, String> {
        self.access(false).and_then(|scope| {
            scope
                .resolve_run_attempt(run_id)
                .map_err(|error| error.to_string())
        })
    }
    fn run_attempts(&self, definition_run_id: &str) -> Result<Vec<String>, String> {
        self.access(false).and_then(|scope| {
            scope
                .run_attempts(definition_run_id)
                .map_err(|error| error.to_string())
        })
    }
    fn load_run_metadata(&self, run_id: &str) -> Result<Option<AnalysisRunResult>, String> {
        self.access(false).and_then(|scope| {
            scope
                .load_run_metadata(run_id)
                .map_err(|error| error.to_string())
        })
    }
    fn load_assessment_metadata(
        &self,
        run_id: &str,
        maximum_bytes: u64,
    ) -> Result<Option<AssessmentRunEvidence>, AssessmentStoreError> {
        self.access(false)
            .map_err(AssessmentStoreError::Invalid)?
            .load_assessment_metadata(run_id, maximum_bytes)
    }
    fn load_field_inventory(&self, run_id: &str) -> Result<Option<Vec<AnalysisField>>, String> {
        self.access(false).and_then(|scope| {
            scope
                .load_field_inventory(run_id)
                .map_err(|error| error.to_string())
        })
    }
    fn load_render_tile_index(&self, run_id: &str) -> Result<Option<RenderTileIndex>, String> {
        self.access(false).and_then(|scope| {
            scope
                .load_render_tile_index(run_id)
                .map_err(|error| error.to_string())
        })
    }
    fn plan_render_view(
        &self,
        run_id: &str,
        view: Option<RenderViewRequest>,
    ) -> Result<Option<RenderViewSelection>, String> {
        self.access(false).and_then(|scope| {
            scope
                .plan_render_view(run_id, view)
                .map_err(|error| error.to_string())
        })
    }
    fn load_render_tile(
        &self,
        run_id: &str,
        mesh_index: usize,
        tile_index: usize,
    ) -> Result<Option<RenderTile>, String> {
        self.access(false).and_then(|scope| {
            scope
                .load_render_tile(run_id, mesh_index, tile_index)
                .map_err(|error| error.to_string())
        })
    }
    fn load_render_lod_node(
        &self,
        run_id: &str,
        mesh_index: usize,
        node_index: usize,
    ) -> Result<Option<RenderTile>, String> {
        self.access(false).and_then(|scope| {
            scope
                .load_render_lod_node(run_id, mesh_index, node_index)
                .map_err(|error| error.to_string())
        })
    }
    fn load_render_topology(
        &self,
        run_id: &str,
    ) -> Result<Option<crate::analysis::AnalysisRenderTopology>, String> {
        self.access(false).and_then(|scope| {
            scope
                .load_render_topology(run_id)
                .map_err(|error| error.to_string())
        })
    }
    fn load_field_page(
        &self,
        run_id: &str,
        selection: &AnalysisFieldSelection,
        options: AnalysisFieldRequestOptions,
    ) -> Result<AnalysisFieldPageLookup, String> {
        self.access(false).and_then(|scope| {
            scope
                .load_field_page(run_id, selection, options)
                .map_err(|error| error.to_string())
        })
    }
    fn list_runs(&self) -> Result<Vec<AnalysisRunResult>, String> {
        self.access(false)
            .and_then(|scope| scope.list_runs().map_err(|error| error.to_string()))
    }
    fn list_run_metadata(&self) -> Result<Vec<AnalysisRunResult>, String> {
        self.access(false)
            .and_then(|scope| scope.list_run_metadata().map_err(|error| error.to_string()))
    }
    fn persist_structural_frequency(
        &self,
        step: &AnalysisStudyStepRunData,
        object_catalog: &runmat_execution::value::PublishedObjectCatalog,
        artifact_roots: &crate::analysis::StructuralFrequencyArtifactRootIndex,
        render_topology: Option<&crate::analysis::AnalysisRenderTopology>,
        source: &dyn runmat_execution::value::PublishedObjectSource,
    ) -> Result<AnalysisArtifactRecord, String> {
        let operation = || -> Result<_, AnalysisPublicationError> {
            let (prepared, tile_objects) = frequency::prepare_artifact(
                step,
                object_catalog,
                artifact_roots,
                render_topology,
                Utc::now().to_rfc3339(),
            )?;
            let objects = frequency::ingest_objects(&prepared, tile_objects, source)?;
            self.access(true)?
                .persist_structural_frequency_record(prepared, objects)
        };
        operation().map_err(|error| error.to_string())
    }
    fn load_structural_frequency(
        &self,
        run_id: &str,
    ) -> Result<Option<AnalysisStudyStepRunData>, String> {
        self.access(false).and_then(|scope| {
            scope
                .load_structural_frequency(run_id)
                .map_err(|error| error.to_string())
        })
    }
    fn persist_acoustic_frequency(
        &self,
        step: &AnalysisStudyStepRunData,
        object_catalog: &runmat_execution::value::PublishedObjectCatalog,
        artifact_roots: &crate::analysis::FrequencySampleArtifactRootInventory,
        source: &dyn runmat_execution::value::PublishedObjectSource,
    ) -> Result<AnalysisArtifactRecord, String> {
        let operation = || -> Result<_, AnalysisPublicationError> {
            let prepared = frequency_sample::prepare_artifact(
                step,
                object_catalog,
                artifact_roots,
                Utc::now().to_rfc3339(),
            )?;
            let objects = frequency_sample::ingest_objects(&prepared, source)?;
            self.access(true)?
                .persist_acoustic_frequency_record(prepared, objects)
        };
        operation().map_err(|error| error.to_string())
    }
    fn load_acoustic_frequency(
        &self,
        run_id: &str,
    ) -> Result<Option<AnalysisStudyStepRunData>, String> {
        self.access(false).and_then(|scope| {
            scope
                .load_acoustic_frequency(run_id)
                .map_err(|error| error.to_string())
        })
    }
}

mod records;

#[cfg(test)]
mod tests;

/// Immutable files staged under one access scope and committed together.
pub(super) struct ImmutableObjects<'s, 'a> {
    access: &'s FilesystemAccess<'a>,
    batch: Box<dyn runmat_filesystem::workspace_access::ImmutableBatch + 's>,
}
impl ImmutableObjects<'_, '_> {
    /// Stages one immutable file; an existing name must already hold exactly these bytes.
    pub(super) fn put(
        &mut self,
        path: impl AsRef<Path>,
        bytes: &[u8],
    ) -> Result<(), AnalysisPublicationError> {
        match self
            .batch
            .stage(
                self.access
                    .relative(path.as_ref())
                    .map_err(AnalysisPublicationError::from_io)?,
                bytes,
                self.access.remaining_bytes(),
            )
            .map_err(AnalysisPublicationError::from_io)?
        {
            runmat_filesystem::workspace_access::ImmutableStage::Staged => Ok(()),
            runmat_filesystem::workspace_access::ImmutableStage::Existing(existing)
                if existing == bytes =>
            {
                Ok(())
            }
            runmat_filesystem::workspace_access::ImmutableStage::Existing(_) => {
                Err("immutable publication content collision".into())
            }
        }
    }
    pub(super) fn commit(self) -> Result<(), AnalysisPublicationError> {
        self.batch
            .commit()
            .map_err(AnalysisPublicationError::from_io)
    }
}
