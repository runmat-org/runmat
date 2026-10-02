//! Filesystem artifact backend, including exact frequency publication storage.
use super::*;

impl AnalysisArtifactStore for FilesystemAnalysisArtifactStore {
    fn load_observation_publication(
        &self,
        run_id: &str,
    ) -> Result<Option<crate::analysis::ThermalObservationPublication>, String> {
        self.observation_publication(run_id)
    }
    fn persist_thermal_observations(
        &self,
        run_id: &str,
        completed: &crate::analysis::ThermalStudyCompletion,
        source: &crate::analysis::ThermalStudyObjectSource,
    ) -> Result<(), String> {
        self.persist_observations(run_id, completed, source)
    }
    fn load_observation_history_sample(
        &self,
        run_id: &str,
        observation_id: &runmat_analysis_core::ObservationId,
        sample: runmat_analysis_fea::results::field::RecoveredFieldSample,
    ) -> Result<ObservationHistoryLookup, String> {
        self.observation_sample(run_id, observation_id, sample)
    }
    fn persist_run(&self, run: &AnalysisRunResult) -> Result<AnalysisArtifactRecord, String> {
        validate_run_identity(run)?;
        let path = self.run_path(&run.run_id)?;
        self.ensure_runs_directory()?;
        self.ensure_objects_directory()?;
        let prepared = prepare_run_artifact(run, Utc::now().to_rfc3339())?;
        for (digest, chunk) in &prepared.chunks {
            match runmat_filesystem::atomic_create_new(self.object_path(*digest), chunk)
                .map_err(|error| format!("failed to publish immutable field chunk: {error}"))?
            {
                AtomicCreateResult::Published { .. } => {}
                AtomicCreateResult::Existing(existing) if existing == *chunk => {}
                AtomicCreateResult::Existing(_) => {
                    return Err("analysis field chunk digest collision".to_string());
                }
            }
        }
        let bytes = serde_json::to_vec_pretty(&prepared.persisted)
            .map_err(|err| format!("failed to encode run artifact: {err}"))?;
        match runmat_filesystem::atomic_create_new(&path, &bytes)
            .map_err(|error| format!("failed to publish immutable run artifact: {error}"))?
        {
            AtomicCreateResult::Published { .. } => {}
            AtomicCreateResult::Existing(existing_bytes) => {
                let PersistedAnalysisArtifact::Standard(existing) =
                    decode_persisted_artifact(&existing_bytes, &run.run_id)?
                else {
                    return Err("analysis run identity already contains another result kind".into());
                };
                if !same_persisted_content(&existing, &prepared.persisted) {
                    return Err("analysis run identity already contains different content".into());
                }
                return artifact_record(run, existing.created_at);
            }
        }
        prune_filesystem_runs(&self.root)?;
        // Retention may have removed a run whose immutable manifest is cached.
        // A later lookup must observe the filesystem rather than resurrecting it.
        *self
            .manifests
            .write()
            .map_err(|_| "analysis manifest cache lock poisoned")? =
            FilesystemManifestCache::default();

        artifact_record(run, prepared.persisted.created_at)
    }

    fn load_run(&self, run_id: &str) -> Result<Option<AnalysisRunResult>, String> {
        let Some(persisted) = self.read_standard(run_id)? else {
            return Ok(None);
        };
        Ok(Some(materialize_run(&persisted, |digest| {
            self.read_chunk(digest)
        })?))
    }

    fn load_run_metadata(&self, run_id: &str) -> Result<Option<AnalysisRunResult>, String> {
        Ok(self.read_standard(run_id)?.map(|run| run.metadata.clone()))
    }

    fn load_field_inventory(&self, run_id: &str) -> Result<Option<Vec<AnalysisField>>, String> {
        Ok(self
            .read_standard(run_id)?
            .map(|run| persisted_field_inventory(&run)))
    }

    fn load_render_tile_index(&self, run_id: &str) -> Result<Option<RenderTileIndex>, String> {
        Ok(self
            .read_standard(run_id)?
            .and_then(|run| run.render_tiles.clone()))
    }

    fn plan_render_view(
        &self,
        run_id: &str,
        view: Option<RenderViewRequest>,
    ) -> Result<Option<RenderViewSelection>, String> {
        let Some(run) = self.read_standard(run_id)? else {
            return Ok(None);
        };
        run.render_tiles
            .as_ref()
            .map(|index| index.select_view_validated(view))
            .transpose()
    }

    fn load_render_tile(
        &self,
        run_id: &str,
        mesh_index: usize,
        tile_index: usize,
    ) -> Result<Option<RenderTile>, String> {
        let Some(run) = self.read_standard(run_id)? else {
            return Ok(None);
        };
        let Some(descriptor) = run
            .render_tiles
            .as_ref()
            .and_then(|index| index.meshes.get(mesh_index))
            .and_then(|mesh| mesh.tiles.get(tile_index))
        else {
            return Ok(None);
        };
        let bytes = self
            .read_chunk(descriptor.digest)?
            .ok_or("render tile object is missing")?;
        render_tiles::decode_tile(descriptor, &bytes).map(Some)
    }

    fn load_render_lod_node(
        &self,
        run_id: &str,
        mesh_index: usize,
        node_index: usize,
    ) -> Result<Option<RenderTile>, String> {
        let Some(run) = self.read_standard(run_id)? else {
            return Ok(None);
        };
        let Some(descriptor) = run
            .render_tiles
            .as_ref()
            .and_then(|index| index.meshes.get(mesh_index))
            .and_then(|mesh| mesh.lod_nodes.get(node_index))
            .map(|node| &node.tile)
        else {
            return Ok(None);
        };
        let bytes = self
            .read_chunk(descriptor.digest)?
            .ok_or("render LOD object is missing")?;
        render_tiles::decode_tile(descriptor, &bytes).map(Some)
    }

    fn load_render_topology(
        &self,
        run_id: &str,
    ) -> Result<Option<crate::analysis::AnalysisRenderTopology>, String> {
        let Some(index) = self.load_render_tile_index(run_id)? else {
            return Ok(None);
        };
        render_tiles::reconstruct(&index, |digest| self.read_chunk(digest)).map(Some)
    }

    fn load_field_page(
        &self,
        run_id: &str,
        selection: &AnalysisFieldSelection,
        options: AnalysisFieldRequestOptions,
    ) -> Result<AnalysisFieldPageLookup, String> {
        if let Some(run) = self.read_standard(run_id)? {
            return field_page_from_artifact(&run, selection, options, |digest| {
                self.read_chunk(digest)
            });
        }
        let path = self.run_path(run_id)?;
        if !owned_directory_exists(&self.root, "analysis artifact root")?
            || !owned_directory_exists(&self.root.join("runs"), "analysis artifact runs directory")?
        {
            return Ok(AnalysisFieldPageLookup::RunNotFound);
        }
        let bytes = match fs_symlink_metadata(&path) {
            Ok(metadata) if metadata.file_type() == FsFileType::File => {
                fs_read(&path).map_err(|error| format!("failed to read run artifact: {error}"))?
            }
            Ok(metadata) => {
                return Err(format!(
                    "analysis run artifact must be a regular file, found {:?}",
                    metadata.file_type()
                ));
            }
            Err(error) if error.kind() == ErrorKind::NotFound => {
                return Ok(AnalysisFieldPageLookup::RunNotFound);
            }
            Err(error) => return Err(format!("failed to inspect run artifact: {error}")),
        };
        match decode_persisted_artifact(&bytes, run_id)? {
            PersistedAnalysisArtifact::Standard(persisted) => {
                field_page_from_artifact(&persisted, selection, options, |digest| {
                    self.read_chunk(digest)
                })
            }
            PersistedAnalysisArtifact::StructuralFrequency(artifact) => {
                frequency::field_page(&artifact, selection, options, |digest| {
                    self.read_chunk(digest)
                })
            }
            PersistedAnalysisArtifact::AcousticFrequency(artifact) => {
                frequency_sample::field_page(&artifact, selection, options, |digest| {
                    self.read_chunk(digest)
                })
            }
        }
    }

    fn list_runs(&self) -> Result<Vec<AnalysisRunResult>, String> {
        self.list_persisted_runs()?
            .into_iter()
            .filter_map(|artifact| match artifact {
                PersistedAnalysisArtifact::Standard(persisted) => Some(persisted),
                PersistedAnalysisArtifact::StructuralFrequency(_)
                | PersistedAnalysisArtifact::AcousticFrequency(_) => None,
            })
            .map(|persisted| materialize_run(&persisted, |digest| self.read_chunk(digest)))
            .collect()
    }

    fn list_run_metadata(&self) -> Result<Vec<AnalysisRunResult>, String> {
        Ok(self
            .list_persisted_runs()?
            .into_iter()
            .filter_map(|artifact| match artifact {
                PersistedAnalysisArtifact::Standard(persisted) => Some(persisted.metadata),
                PersistedAnalysisArtifact::StructuralFrequency(_)
                | PersistedAnalysisArtifact::AcousticFrequency(_) => None,
            })
            .collect())
    }

    fn persist_structural_frequency(
        &self,
        step: &AnalysisStudyStepRunData,
        object_catalog: &runmat_execution::value::PublishedObjectCatalog,
        artifact_roots: &crate::analysis::StructuralFrequencyArtifactRootIndex,
        source: &dyn runmat_execution::value::PublishedObjectSource,
    ) -> Result<AnalysisArtifactRecord, String> {
        let path = self.run_path(&step.run_id)?;
        self.ensure_runs_directory()?;
        self.ensure_objects_directory()?;
        let prepared = frequency::prepare_artifact(
            step,
            object_catalog,
            artifact_roots,
            Utc::now().to_rfc3339(),
        )?;
        let objects = frequency::ingest_objects(&prepared, source)?;
        for (digest, bytes) in objects {
            match runmat_filesystem::atomic_create_new(self.object_path(digest), &bytes)
                .map_err(|error| format!("failed to publish immutable analysis object: {error}"))?
            {
                AtomicCreateResult::Published { .. } => {}
                AtomicCreateResult::Existing(existing) if existing == bytes => {}
                AtomicCreateResult::Existing(_) => {
                    return Err("analysis object digest collision".into());
                }
            }
        }
        let bytes = serde_json::to_vec_pretty(&prepared)
            .map_err(|error| format!("failed to encode structural frequency artifact: {error}"))?;
        match runmat_filesystem::atomic_create_new(&path, &bytes)
            .map_err(|error| format!("failed to publish immutable run artifact: {error}"))?
        {
            AtomicCreateResult::Published { .. } => {}
            AtomicCreateResult::Existing(existing_bytes) => {
                let PersistedAnalysisArtifact::StructuralFrequency(existing) =
                    decode_persisted_artifact(&existing_bytes, &step.run_id)?
                else {
                    return Err("analysis run identity already contains another result kind".into());
                };
                if !frequency::same_content(&existing, &prepared) {
                    return Err("analysis run identity already contains different content".into());
                }
                return Ok(frequency::artifact_record(&existing));
            }
        }
        prune_filesystem_runs(&self.root)?;
        Ok(frequency::artifact_record(&prepared))
    }

    fn load_structural_frequency(
        &self,
        run_id: &str,
    ) -> Result<Option<AnalysisStudyStepRunData>, String> {
        let path = self.run_path(run_id)?;
        if !owned_directory_exists(&self.root, "analysis artifact root")?
            || !owned_directory_exists(&self.root.join("runs"), "analysis artifact runs directory")?
        {
            return Ok(None);
        }
        let bytes = match fs_symlink_metadata(&path) {
            Ok(metadata) if metadata.file_type() == FsFileType::File => {
                fs_read(&path).map_err(|error| {
                    format!("failed to read structural frequency artifact: {error}")
                })?
            }
            Ok(metadata) => {
                return Err(format!(
                    "analysis run artifact must be a regular file, found {:?}",
                    metadata.file_type()
                ));
            }
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(format!("failed to inspect run artifact: {error}")),
        };
        let PersistedAnalysisArtifact::StructuralFrequency(artifact) =
            decode_persisted_artifact(&bytes, run_id)?
        else {
            return Ok(None);
        };
        Ok(Some(artifact.step))
    }

    fn persist_acoustic_frequency(
        &self,
        step: &AnalysisStudyStepRunData,
        object_catalog: &runmat_execution::value::PublishedObjectCatalog,
        artifact_roots: &crate::analysis::FrequencySampleArtifactRootInventory,
        source: &dyn runmat_execution::value::PublishedObjectSource,
    ) -> Result<AnalysisArtifactRecord, String> {
        let path = self.run_path(&step.run_id)?;
        self.ensure_runs_directory()?;
        self.ensure_objects_directory()?;
        let prepared = frequency_sample::prepare_artifact(
            step,
            object_catalog,
            artifact_roots,
            Utc::now().to_rfc3339(),
        )?;
        let objects = frequency_sample::ingest_objects(&prepared, source)?;
        for (digest, bytes) in objects {
            match runmat_filesystem::atomic_create_new(self.object_path(digest), &bytes)
                .map_err(|error| format!("failed to publish immutable analysis object: {error}"))?
            {
                AtomicCreateResult::Published { .. } => {}
                AtomicCreateResult::Existing(existing) if existing == bytes => {}
                AtomicCreateResult::Existing(_) => {
                    return Err("analysis object digest collision".into());
                }
            }
        }
        let bytes = serde_json::to_vec_pretty(&prepared)
            .map_err(|error| format!("failed to encode acoustic frequency artifact: {error}"))?;
        match runmat_filesystem::atomic_create_new(&path, &bytes)
            .map_err(|error| format!("failed to publish immutable run artifact: {error}"))?
        {
            AtomicCreateResult::Published { .. } => {}
            AtomicCreateResult::Existing(existing_bytes) => {
                let PersistedAnalysisArtifact::AcousticFrequency(existing) =
                    decode_persisted_artifact(&existing_bytes, &step.run_id)?
                else {
                    return Err("analysis run identity already contains another result kind".into());
                };
                if !frequency_sample::same_content(&existing, &prepared) {
                    return Err("analysis run identity already contains different content".into());
                }
                return Ok(frequency_sample::artifact_record(&existing));
            }
        }
        prune_filesystem_runs(&self.root)?;
        Ok(frequency_sample::artifact_record(&prepared))
    }

    fn load_acoustic_frequency(
        &self,
        run_id: &str,
    ) -> Result<Option<AnalysisStudyStepRunData>, String> {
        let path = self.run_path(run_id)?;
        if !owned_directory_exists(&self.root, "analysis artifact root")?
            || !owned_directory_exists(&self.root.join("runs"), "analysis artifact runs directory")?
        {
            return Ok(None);
        }
        let bytes = match fs_symlink_metadata(&path) {
            Ok(metadata) if metadata.file_type() == FsFileType::File => fs_read(&path)
                .map_err(|error| format!("failed to read acoustic frequency artifact: {error}"))?,
            Ok(metadata) => {
                return Err(format!(
                    "analysis run artifact must be a regular file, found {:?}",
                    metadata.file_type()
                ));
            }
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(format!("failed to inspect run artifact: {error}")),
        };
        let PersistedAnalysisArtifact::AcousticFrequency(artifact) =
            decode_persisted_artifact(&bytes, run_id)?
        else {
            return Ok(None);
        };
        Ok(Some(artifact.step))
    }
}
