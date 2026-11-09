//! Filesystem artifact backend, including exact frequency publication storage.
use super::*;

impl access::FilesystemAccess<'_> {
    pub(in crate::analysis::storage) fn commit_publication(
        &self,
        publication: PreparedAnalysisPublication,
    ) -> Result<AnalysisPublicationReceipt, AnalysisPublicationError> {
        publication::filesystem::commit(self, publication)
    }
    pub(in crate::analysis::storage) fn load_publication(
        &self,
        identity: Digest,
        maximum_bytes: u64,
    ) -> Result<Option<serde_json::Value>, AnalysisPublicationError> {
        publication::filesystem::load(self, identity, maximum_bytes)
    }

    pub(in crate::analysis::storage) fn load_publication_manifest_path(
        &self,
        path: &Path,
        maximum_bytes: u64,
    ) -> Result<serde_json::Value, AssessmentStoreError> {
        let directory = path
            .parent()
            .and_then(|parent| parent.file_name())
            .and_then(|name| name.to_str())
            .ok_or_else(|| AssessmentStoreError::Invalid("publication path is malformed".into()))?;
        let identity = format!("sha256:{directory}")
            .parse::<Digest>()
            .map_err(|e| AssessmentStoreError::Invalid(e.to_string()))?;
        if path != publication::filesystem::directory(self, identity).join("manifest.json") {
            return Err(AssessmentStoreError::Invalid(
                "publication path is outside its original store".into(),
            ));
        }
        publication::filesystem::load_bounded(self, identity, maximum_bytes)?
            .ok_or_else(|| AssessmentStoreError::Invalid("publication is not committed".into()))
    }
    pub(in crate::analysis::storage) fn load_observation_publication(
        &self,
        run_id: &str,
    ) -> Result<Option<crate::analysis::ThermalObservationPublication>, AnalysisPublicationError>
    {
        self.observation_publication(run_id)
    }

    pub(in crate::analysis::storage) fn load_observation_history_sample(
        &self,
        run_id: &str,
        observation_id: &runmat_analysis_core::ObservationId,
        sample: runmat_analysis_fea::results::field::RecoveredFieldSample,
    ) -> Result<ObservationHistoryLookup, AnalysisPublicationError> {
        self.observation_sample(run_id, observation_id, sample)
    }
    pub(in crate::analysis::storage) fn persist_prepared_run(
        &self,
        run: &AnalysisRunResult,
        prepared: PreparedRunArtifact,
    ) -> Result<AnalysisArtifactRecord, AnalysisPublicationError> {
        validate_run_identity(run)?;
        self.reserve_standalone(&run.run_id)?;
        let path = self.run_path(&run.run_id)?;
        self.ensure_runs_directory()?;
        self.ensure_objects_directory()?;

        publication::filesystem::reclamation::before_standalone(
            self,
            &prepared.persisted.metadata.run_id,
            &prepared.persisted.metadata.run.identity,
            &prepared.persisted.created_at,
            prepared.chunks.keys().copied(),
        )?;

        for (digest, chunk) in &prepared.chunks {
            match self
                .atomic_create_new(self.object_path(*digest), chunk)
                .map_err(|error| {
                    AnalysisPublicationError::from_io(error)
                        .context("failed to publish immutable field chunk")
                })? {
                AtomicCreateResult::Published { .. } => {}
                AtomicCreateResult::Existing(existing) if existing == *chunk => {}
                AtomicCreateResult::Existing(_) => {
                    return Err("analysis field chunk digest collision".to_string().into());
                }
            }
        }
        let bytes = serde_json::to_vec_pretty(&prepared.persisted)
            .map_err(|err| format!("failed to encode run artifact: {err}"))?;
        crate::analysis::fea_run_admission::FeaRunScope::check_publication_checked()
            .map_err(|denied| AnalysisPublicationError::from(denied.to_string()))?;
        match self.atomic_create_new(&path, &bytes).map_err(|error| {
            AnalysisPublicationError::from_io(error)
                .context("failed to publish immutable run artifact")
        })? {
            AtomicCreateResult::Published { .. } => {}
            AtomicCreateResult::Existing(existing_bytes) => {
                let PersistedAnalysisArtifact::Standard(existing) =
                    self.decode_record(&existing_bytes, &run.run_id)?
                else {
                    return Err("analysis run identity already contains another result kind".into());
                };
                if !same_persisted_content(&existing, &prepared.persisted) {
                    return Err("analysis run identity already contains different content".into());
                }
                return artifact_record(run, existing.created_at).map_err(Into::into);
            }
        }
        prune_filesystem_runs(self, &run.run_id)?;
        // Retention may have removed a run whose immutable manifest is cached.
        // A later lookup must observe the filesystem rather than resurrecting it.
        *self
            .manifests
            .write()
            .map_err(|_| "analysis manifest cache lock poisoned")? =
            FilesystemManifestCache::default();

        artifact_record(run, prepared.persisted.created_at).map_err(Into::into)
    }

    pub(in crate::analysis::storage) fn load_run(
        &self,
        run_id: &str,
    ) -> Result<Option<AnalysisRunResult>, AnalysisPublicationError> {
        let Some(persisted) = self.read_standard(run_id)? else {
            return Ok(None);
        };
        Ok(Some(materialize_run(&persisted, |digest| {
            self.read_chunk(digest).map_err(|e| e.to_string())
        })?))
    }

    pub(in crate::analysis::storage) fn load_run_metadata(
        &self,
        run_id: &str,
    ) -> Result<Option<AnalysisRunResult>, AnalysisPublicationError> {
        Ok(self.read_standard(run_id)?.map(|run| run.metadata.clone()))
    }

    pub(in crate::analysis::storage) fn load_assessment_metadata(
        &self,
        run_id: &str,
        maximum_bytes: u64,
    ) -> Result<Option<AssessmentRunEvidence>, AssessmentStoreError> {
        let Some((path, expected, remaining)) =
            self.assessment_record_path(run_id, maximum_bytes)?
        else {
            return Ok(None);
        };
        let Some(bytes) = self.read_bounded_artifact(&path, remaining)? else {
            return Ok(None);
        };
        let record = self
            .decode_record(&bytes, run_id)
            .map_err(AssessmentStoreError::from)?;
        if let Some(expected) = expected {
            if publication::record_digest(&record).map_err(AssessmentStoreError::Invalid)?
                != expected
            {
                return Err(AssessmentStoreError::Invalid(
                    "run differs from its committed publication".into(),
                ));
            }
        }
        match record {
            PersistedAnalysisArtifact::Standard(run) => {
                Ok(Some(AssessmentRunEvidence::from(&run.metadata)))
            }
            PersistedAnalysisArtifact::StructuralFrequency(run) => frequency::assessment_evidence(
                &run,
                remaining - bytes.len() as u64,
                self.remaining_bytes(),
                |digest, limit| self.read_bounded_artifact(&self.object_path(digest), limit),
            )
            .map(Some),
            PersistedAnalysisArtifact::AcousticFrequency(run) => {
                frequency_sample::assessment_evidence(&run, self.remaining_bytes()).map(Some)
            }
        }
    }

    pub(in crate::analysis::storage) fn load_field_inventory(
        &self,
        run_id: &str,
    ) -> Result<Option<Vec<AnalysisField>>, AnalysisPublicationError> {
        Ok(self
            .read_standard(run_id)?
            .map(|run| persisted_field_inventory(&run)))
    }

    /// The render tiles a run saved: a standard run's, or a structural or acoustic frequency
    /// sweep's.
    pub(in crate::analysis::storage) fn load_render_tile_index(
        &self,
        run_id: &str,
    ) -> Result<Option<RenderTileIndex>, AnalysisPublicationError> {
        if let Some(run) = self.read_standard(run_id)? {
            return Ok(run.render_tiles.clone());
        }
        if let Some(artifact) = self.load_structural_frequency_artifact(run_id)? {
            return Ok(artifact.render_tiles);
        }
        Ok(self
            .load_acoustic_frequency_artifact(run_id)?
            .and_then(|artifact| artifact.render_tiles))
    }

    pub(in crate::analysis::storage) fn plan_render_view(
        &self,
        run_id: &str,
        view: Option<RenderViewRequest>,
    ) -> Result<Option<RenderViewSelection>, AnalysisPublicationError> {
        self.load_render_tile_index(run_id)?
            .map(|index| index.select_view_validated(view))
            .transpose()
            .map_err(Into::into)
    }

    pub(in crate::analysis::storage) fn load_render_tile(
        &self,
        run_id: &str,
        mesh_index: usize,
        tile_index: usize,
    ) -> Result<Option<RenderTile>, AnalysisPublicationError> {
        let Some(index) = self.load_render_tile_index(run_id)? else {
            return Ok(None);
        };
        let Some(descriptor) = index
            .meshes
            .get(mesh_index)
            .and_then(|mesh| mesh.tiles.get(tile_index))
        else {
            return Ok(None);
        };
        let bytes = self
            .read_chunk(descriptor.digest)?
            .ok_or("render tile object is missing")?;
        render_tiles::decode_tile(descriptor, &bytes)
            .map(Some)
            .map_err(Into::into)
    }

    pub(in crate::analysis::storage) fn load_render_lod_node(
        &self,
        run_id: &str,
        mesh_index: usize,
        node_index: usize,
    ) -> Result<Option<RenderTile>, AnalysisPublicationError> {
        let Some(index) = self.load_render_tile_index(run_id)? else {
            return Ok(None);
        };
        let Some(descriptor) = index
            .meshes
            .get(mesh_index)
            .and_then(|mesh| mesh.lod_nodes.get(node_index))
            .map(|node| &node.tile)
        else {
            return Ok(None);
        };
        let bytes = self
            .read_chunk(descriptor.digest)?
            .ok_or("render LOD object is missing")?;
        render_tiles::decode_tile(descriptor, &bytes)
            .map(Some)
            .map_err(Into::into)
    }

    pub(in crate::analysis::storage) fn load_render_topology(
        &self,
        run_id: &str,
    ) -> Result<Option<crate::analysis::AnalysisRenderTopology>, AnalysisPublicationError> {
        let Some(index) = self.load_render_tile_index(run_id)? else {
            return Ok(None);
        };
        render_tiles::reconstruct(&index, |digest| {
            self.read_chunk(digest).map_err(|e| e.to_string())
        })
        .map(Some)
        .map_err(Into::into)
    }

    pub(in crate::analysis::storage) fn load_field_page(
        &self,
        run_id: &str,
        selection: &AnalysisFieldSelection,
        options: AnalysisFieldRequestOptions,
    ) -> Result<AnalysisFieldPageLookup, AnalysisPublicationError> {
        if let Some(run) = self.read_standard(run_id)? {
            return field_page_from_artifact(&run, selection, options, |digest| {
                self.read_chunk(digest).map_err(|e| e.to_string())
            })
            .map_err(Into::into);
        }
        let Some((path, _)) = self.publication_record_path(run_id)? else {
            return Ok(AnalysisFieldPageLookup::RunNotFound);
        };
        if !self.owned_directory_exists(&self.root, "analysis artifact root")?
            || !self.owned_directory_exists(
                &self.root.join("runs"),
                "analysis artifact runs directory",
            )?
        {
            return Ok(AnalysisFieldPageLookup::RunNotFound);
        }
        let bytes = match self.fs_symlink_metadata(&path) {
            Ok(metadata) if metadata.file_type() == FsFileType::File => {
                self.fs_read(&path).map_err(|error| {
                    AnalysisPublicationError::from_io(error).context("failed to read run artifact")
                })?
            }
            Ok(metadata) => {
                return Err(format!(
                    "analysis run artifact must be a regular file, found {:?}",
                    metadata.file_type()
                )
                .into());
            }
            Err(error) if error.kind() == ErrorKind::NotFound => {
                return Ok(AnalysisFieldPageLookup::RunNotFound);
            }
            Err(error) => return Err(format!("failed to inspect run artifact: {error}").into()),
        };
        match self.decode_bound_record(&bytes, run_id)? {
            PersistedAnalysisArtifact::Standard(persisted) => {
                field_page_from_artifact(&persisted, selection, options, |digest| {
                    self.read_chunk(digest).map_err(|e| e.to_string())
                })
                .map_err(Into::into)
            }
            PersistedAnalysisArtifact::StructuralFrequency(artifact) => {
                frequency::field_page(&artifact, selection, options, |digest| {
                    self.read_chunk(digest).map_err(|e| e.to_string())
                })
                .map_err(Into::into)
            }
            PersistedAnalysisArtifact::AcousticFrequency(artifact) => {
                frequency_sample::field_page(&artifact, selection, options, |digest| {
                    self.read_chunk(digest).map_err(|e| e.to_string())
                })
                .map_err(Into::into)
            }
        }
    }

    pub(in crate::analysis::storage) fn list_runs(
        &self,
    ) -> Result<Vec<AnalysisRunResult>, AnalysisPublicationError> {
        self.list_persisted_runs()?
            .into_iter()
            .filter_map(|artifact| match artifact {
                PersistedAnalysisArtifact::Standard(persisted) => Some(persisted),
                PersistedAnalysisArtifact::StructuralFrequency(_)
                | PersistedAnalysisArtifact::AcousticFrequency(_) => None,
            })
            .map(|persisted| {
                materialize_run(&persisted, |digest| {
                    self.read_chunk(digest).map_err(|e| e.to_string())
                })
                .map_err(Into::into)
            })
            .collect()
    }

    pub(in crate::analysis::storage) fn list_run_metadata(
        &self,
    ) -> Result<Vec<AnalysisRunResult>, AnalysisPublicationError> {
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

    pub(in crate::analysis::storage) fn load_structural_frequency(
        &self,
        run_id: &str,
    ) -> Result<Option<AnalysisStudyStepRunData>, AnalysisPublicationError> {
        self.load_structural_frequency_record(run_id)
    }

    pub(in crate::analysis::storage) fn load_acoustic_frequency(
        &self,
        run_id: &str,
    ) -> Result<Option<AnalysisStudyStepRunData>, AnalysisPublicationError> {
        self.load_acoustic_frequency_record(run_id)
    }
}

mod frequency_io;
