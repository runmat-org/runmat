//! In-memory artifact backend, including exact frequency publication storage.
use super::*;
mod assessment;
mod installation;

impl InMemoryAnalysisArtifactStore {
    pub fn new() -> Self {
        Self {
            retain_standalone: true,
            publications: RwLock::new(HashMap::new()),
            publication_runs: RwLock::new(HashMap::new()),
            observations: RwLock::new(HashMap::new()),
            runs: RwLock::new(HashMap::new()),
            chunks: RwLock::new(HashMap::new()),
            streamed: RwLock::new(std::collections::HashSet::new()),
        }
    }
    pub(super) fn staging() -> Self {
        Self {
            retain_standalone: false,
            ..Self::new()
        }
    }
    pub(super) fn persist_prepared_run(
        &self,
        run: &AnalysisRunResult,
        prepared: PreparedRunArtifact,
    ) -> Result<AnalysisArtifactRecord, String> {
        let created_at = self.install_standalone(
            &run.run_id,
            PersistedAnalysisArtifact::Standard(Box::new(prepared.persisted)),
            prepared.chunks.into_iter().collect(),
        )?;
        artifact_record(run, created_at)
    }
}

impl AnalysisArtifactStore for InMemoryAnalysisArtifactStore {
    fn export_initial_field(
        &self,
        run_id: &str,
        field_id: &str,
        maximum: u64,
    ) -> Result<Option<runmat_analysis_fea::execution::InitialFieldBinding>, AnalysisPublicationError>
    {
        self.export_original_field(run_id, field_id, maximum)
    }

    fn commit_publication(
        &self,
        publication: PreparedAnalysisPublication,
    ) -> Result<AnalysisPublicationReceipt, AnalysisPublicationError> {
        publication::memory::commit(self, publication).map_err(Into::into)
    }
    fn load_publication(
        &self,
        identity: Digest,
        maximum_bytes: u64,
    ) -> Result<Option<serde_json::Value>, String> {
        publication::memory::load(self, identity, maximum_bytes).map_err(|e| e.to_string())
    }

    fn load_publication_manifest_path(
        &self,
        _: &Path,
        _: u64,
    ) -> Result<serde_json::Value, AssessmentStoreError> {
        Err(AssessmentStoreError::Invalid(
            "memory publication has no filesystem manifest".into(),
        ))
    }
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
        let prepared = prepare_run_artifact(run, Utc::now().to_rfc3339())?;
        self.persist_prepared_run(run, prepared)
    }

    fn load_run(&self, run_id: &str) -> Result<Option<AnalysisRunResult>, String> {
        validate_run_id_token(run_id)?;
        let guard = self
            .runs
            .read()
            .map_err(|_| "analysis artifact store lock poisoned".to_string())?;
        let Some(PersistedAnalysisArtifact::Standard(persisted)) = guard.get(run_id) else {
            return Ok(None);
        };
        validate_persisted_run(persisted, run_id)?;
        let chunks = self
            .chunks
            .read()
            .map_err(|_| "analysis field chunk store lock poisoned".to_string())?;
        Ok(Some(materialize_run(persisted, |digest| {
            Ok(chunks.get(&digest).map(|bytes| bytes.to_vec()))
        })?))
    }

    fn load_run_metadata(&self, run_id: &str) -> Result<Option<AnalysisRunResult>, String> {
        validate_run_id_token(run_id)?;
        let guard = self
            .runs
            .read()
            .map_err(|_| "analysis artifact store lock poisoned".to_string())?;
        let Some(PersistedAnalysisArtifact::Standard(persisted)) = guard.get(run_id) else {
            return Ok(None);
        };
        validate_persisted_run(persisted, run_id)?;
        Ok(Some(persisted.metadata.clone()))
    }

    fn load_assessment_metadata(
        &self,
        run_id: &str,
        maximum_bytes: u64,
    ) -> Result<Option<AssessmentRunEvidence>, AssessmentStoreError> {
        // This public limit is encoded artifact bytes. Project standalone decode work
        // through the shared codec; prepared publication supplies its actual remaining host
        // allowance to the same implementation instead.
        let resident =
            runmat_canonical_codec::projected_canonical_materialization_resident_bytes_estimate(
                maximum_bytes,
                3,
            )
            .ok_or(AssessmentStoreError::ResourceLimit)?;
        self.load_assessment_metadata_with_workspace(run_id, maximum_bytes, resident)
    }

    fn load_field_inventory(&self, run_id: &str) -> Result<Option<Vec<AnalysisField>>, String> {
        validate_run_id_token(run_id)?;
        let guard = self
            .runs
            .read()
            .map_err(|_| "analysis artifact store lock poisoned".to_string())?;
        let Some(PersistedAnalysisArtifact::Standard(persisted)) = guard.get(run_id) else {
            return Ok(None);
        };
        validate_persisted_run(persisted, run_id)?;
        Ok(Some(persisted_field_inventory(persisted)))
    }

    fn load_render_tile_index(&self, run_id: &str) -> Result<Option<RenderTileIndex>, String> {
        validate_run_id_token(run_id)?;
        let runs = self
            .runs
            .read()
            .map_err(|_| "analysis artifact store lock poisoned")?;
        Ok(match runs.get(run_id) {
            Some(PersistedAnalysisArtifact::Standard(run)) => run.render_tiles.clone(),
            Some(PersistedAnalysisArtifact::StructuralFrequency(artifact)) => {
                artifact.render_tiles.clone()
            }
            Some(PersistedAnalysisArtifact::AcousticFrequency(artifact)) => {
                artifact.render_tiles.clone()
            }
            _ => None,
        })
    }

    fn plan_render_view(
        &self,
        run_id: &str,
        view: Option<RenderViewRequest>,
    ) -> Result<Option<RenderViewSelection>, String> {
        self.load_render_tile_index(run_id)?
            .map(|index| index.select_view_validated(view))
            .transpose()
    }

    fn load_render_tile(
        &self,
        run_id: &str,
        mesh_index: usize,
        tile_index: usize,
    ) -> Result<Option<RenderTile>, String> {
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
        let chunks = self
            .chunks
            .read()
            .map_err(|_| "analysis field chunk store lock poisoned")?;
        let bytes = chunks
            .get(&descriptor.digest)
            .ok_or("render tile object is missing")?;
        render_tiles::decode_tile(descriptor, bytes).map(Some)
    }

    fn load_render_lod_node(
        &self,
        run_id: &str,
        mesh_index: usize,
        node_index: usize,
    ) -> Result<Option<RenderTile>, String> {
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
        let chunks = self
            .chunks
            .read()
            .map_err(|_| "analysis field chunk store lock poisoned")?;
        let bytes = chunks
            .get(&descriptor.digest)
            .ok_or("render LOD object is missing")?;
        render_tiles::decode_tile(descriptor, bytes).map(Some)
    }

    fn load_render_topology(
        &self,
        run_id: &str,
    ) -> Result<Option<crate::analysis::AnalysisRenderTopology>, String> {
        let Some(index) = self.load_render_tile_index(run_id)? else {
            return Ok(None);
        };
        let chunks = self
            .chunks
            .read()
            .map_err(|_| "analysis field chunk store lock poisoned")?;
        render_tiles::reconstruct(&index, |digest| {
            Ok(chunks.get(&digest).map(|bytes| bytes.to_vec()))
        })
        .map(Some)
    }

    fn load_field_page(
        &self,
        run_id: &str,
        selection: &AnalysisFieldSelection,
        options: AnalysisFieldRequestOptions,
    ) -> Result<AnalysisFieldPageLookup, String> {
        validate_run_id_token(run_id)?;
        let runs = self
            .runs
            .read()
            .map_err(|_| "analysis artifact store lock poisoned".to_string())?;
        let Some(persisted) = runs.get(run_id) else {
            return Ok(AnalysisFieldPageLookup::RunNotFound);
        };
        let chunks = self
            .chunks
            .read()
            .map_err(|_| "analysis field chunk store lock poisoned".to_string())?;
        match persisted {
            PersistedAnalysisArtifact::Standard(persisted) => {
                validate_persisted_run(persisted, run_id)?;
                field_page_from_artifact(persisted, selection, options, |digest| {
                    Ok(chunks.get(&digest).map(|bytes| bytes.to_vec()))
                })
            }
            PersistedAnalysisArtifact::StructuralFrequency(artifact) => {
                frequency::field_page(artifact, selection, options, |digest| {
                    Ok(chunks.get(&digest).map(|bytes| bytes.to_vec()))
                })
            }
            PersistedAnalysisArtifact::AcousticFrequency(artifact) => {
                frequency_sample::field_page(artifact, selection, options, |digest| {
                    Ok(chunks.get(&digest).map(|bytes| bytes.to_vec()))
                })
            }
        }
    }

    fn list_runs(&self) -> Result<Vec<AnalysisRunResult>, String> {
        let guard = self
            .runs
            .read()
            .map_err(|_| "analysis artifact store lock poisoned".to_string())?;
        guard
            .iter()
            .filter_map(|(run_id, stored)| match stored {
                PersistedAnalysisArtifact::Standard(persisted) => Some((run_id, persisted)),
                PersistedAnalysisArtifact::StructuralFrequency(_)
                | PersistedAnalysisArtifact::AcousticFrequency(_) => None,
            })
            .map(|(run_id, persisted)| {
                validate_persisted_run(persisted, run_id)?;
                let chunks = self
                    .chunks
                    .read()
                    .map_err(|_| "analysis field chunk store lock poisoned".to_string())?;
                materialize_run(persisted, |digest| {
                    Ok(chunks.get(&digest).map(|bytes| bytes.to_vec()))
                })
            })
            .collect()
    }

    fn list_run_metadata(&self) -> Result<Vec<AnalysisRunResult>, String> {
        let guard = self
            .runs
            .read()
            .map_err(|_| "analysis artifact store lock poisoned".to_string())?;
        guard
            .iter()
            .filter_map(|(run_id, stored)| match stored {
                PersistedAnalysisArtifact::Standard(persisted) => Some((run_id, persisted)),
                PersistedAnalysisArtifact::StructuralFrequency(_)
                | PersistedAnalysisArtifact::AcousticFrequency(_) => None,
            })
            .map(|(run_id, persisted)| {
                validate_persisted_run(persisted, run_id)?;
                Ok(persisted.metadata.clone())
            })
            .collect()
    }

    fn persist_structural_frequency(
        &self,
        step: &AnalysisStudyStepRunData,
        object_catalog: &runmat_execution::value::PublishedObjectCatalog,
        artifact_roots: &crate::analysis::StructuralFrequencyArtifactRootIndex,
        render_topology: Option<&crate::analysis::AnalysisRenderTopology>,
        source: &dyn runmat_execution::value::PublishedObjectSource,
    ) -> Result<AnalysisArtifactRecord, String> {
        let (prepared, tile_objects) = frequency::prepare_artifact(
            step,
            object_catalog,
            artifact_roots,
            render_topology,
            Utc::now().to_rfc3339(),
        )?;
        let objects = frequency::ingest_objects(&prepared, tile_objects, source)?;
        let mut record = frequency::artifact_record(&prepared);
        record.created_at = self.install_standalone(
            &step.run_id,
            PersistedAnalysisArtifact::StructuralFrequency(Box::new(prepared)),
            objects.into_iter().collect(),
        )?;
        Ok(record)
    }

    fn load_structural_frequency(
        &self,
        run_id: &str,
    ) -> Result<Option<AnalysisStudyStepRunData>, String> {
        validate_run_id_token(run_id)?;
        let runs = self
            .runs
            .read()
            .map_err(|_| "analysis artifact store lock poisoned".to_string())?;
        let Some(PersistedAnalysisArtifact::StructuralFrequency(artifact)) = runs.get(run_id)
        else {
            return Ok(None);
        };
        frequency::validate_artifact(artifact, run_id)?;
        Ok(Some(artifact.step.clone()))
    }

    fn persist_acoustic_frequency(
        &self,
        step: &AnalysisStudyStepRunData,
        object_catalog: &runmat_execution::value::PublishedObjectCatalog,
        artifact_roots: &crate::analysis::FrequencySampleArtifactRootInventory,
        render_topology: Option<&crate::analysis::AnalysisRenderTopology>,
        source: &dyn runmat_execution::value::PublishedObjectSource,
    ) -> Result<AnalysisArtifactRecord, String> {
        let (prepared, tile_objects) = frequency_sample::prepare_artifact(
            step,
            object_catalog,
            artifact_roots,
            render_topology,
            Utc::now().to_rfc3339(),
        )?;
        let objects = frequency_sample::ingest_objects(&prepared, tile_objects, source)?;
        let mut record = frequency_sample::artifact_record(&prepared);
        record.created_at = self.install_standalone(
            &step.run_id,
            PersistedAnalysisArtifact::AcousticFrequency(Box::new(prepared)),
            objects.into_iter().collect(),
        )?;
        Ok(record)
    }

    fn load_acoustic_frequency(
        &self,
        run_id: &str,
    ) -> Result<Option<AnalysisStudyStepRunData>, String> {
        validate_run_id_token(run_id)?;
        let runs = self
            .runs
            .read()
            .map_err(|_| "analysis artifact store lock poisoned".to_string())?;
        let Some(PersistedAnalysisArtifact::AcousticFrequency(artifact)) = runs.get(run_id) else {
            return Ok(None);
        };
        frequency_sample::validate_artifact(artifact, run_id)?;
        Ok(Some(artifact.step.clone()))
    }
}
