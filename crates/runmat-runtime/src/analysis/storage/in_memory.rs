//! In-memory artifact backend, including exact frequency publication storage.
use super::*;

impl AnalysisArtifactStore for InMemoryAnalysisArtifactStore {
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
        let mut guard = self
            .runs
            .write()
            .map_err(|_| "analysis artifact store lock poisoned".to_string())?;
        if let Some(existing) = guard.get(&run.run_id) {
            let PersistedAnalysisArtifact::Standard(existing) = existing else {
                return Err("analysis run identity already contains another result kind".into());
            };
            validate_persisted_run(existing, &run.run_id)?;
            if !same_persisted_content(existing, &prepared.persisted) {
                return Err("analysis run identity already contains different content".into());
            }
            return artifact_record(run, existing.created_at.clone());
        }
        let created_at = prepared.persisted.created_at.clone();
        let mut chunks = self
            .chunks
            .write()
            .map_err(|_| "analysis field chunk store lock poisoned".to_string())?;
        for (digest, bytes) in prepared.chunks {
            match chunks.get(&digest) {
                Some(existing) if existing.as_ref() != bytes.as_slice() => {
                    return Err("analysis field chunk digest collision".to_string());
                }
                Some(_) => {}
                None => {
                    chunks.insert(digest, Arc::from(bytes));
                }
            }
        }
        guard.insert(
            run.run_id.clone(),
            PersistedAnalysisArtifact::Standard(Box::new(prepared.persisted)),
        );
        artifact_record(run, created_at)
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
        let Some(PersistedAnalysisArtifact::Standard(run)) = runs.get(run_id) else {
            return Ok(None);
        };
        Ok(run.render_tiles.clone())
    }

    fn plan_render_view(
        &self,
        run_id: &str,
        view: Option<RenderViewRequest>,
    ) -> Result<Option<RenderViewSelection>, String> {
        validate_run_id_token(run_id)?;
        let runs = self
            .runs
            .read()
            .map_err(|_| "analysis artifact store lock poisoned")?;
        let Some(PersistedAnalysisArtifact::Standard(run)) = runs.get(run_id) else {
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
        validate_run_id_token(run_id)?;
        let runs = self
            .runs
            .read()
            .map_err(|_| "analysis artifact store lock poisoned")?;
        let Some(PersistedAnalysisArtifact::Standard(run)) = runs.get(run_id) else {
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
        validate_run_id_token(run_id)?;
        let runs = self
            .runs
            .read()
            .map_err(|_| "analysis artifact store lock poisoned")?;
        let Some(PersistedAnalysisArtifact::Standard(run)) = runs.get(run_id) else {
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
        source: &dyn runmat_execution::value::PublishedObjectSource,
    ) -> Result<AnalysisArtifactRecord, String> {
        let prepared = frequency::prepare_artifact(
            step,
            object_catalog,
            artifact_roots,
            Utc::now().to_rfc3339(),
        )?;
        let objects = frequency::ingest_objects(&prepared, source)?;
        let mut runs = self
            .runs
            .write()
            .map_err(|_| "analysis artifact store lock poisoned".to_string())?;
        if let Some(existing) = runs.get(&step.run_id) {
            let PersistedAnalysisArtifact::StructuralFrequency(existing) = existing else {
                return Err("analysis run identity already contains another result kind".into());
            };
            frequency::validate_artifact(existing, &step.run_id)?;
            if !frequency::same_content(existing, &prepared) {
                return Err("analysis run identity already contains different content".into());
            }
            return Ok(frequency::artifact_record(existing));
        }
        let mut stored_objects = self
            .chunks
            .write()
            .map_err(|_| "analysis object store lock poisoned".to_string())?;
        for (digest, bytes) in objects {
            match stored_objects.get(&digest) {
                Some(existing) if existing.as_ref() != bytes.as_slice() => {
                    return Err("analysis object digest collision".into());
                }
                Some(_) => {}
                None => {
                    stored_objects.insert(digest, Arc::from(bytes));
                }
            }
        }
        let record = frequency::artifact_record(&prepared);
        runs.insert(
            step.run_id.clone(),
            PersistedAnalysisArtifact::StructuralFrequency(Box::new(prepared)),
        );
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
        source: &dyn runmat_execution::value::PublishedObjectSource,
    ) -> Result<AnalysisArtifactRecord, String> {
        let prepared = frequency_sample::prepare_artifact(
            step,
            object_catalog,
            artifact_roots,
            Utc::now().to_rfc3339(),
        )?;
        let objects = frequency_sample::ingest_objects(&prepared, source)?;
        let mut runs = self
            .runs
            .write()
            .map_err(|_| "analysis artifact store lock poisoned".to_string())?;
        if let Some(existing) = runs.get(&step.run_id) {
            let PersistedAnalysisArtifact::AcousticFrequency(existing) = existing else {
                return Err("analysis run identity already contains another result kind".into());
            };
            frequency_sample::validate_artifact(existing, &step.run_id)?;
            if !frequency_sample::same_content(existing, &prepared) {
                return Err("analysis run identity already contains different content".into());
            }
            return Ok(frequency_sample::artifact_record(existing));
        }
        let mut stored_objects = self
            .chunks
            .write()
            .map_err(|_| "analysis object store lock poisoned".to_string())?;
        for (digest, bytes) in objects {
            match stored_objects.get(&digest) {
                Some(existing) if existing.as_ref() != bytes.as_slice() => {
                    return Err("analysis object digest collision".into());
                }
                Some(_) => {}
                None => {
                    stored_objects.insert(digest, Arc::from(bytes));
                }
            }
        }
        let record = frequency_sample::artifact_record(&prepared);
        runs.insert(
            step.run_id.clone(),
            PersistedAnalysisArtifact::AcousticFrequency(Box::new(prepared)),
        );
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
