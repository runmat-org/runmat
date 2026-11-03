use super::*;
use runmat_analysis_fea::results::field::{RecoveredField, RecoveredFieldArtifactManifest};
impl AnalysisPublicationBuilder {
    /// Attach authenticated original field definitions before the closed publication freezes.
    pub fn stage_original_fields(
        &self,
        run_id: &str,
        catalog: &RecoveredFieldArtifactManifest,
    ) -> Result<(), AnalysisPublicationError> {
        let admission = self
            .admission
            .lock()
            .map_err(|_| "publication staging lock poisoned")?;
        if *admission {
            return Err("original fields cannot attach after publication inventory freezes".into());
        }
        if self
            .store
            .observations
            .read()
            .map_err(|_| "publication observation store poisoned")?
            .contains_key(run_id)
        {
            return Err("original fields cannot change a run bound to observations".into());
        }
        self.admit(
            resident(catalog, self.maximum_bytes)?
                .checked_mul(3)
                .ok_or("original field catalog profile overflow")?,
        )?;
        catalog.validate().map_err(|e| e.to_string())?;
        let peak = {
            let runs = self
                .store
                .runs
                .read()
                .map_err(|_| "publication staging store poisoned")?;
            let Some(PersistedAnalysisArtifact::Standard(run)) = runs.get(run_id) else {
                return Err("original fields require an already staged standard run".into());
            };
            catalog.fields.iter().try_fold(0u64, |peak, entry| {
                let field = run
                    .fields
                    .iter()
                    .find(|f| f.field_id == entry.definition.field_id())
                    .ok_or("original field is absent from staged run")?;
                let bytes =
                    super::super::initial_fields::read::workspace(field, self.maximum_bytes)?;
                Ok::<_, AnalysisPublicationError>(peak.max(bytes))
            })?
        };
        self.admit(
            resident(catalog, self.maximum_bytes)?
                .checked_mul(3)
                .and_then(|bytes| bytes.checked_add(peak))
                .ok_or_else(|| AnalysisPublicationError::ResourceLimit {
                    message: "original field construction overflows allowance".into(),
                })?,
        )?;
        let mut runs = self
            .store
            .runs
            .write()
            .map_err(|_| "publication staging store poisoned")?;
        let Some(PersistedAnalysisArtifact::Standard(run)) = runs.get_mut(run_id) else {
            return Err("original fields require an already staged standard run".into());
        };
        if run.original_fields.is_some() {
            return Err("original field catalog is already attached".into());
        }
        let candidate = catalog
            .fields
            .iter()
            .map(|entry| -> Result<_, String> {
                let field = run
                    .fields
                    .iter()
                    .find(|field| field.field_id == entry.definition.field_id())
                    .ok_or("original field is absent from staged run")?;
                if field.shape
                    != entry
                        .definition
                        .storage_shape(entry.definition.stored_entities(catalog.entity_counts))
                        .map_err(|e| e.to_string())?
                {
                    return Err("original field shape differs from staged run".into());
                }
                Ok((entry, field))
            })
            .collect::<Result<Vec<_>, String>>()?;
        // Read exactly retained original chunks to verify typed identities; no source solve.
        for (entry, field) in candidate {
            let payload = self.read_staged_chunks(|mut read| {
                super::super::field_reader::read_field_payload(field, &mut read)
            })?;
            let values = decode_field_values(&payload)?;
            let recovered = RecoveredField::new(
                entry.definition.clone(),
                entry.sample,
                AnalysisField::host_f64(
                    &field.field_id,
                    entry.definition.layout(),
                    field.shape.clone(),
                    values,
                ),
            )
            .map_err(|e| e.to_string())?;
            if recovered.identity() != entry.field {
                return Err(
                    "original field source identity differs from staged numerical data".into(),
                );
            }
        }
        let run_value = run.run_projection()?.into_owned();
        let digest = persisted_run_digest(
            &run_value,
            &run.field_inventory,
            &run.materialized_field_ids,
            run.render_tiles.as_ref(),
            &run.observation_histories,
            Some(catalog),
            run.dynamic_balance.as_ref(),
            run.observation_payload.as_ref(),
        )?;
        run.original_fields = Some(catalog.clone());
        run.run_digest = digest;
        Ok(())
    }
}
