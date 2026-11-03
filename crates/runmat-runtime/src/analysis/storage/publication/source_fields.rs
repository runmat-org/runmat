//! Construct original source catalogs from admitted, already staged numerical chunks.
use super::*;
use runmat_analysis_fea::results::field::{
    OriginalFieldSourcePlan, RecoveredField, RecoveredFieldArtifactEntry,
    RecoveredFieldArtifactManifest, RecoveredFieldView,
};

impl AnalysisPublicationBuilder {
    pub fn stage_original_field_source(
        &self,
        run_id: &str,
        source: &OriginalFieldSourcePlan,
    ) -> Result<(), AnalysisPublicationError> {
        let catalog = {
            let admission = self
                .admission
                .lock()
                .map_err(|_| "publication staging lock poisoned")?;
            if *admission {
                return Err(
                    "original fields cannot attach after publication inventory freezes".into(),
                );
            }
            self.admit(
                resident(source, self.maximum_bytes)?
                    .checked_mul(3)
                    .ok_or("original source metadata profile overflow")?,
            )?;
            source.validate().map_err(|error| error.to_string())?;
            let runs = self
                .store
                .runs
                .read()
                .map_err(|_| "publication staging store poisoned")?;
            let Some(PersistedAnalysisArtifact::Standard(run)) = runs.get(run_id) else {
                return Err("original field source requires an already staged standard run".into());
            };
            let definitions = source
                .fields
                .iter()
                .map(|entry| entry.definition.clone())
                .collect::<Vec<_>>();
            let counts = definitions
                .iter()
                .map(|definition| definition.stored_entities(source.entity_counts))
                .collect::<Vec<_>>();
            // Every producer in this vertical uses one sample variant; profiles bound scalar widths.
            let sample = source.fields[0].sample;
            if source.fields.iter().any(|entry| {
                std::mem::discriminant(&entry.sample) != std::mem::discriminant(&sample)
            }) {
                return Err("original field source mixes sample representations".into());
            }
            let metadata = RecoveredFieldArtifactManifest::profile_sampled_shape(
                &source.request,
                &definitions,
                &counts,
                sample,
            )
            .map_err(|error| error.to_string())?
            .materialization_resident_bytes_estimate;
            let peak = source.fields.iter().try_fold(0u64, |peak, entry| {
                let field = run
                    .fields
                    .iter()
                    .find(|field| field.field_id == entry.definition.field_id())
                    .ok_or("original source channel is absent from staged data")?;
                if field.shape
                    != entry
                        .definition
                        .storage_shape(entry.definition.stored_entities(source.entity_counts))
                        .map_err(|error| error.to_string())?
                {
                    return Err("original source shape differs from staged data".into());
                }
                Ok::<_, AnalysisPublicationError>(peak.max(
                    super::super::initial_fields::read::workspace(field, self.maximum_bytes)?,
                ))
            })?;
            self.admit(
                metadata
                    .checked_mul(3)
                    .and_then(|bytes| bytes.checked_add(peak))
                    .ok_or("original source construction profile overflow")?,
            )?;
            let mut entries = Vec::with_capacity(source.fields.len());
            for entry in &source.fields {
                let field = run
                    .fields
                    .iter()
                    .find(|field| field.field_id == entry.definition.field_id())
                    .ok_or("original source channel disappeared")?;
                let payload = self.read_staged_chunks(|mut read| {
                    super::super::field_reader::read_field_payload(field, &mut read)
                })?;
                let recovered = RecoveredField::new(
                    entry.definition.clone(),
                    entry.sample,
                    AnalysisField::host_f64(
                        &field.field_id, entry.definition.layout(),
                        field.shape.clone(),
                        decode_field_values(&payload)?,
                    ),
                )
                .map_err(|error| error.to_string())?;
                let view = RecoveredFieldView::from_field(&recovered)
                    .map_err(|error| error.to_string())?;
                entries.push(RecoveredFieldArtifactEntry {
                    field: recovered.identity(),
                    definition: entry.definition.clone(),
                    sample: entry.sample,
                    partitions: view
                        .canonical_partition_descriptors()
                        .map_err(|error| error.to_string())?,
                });
            }
            RecoveredFieldArtifactManifest::from_entries(
                source.request.identity(),
                source.request.source,
                source.request.topology.discretization_digest(),
                source.entity_counts,
                entries,
            )
            .map_err(|error| error.to_string())?
        };
        self.stage_original_fields(run_id, &catalog)
    }
}
