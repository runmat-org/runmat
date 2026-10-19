use super::*;

pub(super) fn persisted_run_digest(
    value: &serde_json::Value,
    field_inventory: &[AnalysisField],
    materialized_field_ids: &[String],
    render_tiles: Option<&RenderTileIndex>,
    observation_histories: &[PersistedObservationHistory],
    original_fields: Option<&runmat_analysis_fea::results::field::RecoveredFieldArtifactManifest>,
) -> Result<Digest, String> {
    if let Some(fields) = original_fields {
        return serde_json::to_vec(&(
            value,
            field_inventory,
            materialized_field_ids,
            render_tiles,
            observation_histories,
            fields,
        ))
        .map(Digest::sha256)
        .map_err(|e| e.to_string());
    }
    if !observation_histories.is_empty() {
        return serde_json::to_vec(&(
            value,
            field_inventory,
            materialized_field_ids,
            render_tiles,
            observation_histories,
        ))
        .map(Digest::sha256)
        .map_err(|error| error.to_string());
    }
    serde_json::to_vec(&(value, field_inventory, materialized_field_ids, render_tiles))
        .map(Digest::sha256)
        .map_err(|error| format!("failed to encode analysis run metadata: {error}"))
}
