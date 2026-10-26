use super::*;

pub(super) fn persisted_run_digest(
    value: &serde_json::Value,
    field_inventory: &[AnalysisField],
    materialized_field_ids: &[String],
    render_tiles: Option<&RenderTileIndex>,
    observation_histories: &[PersistedObservationHistory],
    original_fields: Option<&runmat_analysis_fea::results::field::RecoveredFieldArtifactManifest>,
    dynamic_balance: Option<&PersistedDynamicBalance>,
    observation_payload: Option<&AnalysisObjectManifest>,
) -> Result<Digest, String> {
    // A chunked observation payload binds its manifest; records without one keep their identity.
    if let Some(payload) = observation_payload {
        return serde_json::to_vec(&(
            value,
            field_inventory,
            materialized_field_ids,
            render_tiles,
            observation_histories,
            original_fields,
            dynamic_balance,
            payload,
        ))
        .map(Digest::sha256)
        .map_err(|e| e.to_string());
    }
    // A chunked dynamic balance binds its manifest; records without one keep their identity.
    if let Some(balance) = dynamic_balance {
        return serde_json::to_vec(&(
            value,
            field_inventory,
            materialized_field_ids,
            render_tiles,
            observation_histories,
            original_fields,
            balance,
        ))
        .map(Digest::sha256)
        .map_err(|e| e.to_string());
    }
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
