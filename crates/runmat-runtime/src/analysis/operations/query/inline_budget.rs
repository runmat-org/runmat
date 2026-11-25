//! The inline field-value budget of a default results view.
use super::*;

/// Field payload a default results object holds inline; larger runs page values through
/// `fea.field`.
pub const RESULTS_INLINE_FIELD_BYTES: u64 = 16 * 1024 * 1024;

/// The default results view of a stored run: values of the selected fields inline when they fit
/// [`RESULTS_INLINE_FIELD_BYTES`], descriptors only otherwise. Only a caller that chose no
/// field-value option applies this; an explicit field-value request is never narrowed.
pub fn bounded_default_results_query(
    run_id: &str,
    mut query: AnalysisResultsQuery,
) -> Result<AnalysisResultsQuery, String> {
    if !query.include_field_values {
        return Ok(query);
    }
    let run_id = storage::resolve_run_attempt(run_id)?;
    let Some(inventory) = storage::load_field_inventory(&run_id)? else {
        return Ok(query);
    };
    let payload = inventory
        .iter()
        .filter(|field| {
            query.include_fields.is_empty() || query.include_fields.contains(&field.field_id)
        })
        .map(|field| {
            field
                .shape
                .iter()
                .try_fold(8u64, |bytes, extent| bytes.checked_mul(*extent as u64))
                .unwrap_or(u64::MAX)
        })
        .fold(0u64, u64::saturating_add);
    if payload > RESULTS_INLINE_FIELD_BYTES {
        query.include_field_values = false;
    }
    Ok(query)
}
