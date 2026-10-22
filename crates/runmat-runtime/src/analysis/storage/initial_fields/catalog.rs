use super::*;
pub(in crate::analysis::storage) fn validate_catalog(
    run: &PersistedRunArtifact,
) -> Result<(), String> {
    let Some(catalog) = &run.original_fields else {
        return Ok(());
    };
    catalog.validate().map_err(|e| e.to_string())?;
    for entry in &catalog.fields {
        let fields = run
            .fields
            .iter()
            .filter(|f| f.field_id == entry.definition.field_id())
            .collect::<Vec<_>>();
        let [field] = fields.as_slice() else {
            return Err("original field has no unique persisted payload".into());
        };
        let count = entry.definition.stored_entities(catalog.entity_counts);
        if field.shape
            != entry
                .definition
                .storage_shape(count)
                .map_err(|e| e.to_string())?
        {
            return Err("original field catalog differs from stored shape".into());
        }
    }
    Ok(())
}

pub(super) struct OriginalFieldSource<'a> {
    pub catalog: &'a RecoveredFieldArtifactManifest,
    pub entry: &'a RecoveredFieldArtifactEntry,
    pub field: &'a PersistedFieldArtifact,
}

pub(super) fn select<'a>(
    run: &'a PersistedRunArtifact,
    field_id: &str,
) -> Result<Option<OriginalFieldSource<'a>>, AnalysisPublicationError> {
    let Some(catalog) = &run.original_fields else {
        return Ok(None);
    };
    let mut selected = catalog
        .fields
        .iter()
        .filter(|e| e.definition.field_id() == field_id);
    let Some(entry) = selected.next() else {
        return Ok(None);
    };
    if selected.next().is_some() {
        return Err("initial field requires an explicit unique source sample".into());
    }
    let field = run
        .fields
        .iter()
        .find(|f| f.field_id == field_id)
        .ok_or("original field payload missing")?;
    Ok(Some(OriginalFieldSource {
        catalog,
        entry,
        field,
    }))
}
