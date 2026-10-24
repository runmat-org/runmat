//! Materialize or page original numerical objects through their shared chunk manifests.
use super::*;

pub(super) fn materialize_run(
    persisted: &PersistedRunArtifact,
    mut read_chunk: impl FnMut(Digest) -> Result<Option<Vec<u8>>, String>,
) -> Result<AnalysisRunResult, String> {
    let mut values = HashMap::with_capacity(persisted.fields.len());
    for field in &persisted.fields {
        let payload = read_field_payload(field, &mut read_chunk)?;
        values.insert(
            field.field_id.clone(),
            (field.shape.clone(), decode_field_values(&payload)?),
        );
    }
    let mut run = persisted.run.clone();
    hydrate_field_values(&mut run, &values)?;
    let mut run = serde_json::from_value::<AnalysisRunResult>(run)
        .map_err(|error| format!("failed to decode hydrated analysis run: {error}"))?;
    for history in &persisted.observation_histories {
        let payload = read_numerical_payload(&history.manifest, &mut read_chunk)?;
        let restored =
            runmat_analysis_fea::results::observation::ObservationHistory::from_canonical_bytes(
                &payload,
            )
            .map_err(|error| error.to_string())?;
        if restored.manifest().identity != history.identity
            || restored.manifest().observation_id != history.observation_id
        {
            return Err("observation history differs from its authenticated inventory".into());
        }
        run.run.observation_histories.push(restored);
    }
    if let Some(balance) = &persisted.dynamic_balance {
        let payload = read_numerical_payload(&balance.manifest, &mut read_chunk)?;
        let values = decode_field_values(&payload)?;
        let transient = run
            .transient_results
            .as_mut()
            .ok_or("dynamic balance belongs to a run without transient results")?;
        if transient.dynamic_balance_snapshots.len() != balance.dof_counts.len() {
            return Err("dynamic balance differs from its accepted states".into());
        }
        let mut offset = 0usize;
        for (snapshot, &dofs) in transient
            .dynamic_balance_snapshots
            .iter_mut()
            .zip(&balance.dof_counts)
        {
            let dofs = usize::try_from(dofs).map_err(|_| "dynamic balance size overflow")?;
            let mut take = || -> Result<Vec<f64>, String> {
                let end = offset
                    .checked_add(dofs)
                    .filter(|end| *end <= values.len())
                    .ok_or("dynamic balance payload is shorter than its inventory")?;
                let vector = values[offset..end].to_vec();
                offset = end;
                Ok(vector)
            };
            *snapshot = runmat_analysis_fea::StructuralDynamicBalanceSnapshot {
                applied_load: take()?,
                inertia_force: take()?,
                damping_force: take()?,
                elastic_force: take()?,
                equilibrium_residual: take()?,
            };
        }
        if offset != values.len() {
            return Err("dynamic balance payload is longer than its inventory".into());
        }
    }
    if let Some(index) = &persisted.render_tiles {
        run.render_topology = Some(render_tiles::reconstruct(index, |digest| {
            read_chunk(digest)
        })?);
    }
    validate_run_identity(&run)?;
    Ok(run)
}

fn hydrate_field_values(
    value: &mut serde_json::Value,
    fields: &HashMap<String, (Vec<usize>, Vec<f64>)>,
) -> Result<(), String> {
    match value {
        serde_json::Value::Array(values) => {
            for value in values {
                hydrate_field_values(value, fields)?;
            }
        }
        serde_json::Value::Object(object) => {
            let field_id = object
                .get("field_id")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned);
            if let Some(field_id) = field_id {
                if let Some((shape, values)) = fields.get(&field_id) {
                    let stored_shape = object
                        .get("shape")
                        .cloned()
                        .ok_or_else(|| format!("analysis field `{field_id}` omits its shape"))?;
                    if stored_shape
                        != serde_json::to_value(shape).map_err(|error| error.to_string())?
                    {
                        return Err(format!(
                            "analysis field `{field_id}` shape differs from its persisted manifest"
                        ));
                    }
                    object.insert(
                        "values".to_string(),
                        serde_json::to_value(AnalysisFieldValues::HostF64(values.clone()))
                            .map_err(|error| error.to_string())?,
                    );
                    return Ok(());
                }
            }
            for value in object.values_mut() {
                hydrate_field_values(value, fields)?;
            }
        }
        _ => {}
    }
    Ok(())
}

pub(in crate::analysis::storage) fn read_field_payload(
    field: &PersistedFieldArtifact,
    read_chunk: &mut impl FnMut(Digest) -> Result<Option<Vec<u8>>, String>,
) -> Result<Vec<u8>, String> {
    read_numerical_payload(&field.manifest, read_chunk)
}

fn read_numerical_payload(
    manifest: &AnalysisObjectManifest,
    read_chunk: &mut impl FnMut(Digest) -> Result<Option<Vec<u8>>, String>,
) -> Result<Vec<u8>, String> {
    manifest.validate().map_err(|error| error.to_string())?;
    let capacity = usize::try_from(manifest.encoded_length)
        .map_err(|_| "analysis field byte length is not representable".to_string())?;
    let mut payload = Vec::with_capacity(capacity);
    for descriptor in &manifest.chunks {
        let bytes = read_chunk(descriptor.digest)?.ok_or_else(|| {
            format!(
                "analysis numerical object is missing chunk {}",
                descriptor.ordinal
            )
        })?;
        validate_chunk(descriptor, &bytes)?;
        payload.extend_from_slice(&bytes);
    }
    if payload.len() as u64 != manifest.encoded_length
        || Digest::sha256(&payload) != manifest.payload_digest
    {
        return Err("analysis numerical object payload differs from its manifest".into());
    }
    Ok(payload)
}

pub(super) fn field_page_from_artifact(
    persisted: &PersistedRunArtifact,
    selection: &AnalysisFieldSelection,
    options: AnalysisFieldRequestOptions,
    mut read_chunk: impl FnMut(Digest) -> Result<Option<Vec<u8>>, String>,
) -> Result<AnalysisFieldPageLookup, String> {
    let field_id = &selection.field_id;
    validate_field_id(field_id)?;
    if selection.sample.is_some() {
        return Ok(AnalysisFieldPageLookup::SampleNotFound);
    }
    let template = match select_persisted_field(&persisted.fields, field_id) {
        Ok(Some(template)) => template,
        Ok(None) => return Ok(AnalysisFieldPageLookup::FieldNotFound),
        Err(available) => return Ok(AnalysisFieldPageLookup::FieldSelectionAmbiguous(available)),
    };
    let selected_field_id = template.field_id.clone();
    let mut descriptor = AnalysisFieldDescriptor::from_field(&template);
    if descriptor.storage == crate::analysis::contracts::AnalysisFieldStorage::HostF64 {
        descriptor.storage_ref.kind = "chunked_field".to_string();
    }
    let total_count = checked_element_count(&template.shape)?;
    let offset = options.offset.unwrap_or(0).min(total_count);
    let limit = options
        .limit
        .unwrap_or(crate::analysis::contracts::ANALYSIS_FIELD_DEFAULT_MATERIALIZE_LIMIT)
        .min(crate::analysis::contracts::ANALYSIS_FIELD_DEFAULT_PAGE_SIZE);
    let count = limit.min(total_count.saturating_sub(offset));
    if count == 0 {
        return Ok(AnalysisFieldPageLookup::Page(Box::new(
            AnalysisFieldPageResult {
                run_id: persisted.metadata.run_id.clone(),
                field_id: selected_field_id,
                descriptor,
                values: AnalysisFieldPageValues::Real(Vec::new()),
                offset,
                count: 0,
                total_count,
                truncated: offset > 0 || total_count > 0,
            },
        )));
    }
    let Some(stored) = persisted
        .fields
        .iter()
        .find(|field| field.field_id == selected_field_id)
    else {
        return Err(format!(
            "analysis field `{field_id}` is device-resident and has no materialized host artifact"
        ));
    };
    let start_byte = offset
        .checked_mul(std::mem::size_of::<f64>())
        .ok_or_else(|| "analysis field page byte offset overflow".to_string())?;
    let end_byte = offset
        .checked_add(count)
        .and_then(|end| end.checked_mul(std::mem::size_of::<f64>()))
        .ok_or_else(|| "analysis field page byte end overflow".to_string())?;
    let mut page = Vec::with_capacity(count * std::mem::size_of::<f64>());
    let mut chunk_start = 0_usize;
    for chunk in &stored.manifest.chunks {
        let chunk_len = usize::try_from(chunk.encoded_length)
            .map_err(|_| "analysis field chunk length is not representable".to_string())?;
        let chunk_end = chunk_start
            .checked_add(chunk_len)
            .ok_or_else(|| "analysis field chunk range overflow".to_string())?;
        let overlap_start = start_byte.max(chunk_start);
        let overlap_end = end_byte.min(chunk_end);
        if overlap_start < overlap_end {
            let bytes = read_chunk(chunk.digest)?.ok_or_else(|| {
                format!(
                    "analysis field `{field_id}` is missing chunk {}",
                    chunk.ordinal
                )
            })?;
            validate_chunk(chunk, &bytes)?;
            page.extend_from_slice(&bytes[overlap_start - chunk_start..overlap_end - chunk_start]);
        }
        chunk_start = chunk_end;
        if chunk_start >= end_byte {
            break;
        }
    }
    let values = decode_field_values(&page)?;
    if values.len() != count {
        return Err(format!(
            "analysis field `{field_id}` page is incomplete: expected {count} values, found {}",
            values.len()
        ));
    }
    Ok(AnalysisFieldPageLookup::Page(Box::new(
        AnalysisFieldPageResult {
            run_id: persisted.metadata.run_id.clone(),
            field_id: selected_field_id,
            descriptor,
            values: AnalysisFieldPageValues::Real(values),
            offset,
            count,
            total_count,
            truncated: offset > 0 || offset.saturating_add(count) < total_count,
        },
    )))
}

fn select_persisted_field(
    fields: &[PersistedFieldArtifact],
    requested: &str,
) -> Result<Option<AnalysisField>, Vec<String>> {
    if let Some(field) = fields.iter().find(|field| field.field_id == requested) {
        return Ok(Some(field.template.clone()));
    }
    let mut matches = fields.iter().filter(|field| {
        crate::analysis::field_selection::field_id_matches(&field.field_id, requested)
    });
    let Some(first) = matches.next() else {
        return Ok(None);
    };
    if matches.next().is_none() {
        return Ok(Some(first.template.clone()));
    }
    Err(fields
        .iter()
        .filter(|field| {
            crate::analysis::field_selection::field_id_matches(&field.field_id, requested)
        })
        .map(|field| field.field_id.clone())
        .collect())
}

pub(super) fn persisted_field_inventory(persisted: &PersistedRunArtifact) -> Vec<AnalysisField> {
    persisted.field_inventory.clone()
}
